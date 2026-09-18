"""
05_train.py
--------------
STEP 3 of the pipeline.  Runs after 04_build_dataset.py.

Trains one detector on a synthetic dataset and records the synthetic
validation summary the paper reports, including the confidence that maximises
F1 on the synthetic split. That threshold is what a practitioner would carry
into deployment, so it is the quantity the real-image evaluation is compared
against.

Every architecture in the comparison trains through this one script with the
schedule in TRAIN_CFG below, which is the only place any of it is written
down. Nothing is per-model. If two arms were allowed to differ in epochs,
batch size, augmentation or seed, a difference between them would no longer be
attributable to the architecture, which is the entire claim the comparison
makes.

Run:   python 05_train.py --pool 59 --model yolov8s.pt
       python 05_train.py --pool 59 --model models/yolov8s-cbam.yaml --tag v8s_cbam
Output: runs/detect/pool<N>[_<tag>]/  (+ synthetic_summary.json)
"""

from pathlib import Path
import argparse
import json
import time

from ultralytics import YOLO

import lib_modules  # noqa: F401  binds CBAM and BiFPNFuse for the YAML parser
from lib_metrics import count_gflops, count_parameters, weight_size_mb
from lib_arms import DEFAULT_POOL, run_name
from pipeline_common import best_f1_point, write_f1_curve

# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# The shared schedule. Batch 16 was chosen by measuring peak VRAM for every
# architecture in the comparison: the heaviest reaches 7.6 GiB of an 8 GiB card
# at batch 32, which leaves nothing for the desktop and has already cost one
# run an out-of-memory failure at epoch 90. Ultralytics accumulates gradients
# to a nominal batch of 64, so 16 and 32 perform identical optimisation and
# only the memory ceiling differs.
TRAIN_CFG = dict(
    # A ceiling, not a schedule: patience below is what is meant to end a run.
    # At 120, only 2 of 11 arms ever reached the patience criterion; three
    # were still improving at the cap (best epoch 119 or 120, one gaining
    # +0.021 mAP50-95 over its last ten epochs), so the budget, not
    # convergence, decided where they stopped. That made "weak architecture"
    # and "stopped early" indistinguishable for exactly the arm that looked
    # weakest. 250 leaves every arm room for patience to fire first.
    epochs=250,
    imgsz=640,
    batch=16,
    patience=30,
    seed=0,
    deterministic=True,
    device=0,
    # Both of these are set low deliberately, and both were measured.
    #
    # cache="ram" holds the decoded dataset on the dataset object, and Windows
    # dataloader workers use spawn, which pickles that whole object into every
    # worker: 1.84 GB times the worker count. It exhausted 32 GB and killed a
    # run outright, and it was also slower -- 36.8 s an epoch against 13.5 s
    # without it, because the pickling cost more than the decode it avoided.
    #
    # Worker count matters for the same reason. Each spawned worker is a fresh
    # interpreter that re-imports torch and initialises CUDA host-side, costing
    # 2-3 GB of private memory that fork would have shared. Measured peak host
    # RAM for one run: 25.1 GB at 8 workers, 17.7 GB at 4, 12.5 GB at 2. The
    # GPU saturates at about 5.6 it/s regardless, so 2 workers cost 5% of epoch
    # time (14.2 s against 13.5 s) and buy back half the memory.
    cache=False,
    workers=2,
    # augmentation tuned for small, partly buried objects
    mosaic=1.0,
    close_mosaic=15,
    scale=0.5,
    translate=0.2,
    fliplr=0.5,
    flipud=0.3,
    degrees=15.0,
    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4,
    erasing=0.3,
    copy_paste=0.0,       # the data is already built by compositing
)

# Which COCO checkpoint a custom YAML inherits from. Only the layers whose
# index and shape still match are transferred, so a config that keeps the stock
# backbone and neck starts from pretrained weights everywhere except the
# detection head, which is rebuilt for one class in any case.
PRETRAINED = {"yolov8s": "yolov8s.pt", "yolo11s": "yolo11s.pt"}

# How many leading layers of each architecture arrive already trained, and are
# therefore frozen during the warm-up phase. One rule applied to every arm:
# freeze whatever inherited pretrained weights, let the new layers settle
# against them, then release everything.
#
# The counts are not guesses. They were read off the checkpoint transfer: the
# COCO weights match layers 0-21 of YOLOv8s and 0-22 of YOLO11s, and the CBAM
# variants keep those same indices because the attention blocks are appended
# rather than inserted. Replacing the neck with BiFPN leaves only the YOLO11s
# backbone, 0-10. The ResNet configurations carry ImageNet weights inside the
# single TorchVision layer at index 0.
WARMUP_FREEZE = {
    "yolov8s": 22,
    "yolo11s": 23,
    "yolov8s-cbam": 22,
    "yolo11s-cbam": 23,
    "yolo11s-bifpn-cbam": 11,
    # Every torchvision backbone arrives as a single TVBackbone layer at index
    # 0, so one frozen layer is the whole pretrained part of those networks.
    "resnet18-fpn-cbam": 1,
    "resnet18-bifpn-cbam": 1,
    "googlenet-fpn-cbam": 1,
    "googlenet-bifpn-cbam": 1,
    "efficientnet-fpn-cbam": 1,
    "efficientnet-bifpn-cbam": 1,
}


def freeze_depth(model_arg):
    """Leading layers to hold fixed during warm-up, or 0 if nothing is pretrained."""
    stem = Path(model_arg).stem
    return WARMUP_FREEZE.get(stem, 0)


def verify_transfer(model, weights, depth):
    """
    Stop if any layer about to be frozen did not actually receive pretrained
    weights.

    Ultralytics transfers by matching parameter name and shape, and when
    nothing matches it logs a count and carries on. A renamed layer or a
    changed width would therefore produce a network whose "pretrained" backbone
    is random -- and the warm-up would then freeze that random backbone in
    place for ten epochs. The invariant checked here is the one the warm-up
    relies on: every tensor in layers 0..depth-1 equals the checkpoint's.
    """
    import torch

    if depth <= 0:
        return
    src = YOLO(weights).model.state_dict()
    dst = model.model.state_dict()
    frozen = [k for k in dst if int(k.split(".")[1]) < depth]
    bad = [k for k in frozen
           if k not in src or src[k].shape != dst[k].shape
           or not torch.equal(src[k].cpu(), dst[k].cpu())]
    if not frozen or bad:
        raise SystemExit(
            f"[!] pretrained transfer from {weights} did not cover the layers "
            f"warm-up freezes (0-{depth - 1}): {len(bad)} of {len(frozen)} "
            f"tensors missing or different, e.g. {bad[:3]}")
    print(f"[+] verified: all {len(frozen)} tensors in layers 0-{depth - 1} "
          f"came from {weights}")


def resolve_weights(model_arg, override):
    if override:
        return override
    if not model_arg.endswith(".yaml"):
        return None
    stem = Path(model_arg).stem
    for key, ckpt in PRETRAINED.items():
        if stem.startswith(key):
            return ckpt
    return None      # ResNet configs carry ImageNet weights inside the YAML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    ap.add_argument("--model", type=str, default="yolov8s.pt",
                    help="a checkpoint (yolov8s.pt) or an architecture from "
                         "models/ (models/yolov8s-cbam.yaml)")
    ap.add_argument("--weights", type=str, default="",
                    help="checkpoint to transfer into a --model YAML; inferred "
                         "from the config name when omitted")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the run directory, used by the "
                         "architecture comparison")
    ap.add_argument("--warmup-epochs", type=int, default=10,
                    help="epochs spent training only the newly initialised "
                         "layers with the pretrained ones frozen, before the "
                         "full run. These are additional to --epochs, not taken "
                         "out of it. 0 disables.")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from weights/last.pt in "
                         "its run directory, exactly where it stopped -- same "
                         "optimizer state, same epoch, same LR schedule")
    ap.add_argument("--workers", type=int, default=TRAIN_CFG["workers"],
                    help="dataloader workers. Changes speed and host RAM, and "
                         "also the augmentation stream: each worker seeds its "
                         "own RNG, so the count changes which augmented images "
                         "the model sees even at a fixed --seed. Hold it "
                         "constant across arms being compared.")
    ap.add_argument("--seed", type=int, default=TRAIN_CFG["seed"],
                    help="training seed; the dataset is held fixed, so varying "
                         "this isolates run-to-run variance from initialisation, "
                         "dataloader order and augmentation draws")
    args = ap.parse_args()

    data = ROOT / f"dataset_pool{args.pool}" / "data.yaml"
    if not data.exists():
        print(f"[!] {data} not found. Run 04_build_dataset.py --pool {args.pool} first.")
        return

    name = run_name(args.pool, args.tag, args.seed)
    detect_dir = ROOT / "runs" / "detect"
    warm_name = f"{name}_warmup"

    # Refuse rather than overwrite or rename. Ultralytics does not fail when a
    # run directory already exists: it quietly trains into "<name>-2", while
    # every script downstream still reads "<name>". That produced a run whose
    # summary and threshold came from one training and whose weights came from
    # another. Starting again means moving the old directories out first.
    if not args.resume:
        clash = [d for d in (detect_dir / name, detect_dir / warm_name) if d.exists()]
        if clash:
            print("[!] refusing to start: already exists")
            for d in clash:
                print(f"      {d.relative_to(ROOT).as_posix()}")
            print("    move or delete it to retrain, or pass --resume to continue it.")
            return

    # Built unconditionally: the summary below reports batch/imgsz regardless
    # of which branch trained, and a resumed run never rebuilds this dict.
    cfg = dict(TRAIN_CFG, epochs=args.epochs, seed=args.seed, workers=args.workers)

    if args.resume:
        last = detect_dir / name / "weights" / "last.pt"
        if not last.exists():
            print(f"[!] no checkpoint to resume: {last}")
            return
        weights = None
        model = YOLO(str(last))
        t0 = time.time()
        # resume=True restores data, epochs, optimizer and LR schedule from the
        # checkpoint itself -- passing them again would be ignored at best and
        # contradictory at worst, so nothing else from TRAIN_CFG is passed.
        model.train(resume=True)
        train_seconds = time.time() - t0
        print("[+] resumed from " + last.name + f"; {train_seconds:.0f}s covers "
              "only the segment run just now, not the interrupted portion")
    else:
        model = YOLO(args.model)
        weights = resolve_weights(args.model, args.weights)
        depth = freeze_depth(args.model)
        if weights:
            print(f"[+] transferring matching layers from {weights}")
            model.load(weights)
            verify_transfer(model, weights, depth)

        t0 = time.time()

        if args.warmup_epochs > 0 and depth > 0:
            # Phase one. The new layers start from random values and sit against
            # a converged backbone; letting them move while everything else is
            # held still stops that noise from being back-propagated into
            # weights that were already right.
            print(f"[+] warm-up: {args.warmup_epochs} epochs with layers "
                  f"0-{depth - 1} frozen")
            model.train(data=str(data), name=warm_name,
                        **dict(cfg, epochs=args.warmup_epochs, freeze=depth))
            # Where Ultralytics actually wrote it, not where it was asked to.
            warmed = Path(model.trainer.save_dir) / "weights" / "last.pt"
            # last.pt, not best.pt: warm-up is initialisation, not model
            # selection, and the best epoch of a frozen run is not meaningful.
            model = YOLO(str(warmed))
            print(f"[+] warm-up complete, continuing from {warmed}")

        model.train(data=str(data), name=name, **cfg)
        train_seconds = time.time() - t0

    # Every output below goes where training actually wrote, read back from the
    # trainer, so the summary can never land beside a different run's weights.
    out_dir = Path(model.trainer.save_dir)
    if out_dir != detect_dir / name:
        print(f"[!] trained into {out_dir.name}, expected {name}; "
              "downstream scripts will not find it")

    print("\n--- validation on the SYNTHETIC split ---")
    metrics = model.val()
    conf, f1 = best_f1_point(metrics)

    write_f1_curve(out_dir, metrics)

    best = out_dir / "weights" / "best.pt"
    scored = YOLO(str(best)).model

    summary = {
        "pool": args.pool,
        "seed": args.seed,
        "model": args.model,
        "pretrained_from": weights,
        "epochs": args.epochs,
        "batch": cfg["batch"],
        "warmup_epochs": args.warmup_epochs if freeze_depth(args.model) else 0,
        "warmup_frozen_layers": freeze_depth(args.model),
        "imgsz": cfg["imgsz"],
        "dataset": data.parent.name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "best_f1": f1,
        "best_f1_conf": conf,
        "n_params": count_parameters(scored),
        "gflops": count_gflops(scored, cfg["imgsz"]),
        "model_size_mb": weight_size_mb(best),
        "train_seconds": round(train_seconds, 1),
    }
    (out_dir / "synthetic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSynthetic validation summary")
    for k, v in summary.items():
        print(f"  {k:<16} {v}")
    print(f"\nWeights: {best}")
    tag = f" --tag {args.tag}" if args.tag else ""
    print(f"Then run:  python src/06_evaluate.py --pool {args.pool}{tag}")


if __name__ == "__main__":
    main()
