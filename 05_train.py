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

Run:   python 05_train.py --pool 60 --model yolov8s.pt
       python 05_train.py --pool 60 --model models/yolov8s-cbam.yaml --tag v8s_cbam
Output: runs/detect/pool<N>[_<tag>]/  (+ synthetic_summary.json)
"""

from pathlib import Path
import argparse
import json
import time

from ultralytics import YOLO

import lib_modules  # noqa: F401  binds CBAM and BiFPNFuse for the YAML parser
from lib_metrics import count_gflops, count_parameters, weight_size_mb
from pipeline_common import best_f1_point, write_f1_curve

ROOT = Path(__file__).parent

# The shared schedule. Batch 16 was chosen by measuring peak VRAM for every
# architecture in the comparison: the heaviest reaches 7.6 GiB of an 8 GiB card
# at batch 32, which leaves nothing for the desktop and has already cost one
# run an out-of-memory failure at epoch 90. Ultralytics accumulates gradients
# to a nominal batch of 64, so 16 and 32 perform identical optimisation and
# only the memory ceiling differs.
TRAIN_CFG = dict(
    epochs=120,
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
    ap.add_argument("--pool", type=int, default=60)
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
    ap.add_argument("--workers", type=int, default=TRAIN_CFG["workers"],
                    help="dataloader workers; affects speed and host RAM only, "
                         "never the result")
    ap.add_argument("--seed", type=int, default=TRAIN_CFG["seed"],
                    help="training seed; the dataset is held fixed, so varying "
                         "this isolates run-to-run variance from initialisation, "
                         "dataloader order and augmentation draws")
    args = ap.parse_args()

    data = ROOT / f"dataset_pool{args.pool}" / "data.yaml"
    if not data.exists():
        print(f"[!] {data} not found. Run 04_build_dataset.py --pool {args.pool} first.")
        return

    run_name = f"pool{args.pool}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"
    elif args.seed != 0:
        run_name = f"{run_name}_s{args.seed}"

    model = YOLO(args.model)
    weights = resolve_weights(args.model, args.weights)
    if weights:
        print(f"[+] transferring matching layers from {weights}")
        model.load(weights)

    cfg = dict(TRAIN_CFG, epochs=args.epochs, seed=args.seed,
               workers=args.workers)

    t0 = time.time()
    model.train(data=str(data), name=run_name, **cfg)
    train_seconds = time.time() - t0

    print("\n--- validation on the SYNTHETIC split ---")
    metrics = model.val()
    conf, f1 = best_f1_point(metrics)

    out_dir = ROOT / "runs" / "detect" / run_name
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
    print(f"Then run:  python 06_evaluate.py --pool {args.pool}{tag}")


if __name__ == "__main__":
    main()
