"""
05b_train_ednet.py
-----------------
Trains EDNet-S on the same synthetic dataset as 05_train.py, so the
architecture comparison in the paper varies the detector and nothing else.

EDNet (Song, Zhang and Abu Ebayyeh, UIC 2024; arXiv:2501.05885) is a YOLOv10
derivative aimed at small targets in drone imagery. Its XSmall detection head
and Cross Concat feature fusion are meant for objects that occupy very few
pixels, which is the regime this task sits in once an object is partly buried,
so it is a reasonable thing to try here.

TWO DIFFERENCES THE PAPER MUST DECLARE
    1. Pretraining corpus. The Ultralytics baselines start from COCO weights;
       the EDNet checkpoints published by its authors are trained on VisDrone.
       Any difference in the result therefore confounds architecture with
       pretraining data, and cannot be attributed to architecture alone.
    2. Environment. EDNet pins torch 2.0.1 and numpy<2.0, so it runs in
       external/.venv-ednet on Python 3.11 rather than the project
       environment. The dataset, the split manifests and the augmentation
       settings are identical.

Everything else -- epochs, image size, batch, patience, seed and every
augmentation probability -- is copied from 05_train.py deliberately.

Run (note the interpreter -- the project environment cannot import ednet):
    external/.venv-ednet/Scripts/python.exe 05b_train_ednet.py --pool 200

Output: runs/detect/pool200_ednets/  (+ synthetic_summary.json)
"""

from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).parent
EDNET_REPO = ROOT / "external" / "EDNet"

# Import the vendored package from the cloned repository.
sys.path.insert(0, str(EDNET_REPO))

from pipeline_common import best_f1_point, write_f1_curve  # noqa: E402

EPOCHS = 120
IMG_SIZE = 640
BATCH = 16

# EDNet-S: 9.3M parameters, the closest variant to yolov8s (11.2M) and
# yolo11s (9.4M). Comparing capacity-matched models keeps the comparison
# about architecture rather than size.
VARIANT = "small.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", type=str, default=VARIANT,
                    help="checkpoint in external/EDNet/pretrained/, e.g. small.pt")
    ap.add_argument("--tag", type=str, default="ednets")
    args = ap.parse_args()

    data = ROOT / f"dataset_pool{args.pool}" / "data.yaml"
    if not data.exists():
        print(f"[!] {data} not found. Run 04_build_dataset.py --pool {args.pool} first.")
        return 1

    weights = EDNET_REPO / "pretrained" / args.variant
    if not weights.exists():
        print(f"[!] {weights} not found. Clone github.com/zsniko/EDNet into external/.")
        return 1

    try:
        from ednet import EDNet
    except ImportError:
        print("[!] cannot import ednet.")
        print("    Use the isolated interpreter, not the project one:")
        print("    external/.venv-ednet/Scripts/python.exe 05b_train_ednet.py")
        return 1

    run_name = f"pool{args.pool}_{args.tag}"
    model = EDNet(str(weights))

    t0 = time.time()
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name=run_name,
        project=str(ROOT / "runs" / "detect"),
        patience=30,
        seed=args.seed,
        deterministic=True,
        device=0,
        # --- identical to 05_train.py; do not tune these independently,
        # --- or the comparison stops being about the architecture ---
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
        copy_paste=0.0,        # the data is already built by compositing
    )
    train_seconds = time.time() - t0

    print("\n--- validation on the SYNTHETIC split ---")
    metrics = model.val()
    conf, f1 = best_f1_point(metrics)

    out_dir = ROOT / "runs" / "detect" / run_name
    write_f1_curve(out_dir, metrics)

    summary = {
        "pool": args.pool,
        "seed": args.seed,
        "model": f"ednet-{args.variant}",
        "pretrained_on": "VisDrone",       # the baselines are COCO; see docstring
        "epochs": args.epochs,
        "dataset": data.parent.name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "best_f1": f1,
        "best_f1_conf": conf,
        "train_seconds": round(train_seconds, 1),
    }
    (out_dir / "synthetic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSynthetic validation summary")
    for k, v in summary.items():
        print(f"  {k:<15} {v}")
    print(f"\nWeights: {out_dir / 'weights' / 'best.pt'}")
    print(f"Then run:  python 06_evaluate.py --pool {args.pool} "
          f"--tag {args.tag} --backend ednet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
