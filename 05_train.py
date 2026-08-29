"""
05_train.py
--------------
STEP 3 of the v3 pipeline.  Runs after 04_build_dataset.py.

Trains YOLOv8s on a v3 synthetic dataset with the same hyperparameters as
03_train.py, and records the synthetic validation summary the paper reports,
including the confidence that maximises F1 on the synthetic split. That
threshold is what a practitioner would carry into deployment, so it is the
quantity the real-image evaluation is compared against.

Run:   python 05_train.py --pool 200
       python 05_train.py --pool 25
Output: runs/detect/pool<N>/  (+ synthetic_summary.json)
"""

from pathlib import Path
import argparse
import json
import time

from ultralytics import YOLO

from pipeline_common import best_f1_point, write_f1_curve

ROOT = Path(__file__).parent

MODEL = "yolov8s.pt"
EPOCHS = 120
IMG_SIZE = 640
BATCH = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--model", type=str, default=MODEL,
                    help="pretrained checkpoint: yolov8s.pt (primary) or "
                         "yolo11s.pt (comparison)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the run directory, used by the "
                         "architecture comparison")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed; the dataset is held fixed, so varying "
                         "this isolates run-to-run variance from initialisation, "
                         "dataloader order and augmentation draws")
    args = ap.parse_args()

    data = ROOT / f"dataset_pool{args.pool}" / "data.yaml"
    if not data.exists():
        print(f"[!] {data} not found. Run 04_build_dataset.py --pool {args.pool} first.")
        return

    # seed 0 keeps the original directory name so existing results stay valid
    run_name = (f"pool{args.pool}" if args.seed == 0
                else f"pool{args.pool}_s{args.seed}")
    if args.tag:
        run_name = f"pool{args.pool}_{args.tag}"
    model = YOLO(args.model)

    t0 = time.time()
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name=run_name,
        patience=30,
        seed=args.seed,
        deterministic=True,
        device=0,
        # --- augmentation tuned for small, partly buried objects ---
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
        "model": args.model,
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
        print(f"  {k:<14} {v}")
    print(f"\nWeights: {out_dir / 'weights' / 'best.pt'}")
    print(f"Then run:  python 06_evaluate.py --pool {args.pool}")


if __name__ == "__main__":
    main()
