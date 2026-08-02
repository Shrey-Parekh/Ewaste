"""
03_train.py  (v2)
-----------------
STEP 3 of 3.  Trains a YOLO detector on the synthetic dataset.

Changes vs v1:
  * augmentation tuned for SMALL, OCCLUDED objects (the actual problem here)
  * mosaic on, then turned off for the last epochs so the model settles on
    realistic full frames instead of 4-way collages
  * fixed seed, so the run is reproducible for the paper
  * built-in evaluation on a REAL held-out folder, which is the number that
    matters for publication

Run:   python 03_train.py

Outputs land in runs/detect/ewaste_v2/
Best weights:  runs/detect/ewaste_v2/weights/best.pt
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "dataset" / "data.yaml"
REAL_DATA = ROOT / "real_test" / "data.yaml"   # optional, see note at bottom

MODEL     = "yolov8s.pt"   # 's' handles small objects better than 'n'
EPOCHS    = 120
IMG_SIZE  = 640
BATCH     = 16             # lower to 8 or 4 if you run out of memory
RUN_NAME  = "ewaste_v2"


def main():
    if not DATA.exists():
        print(f"[!] {DATA} not found. Run 02_composite.py first.")
        return

    model = YOLO(MODEL)

    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name=RUN_NAME,
        patience=30,
        seed=0,                # reproducible for the paper
        deterministic=True,
        device = 0,
        BATCH = 16,
        # --- augmentation tuned for small, partly buried objects ---
        mosaic=1.0,            # 4-image collages: more small-object samples
        close_mosaic=15,       # switch mosaic OFF for the final 15 epochs
        scale=0.5,             # heavy scale jitter -> scale invariance
        translate=0.2,
        fliplr=0.5,
        flipud=0.3,            # waste has no natural "up"
        degrees=15.0,
        hsv_h=0.015,           # mild colour jitter; keeps PCB-green learnable
        hsv_s=0.6,
        hsv_v=0.4,
        erasing=0.3,           # random erasing mimics extra burial
        copy_paste=0.0,        # our data is ALREADY copy-paste; don't double up
    )

    print("\n--- validation on the SYNTHETIC split ---")
    model.val()

    # --- the number that actually matters ---
    if REAL_DATA.exists():
        print("\n--- validation on REAL held-out images ---")
        real_metrics = model.val(data=str(REAL_DATA), split="val")
        print("\nReport BOTH numbers in the paper. The gap between them IS the")
        print("synthetic-to-real gap, and that gap is itself a finding.")
    else:
        print(f"\n[note] No real test set found at {REAL_DATA}")
        print("       Synthetic-only results are not publishable on their own.")
        print("       Collect ~100-200 real photos of e-waste in wet organic waste,")
        print("       label them, and put them in real_test/ with a data.yaml.")

    print(f"\nBest weights: runs/detect/{RUN_NAME}/weights/best.pt")
    print("Predict on a photo:")
    print(f"   yolo predict model=runs/detect/{RUN_NAME}/weights/best.pt source=photo.jpg")


if __name__ == "__main__":
    main()
