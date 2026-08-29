"""
04_build_dataset.py
------------------
STEP 2 of the v3 pipeline.  Runs after 02_make_cutouts.py.

Generates the synthetic training set using the compositing algorithm in
lib_composite.py unchanged -- this script imports that module and overrides its
configuration globals rather than reimplementing anything, so the two cannot
diverge.

What it changes relative to running lib_composite.py directly:

  * cut-outs and backgrounds come from the v3 splits, which are disjoint from
    both real evaluation sets
  * the object pool size is a parameter, so the pool-diversity ablation is a
    controlled comparison rather than two ad-hoc runs
  * generation is seeded, so the dataset is reproducible (lib_composite.py ran
    with SEED = None)

Run:   python 04_build_dataset.py --pool 200
       python 04_build_dataset.py --pool 25
Output: dataset_pool<N>/
"""

from pathlib import Path
import argparse
import csv
import importlib.util
import random
import shutil

import numpy as np
from PIL import ImageDraw

ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"
CUTOUTS = ROOT / "cutouts"
BACKGROUNDS = ROOT / "backgrounds"

PREVIEW_COUNT = 12


def _load_compositor():
    """Import lib_composite.py by path (the name is not a valid identifier)."""
    spec = importlib.util.spec_from_file_location(
        "compositor_v2", ROOT / "lib_composite.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def materialise_backgrounds():
    """
    make_background() in lib_composite.py reads a directory. Copy the 50
    photographs assigned to the background role into one, so the generator
    cannot reach any image held out for evaluation.
    """
    BACKGROUNDS.mkdir(exist_ok=True)
    for old in BACKGROUNDS.iterdir():
        old.unlink()
    with open(SPLITS / "organic_bg.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, 1):
        src = ROOT / r["path"]
        shutil.copy2(src, BACKGROUNDS / f"bg_{i:04d}{src.suffix.lower()}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200,
                    help="number of e-waste cut-outs to draw objects from")
    ap.add_argument("--images", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (CUTOUTS / "screening_log.csv").exists():
        print("[!] no screened pool. Run 02_make_cutouts.py then 03_screen_cutouts.py.")
        return

    C = _load_compositor()
    out = ROOT / f"dataset_pool{args.pool}"

    # ---- point the compositor at the v3 inputs ----
    n_bg = materialise_backgrounds()
    C.RAW_ORGANIC = BACKGROUNDS
    C.EWASTE_CUTS = CUTOUTS / "ewaste_clean"   # screened by 03_screen_cutouts.py
    C.ORGANIC_CUTS = CUTOUTS / "organic"
    C.OUT = out
    C.N_IMAGES = args.images
    C.PHI_LOG.clear()

    ewaste_all = C.load_cutouts(C.EWASTE_CUTS)
    organic_cuts = C.load_cutouts(C.ORGANIC_CUTS)
    if len(ewaste_all) < args.pool:
        print(f"[!] only {len(ewaste_all)} e-waste cut-outs available, "
              f"asked for {args.pool}")
        return

    # Subset deterministically and independently of the generation stream, so
    # the pool-25 objects are a strict subset of the pool-200 objects and the
    # ablation isolates pool size rather than object identity.
    ewaste_cuts = sorted(random.Random(args.seed).sample(ewaste_all, args.pool))

    random.seed(args.seed)
    np.random.seed(args.seed)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)

    n_val = int(args.images * C.VAL_SPLIT)
    print(f"Generating {args.images} images "
          f"({args.images - n_val} train / {n_val} val) -> {out.name}")
    print(f"  e-waste cut-outs: {len(ewaste_cuts)} of {len(ewaste_all)}   "
          f"organic cut-outs: {len(organic_cuts)}   backgrounds: {n_bg}")
    print(f"  seed: {args.seed}")

    empty = 0
    for i in range(args.images):
        split = "val" if i < n_val else "train"
        img, boxes = C.build_one(ewaste_cuts, organic_cuts)
        name = f"synth_{i:05d}"

        img.save(out / "images" / split / f"{name}.jpg", quality=92)
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(C.to_yolo(b) for b in boxes))
        if not boxes:
            empty += 1

        if i < PREVIEW_COUNT:
            pv = img.copy()
            d = ImageDraw.Draw(pv)
            for b in boxes:
                d.rectangle(b, outline=(0, 255, 0), width=2)
            pv.save(out / "preview" / f"{name}_boxes.jpg", quality=92)

        if (i + 1) % 250 == 0:
            print(f"    {i + 1}/{args.images}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(C.CLASS_NAMES)}\n"
        f"names: {C.CLASS_NAMES}\n"
    )

    # ---- burial statistics, measured rather than assumed ----
    phi = np.sort(np.array(C.PHI_LOG))
    (out / "visible_fraction.csv").write_text(
        "visible_fraction\n" + "\n".join(f"{v:.6f}" for v in phi))

    def q(t):
        return float(np.quantile(phi, t))

    print(f"\nSTEP 2 complete -> {out.name}")
    print(f"  images with no visible label: {empty} ({empty / args.images:.1%})")
    print(f"  objects placed: {len(phi)}")
    print(f"  visible fraction: median {q(0.50):.3f}  mean {phi.mean():.3f}  "
          f"p10 {q(0.10):.3f}  p90 {q(0.90):.3f}")
    print(f"  retained at phi >= {C.MIN_VISIBLE_FRAC}: "
          f"{(phi >= C.MIN_VISIBLE_FRAC).mean():.1%}")
    print(f"\nThen run:  python 05_train.py --pool {args.pool}")


if __name__ == "__main__":
    main()
