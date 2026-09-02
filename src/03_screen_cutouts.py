
from pathlib import Path
import argparse
import csv
import shutil

import numpy as np
from PIL import Image
from ultralytics import YOLO

# ------------------------- CONFIG -------------------------
# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
CUTOUTS = ROOT / "cutouts"
SRC = CUTOUTS / "ewaste"
DST = CUTOUTS / "ewaste_clean"

SCREEN_MODEL = "yolov8s.pt"      # COCO weights, already in the project root
SCREEN_CONF = 0.35
PERSON_CONF = 0.50
DOMINANT_CONF = 0.50

# Calibrated by inspecting cut-outs either side of the boundary. Above 0.28 the
# sample was uniformly sound, including thin objects -- cables, a hair dryer, a
# vacuum hose -- whose slender geometry gives them a naturally low solid
# fraction. Below 0.15 it was almost entirely ghosts and translucent smears.
# 0.25 sits between the two and does not discard the thin cable-type
# contaminants that matter most for this task.
MIN_SOLID_FRACTION = 0.25
MAX_ELECTRONIC = 3               # this many distinct devices implies a collage

ELECTRONIC = {
    "laptop", "tv", "cell phone", "keyboard", "mouse", "remote", "microwave",
    "toaster", "refrigerator", "oven", "hair drier", "clock",
}
# Confident detections of these as the largest object indicate a source label
# error rather than an electronic item.
NON_ELECTRONIC = {
    "cup", "bowl", "bottle", "wine glass", "fork", "knife", "spoon", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "book", "vase", "teddy bear", "cat", "dog", "bird",
}
BATCH = 32
# ----------------------------------------------------------


def alpha_metrics(path: Path):
    """Solid and semi-transparent fractions of the cut-out's foreground."""
    a = np.array(Image.open(path).convert("RGBA"))[:, :, 3].astype(np.float32) / 255.0
    fg = a > 0.05
    n = max(int(fg.sum()), 1)
    return float((a > 0.9).sum() / n), float(((a > 0.05) & (a < 0.9)).sum() / n)


def on_white(path: Path) -> Image.Image:
    """Composite onto white; the screening detector expects an opaque image."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, (0, 0), im)
    return bg


def screen_content(paths):
    """Run the COCO detector over every cut-out and summarise what it found."""
    model = YOLO(SCREEN_MODEL)
    out = {}
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        results = model.predict([on_white(p) for p in chunk],
                                conf=SCREEN_CONF, verbose=False, device=0)
        for p, r in zip(chunk, results):
            person = False
            n_elec = 0
            dominant, dom_area, dom_conf = "", -1.0, 0.0
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()
                for c, cf, b in zip(cls, conf, xyxy):
                    name = r.names[int(c)]
                    if name == "person" and cf >= PERSON_CONF:
                        person = True
                    if name in ELECTRONIC:
                        n_elec += 1
                    area = float((b[2] - b[0]) * (b[3] - b[1]))
                    if area > dom_area:
                        dominant, dom_area, dom_conf = name, area, float(cf)
            out[p.name] = (person, n_elec, dominant, dom_conf)
        if (i + BATCH) % 128 == 0:
            print(f"    screened {min(i + BATCH, len(paths))}/{len(paths)}")
    return out


def verdict(solid, person, n_elec, dominant, dom_conf):
    """
    The dominant-class test was trialled and removed. Inspecting its 17
    rejections showed 16 were genuine e-waste that COCO has no vocabulary for:
    a yellow extension cord read as "banana", green circuit boards as "carrot"
    and "broccoli", a hair dryer as "toilet", earphones as "cup". Only one was a
    true source label error. A rule that is wrong 94% of the time removes more
    signal than noise, so the residual label error is left in the pool and
    reported instead. The dominant class is still recorded in the log.
    """
    if solid < MIN_SOLID_FRACTION:
        return "reject", "matting failure"
    if person:
        return "reject", "contains a person"
    if n_elec >= MAX_ELECTRONIC:
        return "reject", "multi-device collage"
    return "keep", ""


def stratified_select(kept, cats, n):
    """
    Take n survivors spread proportionally across source categories, using
    largest-remainder rounding, so the object pool mirrors the composition of
    the screened candidates rather than whichever categories happen to sort
    first. Order within a category is preserved.
    """
    by_cat = {}
    for p in kept:
        by_cat.setdefault(cats.get(p.name, ""), []).append(p)
    if n >= len(kept):
        return kept

    total = len(kept)
    exact = {c: n * len(v) / total for c, v in by_cat.items()}
    alloc = {c: int(v) for c, v in exact.items()}
    for c in sorted(exact, key=lambda c: exact[c] - alloc[c],
                    reverse=True)[:n - sum(alloc.values())]:
        alloc[c] += 1

    out = []
    for c in sorted(by_cat):
        out.extend(by_cat[c][:alloc[c]])
    return out


def category_of(paths):
    """Map cut-out filename -> source category, via the extraction log."""
    log = CUTOUTS / "extraction_log.csv"
    if not log.exists():
        return {}
    with open(log, encoding="utf-8") as f:
        return {r["output"]: r["category"] for r in csv.DictReader(f) if r["output"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="print metric distributions and stop, for calibration")
    ap.add_argument("--keep", type=int, default=200,
                    help="size of the final object pool")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"[!] {SRC} not found. Run 02_make_cutouts.py first.")
        return

    paths = sorted(SRC.glob("*.png"))
    print(f"Screening {len(paths)} cut-outs")

    solids, hazes = {}, {}
    for p in paths:
        solids[p.name], hazes[p.name] = alpha_metrics(p)

    if args.report:
        s = np.array(list(solids.values()))
        h = np.array(list(hazes.values()))
        print("\nsolid fraction (share of foreground fully opaque)")
        for q in (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95):
            print(f"  p{int(q * 100):<3} {np.quantile(s, q):.3f}")
        print("\nsemi-transparent fraction")
        for q in (0.05, 0.25, 0.50, 0.75, 0.95):
            print(f"  p{int(q * 100):<3} {np.quantile(h, q):.3f}")
        for t in (0.20, 0.30, 0.35, 0.40, 0.50, 0.60):
            print(f"  solid < {t:.2f} would reject {int((s < t).sum()):>3}"
                  f"  ({(s < t).mean():.0%})")
        print("\nlowest solid fraction:")
        for name, v in sorted(solids.items(), key=lambda kv: kv[1])[:10]:
            print(f"  {name}  solid {v:.3f}  haze {hazes[name]:.3f}")
        return

    content = screen_content(paths)
    cats = category_of(paths)

    rows, kept = [], []
    for p in paths:
        person, n_elec, dominant, dom_conf = content.get(p.name, (False, 0, "", 0.0))
        v, reason = verdict(solids[p.name], person, n_elec, dominant, dom_conf)
        rows.append([p.name, cats.get(p.name, ""), f"{solids[p.name]:.4f}",
                     f"{hazes[p.name]:.4f}", int(person), n_elec, dominant,
                     v, reason])
        if v == "keep":
            kept.append(p)

    # Select the pool proportionally across categories. Candidates are written
    # grouped by category, not interleaved, so a simple prefix would truncate
    # the last categories entirely -- an early version of this took kept[:200]
    # and dropped every smartphone.
    selected = stratified_select(kept, cats, args.keep)

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for i, p in enumerate(selected, 1):
        shutil.copy2(p, DST / f"ewaste_{i:04d}.png")

    with open(CUTOUTS / "screening_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["output", "category", "solid_fraction", "haze_fraction",
                    "person", "n_electronic", "dominant_class", "verdict", "reason"])
        w.writerows(rows)

    from collections import Counter
    reasons = Counter(r[8] for r in rows if r[7] == "reject")
    print(f"\n  candidates      {len(paths)}")
    for reason, n in reasons.most_common():
        print(f"    rejected: {reason:<28} {n}")
    print(f"  passed screen   {len(kept)}")
    print(f"  object pool     {len(selected)} -> {DST.relative_to(ROOT).as_posix()}")
    if len(kept) < args.keep:
        print(f"  [!] only {len(kept)} survived; pool is smaller than requested")
    print(f"  log             cutouts/screening_log.csv")
    print("\nThen run:  python 04_build_dataset.py --pool 200")


if __name__ == "__main__":
    main()
