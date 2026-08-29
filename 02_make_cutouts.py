

from pathlib import Path
import csv
import importlib.util

from PIL import Image, ImageOps
from rembg import remove

# ------------------------- CONFIG -------------------------
ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"
OUT = ROOT / "cutouts"

# manifest -> output subdirectory / cut-out name prefix
JOBS = [
    ("ewaste_pool", "ewaste"),
    ("organic_clutter", "organic"),
]

# Alpha matting cost scales with pixel count, and TrashBox contains a few
# multi-megapixel photographs that take minutes each. Every cut-out is scaled
# to 5-22% of a 640 px frame downstream, roughly 32-140 px, so matting above
# this resolution is discarded work rather than extra detail.
MAX_INPUT_SIDE = 1536
# ----------------------------------------------------------


def _load_v1():
    """Import lib_segment.py by path (the name is not a valid identifier)."""
    spec = importlib.util.spec_from_file_location(
        "make_cutouts_v1", ROOT / "lib_segment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V1 = _load_v1()


def read_manifest(name: str):
    with open(SPLITS / f"{name}.csv", encoding="utf-8") as f:
        return [(r["category"], ROOT / r["path"]) for r in csv.DictReader(f)]


def extract(src: Path):
    """Return a refined RGBA cut-out, or (None, reason)."""
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGBA")
    if max(img.size) > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / max(img.size)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))), Image.LANCZOS)
    cut = remove(img, session=V1.SESSION, alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_size=10)
    cut = V1.trim_to_alpha(cut)
    if cut is None:
        return None, "empty alpha"
    cut = V1.refine(cut)
    if not V1.passes_quality(cut):
        return None, "failed quality gate"
    return cut, ""


def main():
    if not (SPLITS / "summary.json").exists():
        print("[!] no splits found. Run 01_build_splits.py first.")
        return

    log = []
    for manifest, kind in JOBS:
        out_dir = OUT / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = read_manifest(manifest)
        print(f"[{kind}] {len(rows)} images from {manifest}.csv")

        kept = 0
        for i, (category, src) in enumerate(rows, 1):
            try:
                cut, reason = extract(src)
            except Exception as e:                      # noqa: BLE001
                cut, reason = None, f"{type(e).__name__}: {e}"

            if cut is None:
                log.append([manifest, category, src.relative_to(ROOT).as_posix(),
                            "", 0, reason])
            else:
                kept += 1
                name = f"{kind}_{kept:04d}.png"
                cut.save(out_dir / name)
                log.append([manifest, category, src.relative_to(ROOT).as_posix(),
                            name, 1, ""])

            if i % 25 == 0:
                print(f"    {i}/{len(rows)}  kept {kept}")

        print(f"[{kind}] done. kept {kept}, rejected {len(rows) - kept}"
              f"  -> {out_dir.relative_to(ROOT).as_posix()}\n")

    with open(OUT / "extraction_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role", "category", "source_path", "output", "kept", "reason"])
        w.writerows(log)

    print(f"STEP 1 complete -> {OUT.relative_to(ROOT).as_posix()}")
    print("Then run:  python 04_build_dataset.py")


if __name__ == "__main__":
    main()
