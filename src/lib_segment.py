"""
lib_segment.py
-------------------
Segmentation library for the v3 pipeline. 02_make_cutouts.py imports
process_folder() and passes_quality() from this module by file path (see the
loader in 02_make_cutouts.py) rather than duplicating them.

  * alpha matting  -> soft, accurate edges instead of a hard binary cut
  * edge erosion   -> removes the 1-2px halo of the OLD background colour
  * colour despill -> kills the bright fringe left at the boundary
  * edge feather   -> slightly soft boundary, like a real photograph
  * quality filter -> automatically discards failed removals

DO NOT run this file directly. RAW below points at the whole raw/ewaste and
raw/organic collections, with no regard for which photos 01_build_splits.py
assigned to the training pool vs the held-out test sets -- extracting from it
directly is exactly the train/test leak the v3 split manifests exist to
prevent. Use the pipeline entry point instead:

Run:   python 02_make_cutouts.py
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from rembg import remove, new_session

# ------------------------- CONFIG -------------------------
# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RAW = {"ewaste": ROOT / "raw" / "ewaste", "organic": ROOT / "raw" / "organic"}
OUT = {"ewaste": ROOT / "cutouts" / "ewaste", "organic": ROOT / "cutouts" / "organic"}

MODEL_NAME = "isnet-general-use"   # cleaner edges than the default model

USE_ALPHA_MATTING = True   # much softer edges. Slower. Set False if too slow.
ERODE_PX          = 1      # shave this many px off the edge (kills halo)
FEATHER_PX        = 0.8    # gaussian blur on alpha (soft boundary)

# quality gates - reject obviously failed removals
MIN_COVERAGE = 0.02   # <2% of frame kept  -> removal ate the object
MAX_COVERAGE = 0.97   # >97% kept          -> removal did nothing
MIN_SIDE_PX  = 40     # too small to be useful once scaled down
# ----------------------------------------------------------

SESSION = new_session(MODEL_NAME)
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def erode_alpha(alpha: Image.Image, px: int) -> Image.Image:
    """Shrink the alpha mask inward. Removes the coloured halo at the border."""
    if px <= 0:
        return alpha
    return alpha.filter(ImageFilter.MinFilter(size=2 * px + 1))


def despill(rgba: np.ndarray) -> np.ndarray:
    """
    Edge pixels often keep a tint of the ORIGINAL background (white studio, etc).
    Blend those semi-transparent pixels toward the object's own average colour.
    """
    a = rgba[:, :, 3].astype(np.float32) / 255.0
    solid = a > 0.9
    edge = (a > 0.05) & (a <= 0.9)
    if solid.sum() < 20 or edge.sum() == 0:
        return rgba
    inner_mean = rgba[:, :, :3][solid].mean(axis=0)
    w = ((0.9 - a[edge]) / 0.85)[:, None]       # 0 at a=0.9, 1 at a=0.05
    rgba[:, :, :3][edge] = (
        rgba[:, :, :3][edge] * (1 - w) + inner_mean[None, :] * w
    ).astype(np.uint8)
    return rgba


def trim_to_alpha(img: Image.Image):
    """Crop away fully transparent borders so the cut-out is tight."""
    arr = np.array(img)
    if arr.shape[2] < 4:
        return img
    ys, xs = np.where(arr[:, :, 3] > 10)
    if len(xs) == 0:
        return None
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def refine(cut: Image.Image) -> Image.Image:
    """Erode -> despill -> feather. This is where edge realism comes from."""
    r, g, b, a = cut.split()
    a = erode_alpha(a, ERODE_PX)
    arr = np.dstack([np.array(r), np.array(g), np.array(b), np.array(a)])
    arr = despill(arr)
    out = Image.fromarray(arr, "RGBA")
    if FEATHER_PX > 0:
        r, g, b, a = out.split()
        a = a.filter(ImageFilter.GaussianBlur(FEATHER_PX))
        out = Image.merge("RGBA", (r, g, b, a))
    return out


def passes_quality(cut: Image.Image) -> bool:
    a = np.array(cut.split()[-1])
    coverage = (a > 25).mean()
    if not (MIN_COVERAGE < coverage < MAX_COVERAGE):
        return False
    if min(cut.size) < MIN_SIDE_PX:
        return False
    return True


def process_folder(kind: str):
    in_dir, out_dir = RAW[kind], OUT[kind]
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in VALID_EXT)
    if not files:
        print(f"[!] No images in {in_dir} -- did you put photos there?")
        return

    print(f"[{kind}] processing {len(files)} images ...")
    kept = rejected = 0
    for i, fp in enumerate(files, 1):
        try:
            src = Image.open(fp)
            src = ImageOps.exif_transpose(src).convert("RGBA")   # fix phone rotation

            if USE_ALPHA_MATTING:
                cut = remove(src, session=SESSION, alpha_matting=True,
                             alpha_matting_foreground_threshold=240,
                             alpha_matting_background_threshold=10,
                             alpha_matting_erode_size=10)
            else:
                cut = remove(src, session=SESSION)

            cut = trim_to_alpha(cut)
            if cut is None:
                rejected += 1
                continue
            cut = refine(cut)
            if not passes_quality(cut):
                rejected += 1
                continue

            cut.save(out_dir / f"{kind}_{i:04d}.png")
            kept += 1
        except Exception as e:
            print(f"    skipped {fp.name}: {e}")
            rejected += 1
        if i % 25 == 0:
            print(f"    {i}/{len(files)}")

    print(f"[{kind}] done. kept {kept}, rejected {rejected}  ->  {out_dir}\n")


if __name__ == "__main__":
    print("[!] This runs extraction on the WHOLE raw/ collection, ignoring the")
    print("    train/test split manifests -- that is a leak. Use the pipeline")
    print("    entry point instead:  python 02_make_cutouts.py")
    raise SystemExit(1)
