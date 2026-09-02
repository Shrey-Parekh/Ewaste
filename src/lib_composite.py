"""
lib_composite.py
----------------
Compositing library for the v3 pipeline. 04_build_dataset.py imports this
module by file path, overrides EWASTE_CUTS/RAW_ORGANIC/OUT below to the v3
split-manifest paths, then calls main() -- see the loader in
04_build_dataset.py.

Realism corrections applied while compositing (these calibrate difficulty,
they are not decoration -- an object pasted without them carries a
photometric seam a detector can key on instead of learning the object):

  1. Contact shadow      -> objects don't float.
  2. Colour harmonisation -> studio-lit object matched to the local patch.
  3. Sharpness matching   -> background detail measured, object blurred to match.
  4. Grain matching       -> matching noise added before pasting.
  5. Perspective warp     -> breaks the flat "sticker" look.
  6. Visibility-tracked labels -> the box shrinks to the visible fraction,
                            and is dropped below MIN_VISIBLE_FRAC.

DO NOT run this file directly. EWASTE_CUTS/RAW_ORGANIC below point at the
whole raw collections, not the disjoint train/test split manifests -- that is
the same train/test leak lib_segment.py's direct execution would cause.
Use the pipeline entry point instead:

Run:   python 04_build_dataset.py --pool 200
"""

from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

# ------------------------- CONFIG -------------------------
# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
EWASTE_CUTS  = ROOT / "cutouts" / "ewaste"
ORGANIC_CUTS = ROOT / "cutouts" / "organic"
RAW_ORGANIC  = ROOT / "raw" / "organic"
OUT          = ROOT / "dataset"

N_IMAGES  = 1500
IMG_SIZE  = 640
VAL_SPLIT = 0.15

MIN_EWASTE, MAX_EWASTE = 1, 3
MIN_FRAC,   MAX_FRAC   = 0.05, 0.22   # object size as fraction of frame
FRAGMENT_PROB = 0.45
OCCLUDE_PROB  = 0.85
OCCLUDERS_PER_OBJECT = (1, 3)     # how many organic pieces land on an object
OCCLUDER_SCALE       = (0.45, 1.1)  # size RELATIVE to the object's span
OCCLUDER_JITTER      = 0.30       # offset from centre, as fraction of object
CLUTTER_MIN, CLUTTER_MAX = 2, 6       # organic pieces layered for density

# --- realism controls ---
HARMONIZE_STRENGTH = 0.45   # 0 = keep original colour, 1 = fully match background
SHADOW_PROB        = 0.9
SHADOW_OPACITY     = (0.20, 0.45)
WARP_PROB          = 0.6    # mild perspective, breaks the flat-sticker look
WARP_AMOUNT        = 0.06   # fraction of object size
GRAIN_MATCH        = True

# --- label quality ---
MIN_VISIBLE_FRAC = 0.35     # drop the box if less than this is still visible
MIN_BOX_PX       = 8        # ignore boxes smaller than this

PREVIEW_COUNT = 12          # sample images saved with boxes drawn

CLASS_NAMES = ["ewaste_contaminant"]
SEED = None                 # set an int for reproducible datasets
# ----------------------------------------------------------

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Per-object visible fraction phi, appended by build_one(). Written out by
# main() to <OUT>/visible_fraction.csv. Logging only; does not consume RNG.
PHI_LOG = []

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)


# =====================================================================
# helpers
# =====================================================================

def load_cutouts(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")


def detail_level(img: Image.Image) -> float:
    """Rough sharpness measure: energy left after subtracting a blurred copy."""
    g = img.convert("L")
    hi = np.array(g, dtype=np.float32) - np.array(
        g.filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32)
    return float(hi.std())


def noise_level(img: Image.Image) -> float:
    """Rough grain estimate from high-frequency residual of a smooth region."""
    g = np.array(img.convert("L"), dtype=np.float32)
    lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    return float(np.median(np.abs(lap)) * 0.6)


def make_background() -> Image.Image:
    """
    Random crop + flip from a real organic photo.
    v1 always centre-cropped, so every image looked like the same framing.
    """
    raws = [p for p in RAW_ORGANIC.iterdir() if p.suffix.lower() in VALID_EXT]
    bg = Image.open(random.choice(raws)).convert("RGB")
    w, h = bg.size

    # random square crop covering 60-100% of the short side
    s = int(min(w, h) * random.uniform(0.6, 1.0))
    x = random.randint(0, max(0, w - s))
    y = random.randint(0, max(0, h - s))
    bg = bg.crop((x, y, x + s, y + s)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    if random.random() < 0.5:
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.2:
        bg = bg.transpose(Image.FLIP_TOP_BOTTOM)
    return bg


def fit_object(obj: Image.Image, frac: float) -> Image.Image:
    target = frac * IMG_SIZE
    w, h = obj.size
    scale = target / max(w, h)
    return obj.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def maybe_fragment(obj: Image.Image) -> Image.Image:
    """Crop to a sub-piece so it reads as a broken fragment, not a whole product."""
    if random.random() > FRAGMENT_PROB:
        return obj
    w, h = obj.size
    fw = max(8, int(w * random.uniform(0.35, 0.75)))
    fh = max(8, int(h * random.uniform(0.35, 0.75)))
    x = random.randint(0, max(0, w - fw))
    y = random.randint(0, max(0, h - fh))
    return obj.crop((x, y, x + fw, y + fh))


def _perspective_coeffs(src, dst):
    """Solve the 8 coefficients PIL needs for a PERSPECTIVE transform."""
    A = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        A.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
    A = np.array(A, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B)


def maybe_warp(obj: Image.Image) -> Image.Image:
    """Mild perspective tilt so the object does not look like a flat decal."""
    if random.random() > WARP_PROB:
        return obj
    w, h = obj.size
    if w < 12 or h < 12:
        return obj
    d = WARP_AMOUNT
    jitter = lambda v: v * random.uniform(-d, d)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(jitter(w), jitter(h)),
           (w + jitter(w), jitter(h)),
           (w + jitter(w), h + jitter(h)),
           (jitter(w), h + jitter(h))]
    try:
        coeffs = _perspective_coeffs(src, dst)
    except np.linalg.LinAlgError:
        return obj
    return obj.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC)


def harmonize(obj: Image.Image, bg_patch: Image.Image,
              strength: float = HARMONIZE_STRENGTH) -> Image.Image:
    """
    Shift the object's exposure and colour cast toward the background it is
    landing on. Partial strength keeps the object recognisable (a green PCB
    stays green) while removing the "shot in a different room" mismatch.
    """
    arr = np.array(obj).astype(np.float32)
    a = arr[:, :, 3] / 255.0
    solid = a > 0.6
    if solid.sum() < 20:
        return obj

    bg = np.array(bg_patch.convert("RGB")).astype(np.float32).reshape(-1, 3)
    if bg.shape[0] < 10:
        return obj

    obj_rgb = arr[:, :, :3]
    o_mean = obj_rgb[solid].mean(axis=0)
    o_std  = obj_rgb[solid].std(axis=0) + 1e-5
    b_mean = bg.mean(axis=0)
    b_std  = bg.std(axis=0) + 1e-5

    # partial contrast match, then partial mean (exposure + cast) match
    gain = 1.0 + (b_std / o_std - 1.0) * (strength * 0.6)
    shifted = (obj_rgb - o_mean) * gain + o_mean + (b_mean - o_mean) * strength

    arr[:, :, :3] = np.clip(shifted, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def match_sharpness_and_grain(obj: Image.Image, bg_detail: float,
                              bg_noise: float) -> Image.Image:
    """Blur the object toward the background's detail level, then add grain."""
    d = detail_level(obj.convert("RGB"))
    if d > bg_detail and bg_detail > 0:
        ratio = d / max(bg_detail, 1e-3)
        radius = float(np.clip((ratio - 1.0) * 0.45, 0.0, 1.6))
        if radius > 0.05:
            r, g, b, al = obj.split()
            rgb = Image.merge("RGB", (r, g, b)).filter(
                ImageFilter.GaussianBlur(radius))
            obj = Image.merge("RGBA", (*rgb.split(), al))

    if GRAIN_MATCH and bg_noise > 0.3:
        arr = np.array(obj).astype(np.float32)
        n = np.random.normal(0, bg_noise, arr[:, :, :3].shape)
        arr[:, :, :3] = np.clip(arr[:, :, :3] + n, 0, 255)
        obj = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return obj


def draw_shadow(canvas: Image.Image, alpha_full: Image.Image,
                obj_size) -> Image.Image:
    """
    Soft contact shadow. Single biggest fix for the floating look.
    IMPORTANT: call this BEFORE pasting the object, or the shadow
    darkens the object itself.
    """
    if random.random() > SHADOW_PROB:
        return canvas
    ow, oh = obj_size
    blur = max(2.0, 0.10 * max(ow, oh))
    dx = int(random.uniform(-0.10, 0.10) * ow)
    dy = int(random.uniform(0.02, 0.14) * oh)      # mostly downward

    shadow = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    shadow.paste(alpha_full, (dx, dy))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

    s = np.array(shadow, dtype=np.float32) / 255.0
    s *= random.uniform(*SHADOW_OPACITY)
    arr = np.array(canvas).astype(np.float32)
    arr *= (1.0 - s[:, :, None])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def prepare(obj: Image.Image, allow_offframe=0.25):
    """
    Rotate, trim and choose a position. Does NOT paste yet, so the caller
    can draw the shadow first and then paste the object on top.
    Returns (obj_final, px, py, alpha_full) or None.
    """
    obj = obj.rotate(random.uniform(0, 360), expand=True, resample=Image.BICUBIC)

    a = np.array(obj.split()[-1])
    ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return None
    obj = obj.crop((int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1))

    ow, oh = obj.size
    if ow >= IMG_SIZE or oh >= IMG_SIZE:
        return None

    mx, my = int(ow * allow_offframe), int(oh * allow_offframe)
    px = random.randint(-mx, IMG_SIZE - ow + mx)
    py = random.randint(-my, IMG_SIZE - oh + my)

    alpha_full = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    alpha_full.paste(obj.split()[-1], (px, py))
    return obj, px, py, alpha_full


def bg_patch_under(canvas, px, py, w, h):
    """The background region an object is about to land on (for harmonising)."""
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(IMG_SIZE, px + w), min(IMG_SIZE, py + h)
    if x1 <= x0 or y1 <= y0:
        return canvas.crop((0, 0, min(32, IMG_SIZE), min(32, IMG_SIZE)))
    return canvas.crop((x0, y0, x1, y1))


def degrade(canvas: Image.Image) -> Image.Image:
    """Whole-frame camera pass: colour grade, vignette, grain, soft focus, JPEG."""
    canvas = canvas.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.9)))
    canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.72, 1.06))
    canvas = ImageEnhance.Color(canvas).enhance(random.uniform(0.72, 1.08))
    canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.88, 1.08))

    arr = np.array(canvas).astype(np.float32)

    # wet / organic colour cast
    arr += np.array([random.uniform(-7, 5),
                     random.uniform(-4, 6),
                     random.uniform(-10, 3)])

    # vignette
    if random.random() < 0.7:
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        cx = cy = IMG_SIZE / 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (IMG_SIZE / 2)
        v = 1.0 - random.uniform(0.10, 0.30) * np.clip(r - 0.45, 0, None) ** 2 * 4
        arr *= np.clip(v, 0, 1)[:, :, None]

    arr += np.random.normal(0, random.uniform(2, 9), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # a real photo has been through JPEG at least once
    import io
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=random.randint(72, 93))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# =====================================================================
# build one image
# =====================================================================

def build_one(ewaste_cuts, organic_cuts):
    canvas = make_background()
    bg_detail = detail_level(canvas)
    bg_noise = noise_level(canvas)

    # owner mask: 0 = background/occluded, k = e-waste object k
    owner = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)

    # ---- layer 1: organic clutter, to build up a pile ----
    for _ in range(random.randint(CLUTTER_MIN, CLUTTER_MAX)):
        if not organic_cuts:
            break
        org = Image.open(random.choice(organic_cuts)).convert("RGBA")
        org = fit_object(org, random.uniform(MIN_FRAC, MAX_FRAC + 0.14))
        org = maybe_warp(org)
        org = harmonize(org, canvas, HARMONIZE_STRENGTH * 0.7)
        org = match_sharpness_and_grain(org, bg_detail, bg_noise)

        res = prepare(org, allow_offframe=0.4)
        if not res:
            continue
        obj_f, px, py, alpha_full = res
        canvas = draw_shadow(canvas, alpha_full, obj_f.size)   # shadow first
        canvas.paste(obj_f, (px, py), obj_f)                   # then object

    # ---- layer 2: the e-waste contaminants ----
    placed = []                      # (index, original visible pixel count)
    n_objects = random.randint(MIN_EWASTE, MAX_EWASTE)
    for k in range(1, n_objects + 1):
        obj = Image.open(random.choice(ewaste_cuts)).convert("RGBA")
        obj = maybe_fragment(obj)
        obj = fit_object(obj, random.uniform(MIN_FRAC, MAX_FRAC))
        obj = maybe_warp(obj)

        res = prepare(obj, allow_offframe=0.2)
        if not res:
            continue
        obj_f, px, py, alpha_full = res

        # harmonise against the exact patch it is landing on
        patch = bg_patch_under(canvas, px, py, *obj_f.size)
        obj_f = harmonize(obj_f, patch)
        obj_f = match_sharpness_and_grain(obj_f, bg_detail, bg_noise)

        canvas = draw_shadow(canvas, alpha_full, obj_f.size)   # shadow first
        canvas.paste(obj_f, (px, py), obj_f)                   # then object

        mask = np.array(alpha_full) > 128
        if mask.sum() < MIN_BOX_PX ** 2:
            continue
        owner[mask] = k
        placed.append((k, int(mask.sum())))

    # ---- layer 3: occluders on top (partial burial) ----
    # Occluders are sized RELATIVE TO THE OBJECT and placed near its centre,
    # otherwise they drift off and nothing is really buried.
    for k, _ in placed:
        if random.random() > OCCLUDE_PROB or not organic_cuts:
            continue
        ys, xs = np.where(owner == k)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        obj_w = int(xs.max() - xs.min()) + 1
        obj_h = int(ys.max() - ys.min()) + 1
        obj_span = max(obj_w, obj_h)

        for _ in range(random.randint(*OCCLUDERS_PER_OBJECT)):
            org = Image.open(random.choice(organic_cuts)).convert("RGBA")
            # size relative to the object, not the whole frame
            target = obj_span * random.uniform(*OCCLUDER_SCALE)
            ow0, oh0 = org.size
            sc = target / max(ow0, oh0)
            org = org.resize((max(4, int(ow0 * sc)), max(4, int(oh0 * sc))),
                             Image.LANCZOS)
            org = maybe_warp(org)
            org = harmonize(org,
                            bg_patch_under(canvas, cx - 32, cy - 32, 64, 64),
                            HARMONIZE_STRENGTH * 0.7)
            org = match_sharpness_and_grain(org, bg_detail, bg_noise)
            org = org.rotate(random.uniform(0, 360), expand=True,
                             resample=Image.BICUBIC)

            ow, oh = org.size
            # keep it ON the object: jitter is a fraction of the OBJECT size
            px = cx - ow // 2 + int(random.uniform(-OCCLUDER_JITTER,
                                                   OCCLUDER_JITTER) * obj_w)
            py = cy - oh // 2 + int(random.uniform(-OCCLUDER_JITTER,
                                                   OCCLUDER_JITTER) * obj_h)

            occ_full = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
            occ_full.paste(org.split()[-1], (px, py))

            canvas = draw_shadow(canvas, occ_full, org.size)
            canvas.paste(org, (px, py), org)

            owner[np.array(occ_full) > 128] = 0   # those pixels are now hidden

    # ---- labels built from what is ACTUALLY still visible ----
    boxes = []
    for k, orig_count in placed:
        ys, xs = np.where(owner == k)
        phi = len(xs) / max(orig_count, 1)
        PHI_LOG.append(phi)
        if len(xs) == 0:
            continue
        if phi < MIN_VISIBLE_FRAC:
            continue                       # too buried to be a fair label
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if (x1 - x0) < MIN_BOX_PX or (y1 - y0) < MIN_BOX_PX:
            continue
        boxes.append((x0, y0, x1 + 1, y1 + 1))

    canvas = degrade(canvas)
    return canvas, boxes


def to_yolo(bbox):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / IMG_SIZE
    cy = (y0 + y1) / 2 / IMG_SIZE
    w = (x1 - x0) / IMG_SIZE
    h = (y1 - y0) / IMG_SIZE
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# =====================================================================
# main
# =====================================================================

def main():
    ewaste_cuts = load_cutouts(EWASTE_CUTS)
    organic_cuts = load_cutouts(ORGANIC_CUTS)
    if not ewaste_cuts:
        print("[!] No e-waste cut-outs. Run lib_segment.py first.")
        return
    if not any(RAW_ORGANIC.iterdir()):
        print(f"[!] No background photos in {RAW_ORGANIC}")
        return

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "preview").mkdir(parents=True, exist_ok=True)

    n_val = int(N_IMAGES * VAL_SPLIT)
    print(f"Generating {N_IMAGES} images ({N_IMAGES - n_val} train / {n_val} val)")
    print(f"  e-waste cut-outs: {len(ewaste_cuts)}   organic cut-outs: {len(organic_cuts)}")

    empty = 0
    for i in range(N_IMAGES):
        split = "val" if i < n_val else "train"
        img, boxes = build_one(ewaste_cuts, organic_cuts)
        name = f"synth_{i:05d}"

        img.save(OUT / "images" / split / f"{name}.jpg", quality=92)
        (OUT / "labels" / split / f"{name}.txt").write_text(
            "\n".join(to_yolo(b) for b in boxes))
        if not boxes:
            empty += 1

        if i < PREVIEW_COUNT:
            pv = img.copy()
            d = ImageDraw.Draw(pv)
            for b in boxes:
                d.rectangle(b, outline=(0, 255, 0), width=2)
            pv.save(OUT / "preview" / f"{name}_boxes.jpg", quality=92)

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{N_IMAGES}")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    print(f"\nSTEP 2 complete -> {OUT}")
    print(f"  images with no visible label: {empty} ({empty / N_IMAGES:.1%})")

    if PHI_LOG:
        rows = "\n".join(f"{v:.6f}" for v in PHI_LOG)
        (OUT / "visible_fraction.csv").write_text("visible_fraction\n" + rows)
        arr = np.sort(np.array(PHI_LOG))
        q = lambda t: float(np.quantile(arr, t))
        print(f"  visible fraction over {len(arr)} placed objects: "
              f"median {q(0.50):.3f}  p10 {q(0.10):.3f}  p90 {q(0.90):.3f}  "
              f"mean {arr.mean():.3f}")
        print(f"  retained at phi >= {MIN_VISIBLE_FRAC}: "
              f"{(arr >= MIN_VISIBLE_FRAC).mean():.1%}")
    print(f"  LOOK AT dataset/preview/ NOW. If the objects look obviously pasted,")
    print(f"  raise HARMONIZE_STRENGTH or lower MAX_FRAC, then rerun.")
    print("Then run:  python 05_train.py --pool 200")


if __name__ == "__main__":
    print("[!] This composites from the WHOLE raw/organic collection and the")
    print("    default cutouts/ directory, ignoring the train/test split")
    print("    manifests -- that is a leak. Use the pipeline entry point instead:")
    print("    python 04_build_dataset.py --pool 200")
    raise SystemExit(1)
