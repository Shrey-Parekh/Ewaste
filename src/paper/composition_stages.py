"""
Figure: one synthetic image built stage by stage.

Runs the compositing functions of src/lib_composite.py in the order build_one()
calls them and keeps the canvas after every stage, so the figure shows what the
pipeline does rather than a redrawn imitation. The optional steps (fragment
crop, perspective warp, shadow, burial) are forced on so that each one is
visible; in the dataset they fire with the probabilities stated in the paper.

Run from the project root:
    python src/paper/composition_stages.py --seed 7 --obj ewaste_0015.png
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Manuscripts" / "latex" / "figures"
MM = 1 / 25.4

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "font.size": 7, "pdf.fonttype": 42})


def load_compositor():
    spec = importlib.util.spec_from_file_location("lc", ROOT / "src" / "lib_composite.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pasted(canvas, obj, px, py):
    out = canvas.copy()
    out.paste(obj, (px, py), obj)
    return out


def checker(size, n=16):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((xx // n) + (yy // n)) % 2) * 40 + 200
    return Image.fromarray(np.dstack([c, c, c]).astype(np.uint8))


def on_checker(rgba):
    base = checker(rgba.size)
    base.paste(rgba, (0, 0), rgba)
    return base


def centre_square(img):
    """Centre crop to a square, so every panel fills its cell."""
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def square_rgba(img):
    """Pad a cut-out to a square with transparency, shown on the checkerboard."""
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return on_checker(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--obj", default="ewaste_0015.png")
    ap.add_argument("--frac", type=float, default=0.22)
    ap.add_argument("--out", default="fig03_composition_stages")
    args = ap.parse_args()

    C = load_compositor()
    C.RAW_ORGANIC = ROOT / "backgrounds" / "train"
    C.FRAGMENT_PROB = C.WARP_PROB = C.SHADOW_PROB = C.OCCLUDE_PROB = 1.0
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(ROOT / "splits" / "synthetic_pool54.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    organic = [ROOT / "cutouts" / "organic" / r["name"] for r in rows
               if r["kind"] == "organic" and r["split"] == "train"]
    with open(ROOT / "cutouts" / "extraction_log.csv", encoding="utf-8") as f:
        source = {r["output"]: ROOT / r["source_path"] for r in csv.DictReader(f)}

    cut = Image.open(ROOT / "cutouts" / "ewaste_clean" / args.obj).convert("RGBA")
    photo = Image.open(source[args.obj]).convert("RGB")

    # ---- scene: background, then clutter, exactly as build_one ----
    canvas = C.make_background()
    s_bg = canvas.copy()
    bg_detail, bg_noise = C.detail_level(canvas), C.noise_level(canvas)
    for _ in range(random.randint(C.CLUTTER_MIN, C.CLUTTER_MAX)):
        org = Image.open(random.choice(organic)).convert("RGBA")
        org = C.fit_object(org, random.uniform(C.MIN_FRAC, C.MAX_FRAC + 0.14))
        org = C.maybe_warp(org)
        org = C.harmonize(org, canvas, C.HARMONIZE_STRENGTH * 0.7)
        org = C.match_sharpness_and_grain(org, bg_detail, bg_noise)
        res = C.prepare(org, allow_offframe=0.4)
        if res:
            o, px, py, a = res
            canvas = C.draw_shadow(canvas, a, o.size)
            canvas.paste(o, (px, py), o)
    s_clutter = canvas.copy()

    # ---- the contaminant, one correction at a time ----
    frag = C.maybe_fragment(cut)
    obj = C.maybe_warp(C.fit_object(frag, args.frac))
    obj_f, px, py, alpha_full = C.prepare(obj, allow_offframe=0.0)
    patch = C.bg_patch_under(canvas, px, py, *obj_f.size)
    harm = C.harmonize(obj_f, patch)
    sharp = C.match_sharpness_and_grain(harm, bg_detail, bg_noise)

    s_naive = pasted(canvas, obj_f, px, py)
    s_harm = pasted(canvas, harm, px, py)
    s_sharp = pasted(canvas, sharp, px, py)
    canvas = C.draw_shadow(canvas, alpha_full, sharp.size)
    canvas.paste(sharp, (px, py), sharp)
    s_shadow = canvas.copy()

    placed_mask = np.array(alpha_full) > 128
    owner = placed_mask.copy()

    # ---- partial burial, as layer 3 of build_one ----
    ys, xs = np.where(owner)
    cx, cy = int(xs.mean()), int(ys.mean())
    ow_, oh_ = int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1
    span = max(ow_, oh_)
    for _ in range(random.randint(*C.OCCLUDERS_PER_OBJECT)):
        org = Image.open(random.choice(organic)).convert("RGBA")
        sc = span * random.uniform(*C.OCCLUDER_SCALE) / max(org.size)
        org = org.resize((max(4, int(org.width * sc)), max(4, int(org.height * sc))),
                         Image.LANCZOS)
        org = C.maybe_warp(org)
        org = C.harmonize(org, C.bg_patch_under(canvas, cx - 32, cy - 32, 64, 64),
                          C.HARMONIZE_STRENGTH * 0.7)
        org = C.match_sharpness_and_grain(org, bg_detail, bg_noise)
        org = org.rotate(random.uniform(0, 360), expand=True, resample=Image.BICUBIC)
        opx = cx - org.width // 2 + int(random.uniform(-C.OCCLUDER_JITTER, C.OCCLUDER_JITTER) * ow_)
        opy = cy - org.height // 2 + int(random.uniform(-C.OCCLUDER_JITTER, C.OCCLUDER_JITTER) * oh_)
        occ = Image.new("L", (C.IMG_SIZE, C.IMG_SIZE), 0)
        occ.paste(org.split()[-1], (opx, opy))
        canvas = C.draw_shadow(canvas, occ, org.size)
        canvas.paste(org, (opx, opy), org)
        owner[np.array(occ) > 128] = False
    s_buried = canvas.copy()
    final = C.degrade(canvas)

    phi = owner.sum() / placed_mask.sum()
    vy, vx = np.where(owner)
    vis_box = (vx.min(), vy.min(), vx.max() + 1, vy.max() + 1)
    py_, px_ = np.where(placed_mask)
    full_box = (px_.min(), py_.min(), px_.max() + 1, py_.max() + 1)

    # visibility map: clay = hidden part of the object, verdigris = still visible
    vis = np.full((C.IMG_SIZE, C.IMG_SIZE, 3), 245, np.uint8)
    vis[placed_mask] = (176, 124, 82)
    vis[owner] = (31, 59, 51)
    vis_img = Image.fromarray(vis)

    # zoom window around the object
    x0, y0, x1, y1 = full_box
    half = int(max(x1 - x0, y1 - y0) * 0.85)
    zx = int(np.clip((x0 + x1) // 2, half, C.IMG_SIZE - half))
    zy = int(np.clip((y0 + y1) // 2, half, C.IMG_SIZE - half))
    win = (zx - half, zy - half, zx + half, zy + half)

    def zoom(im):
        return im.crop(win)

    rows_ = [
        [(centre_square(photo), "(a) Source photograph"),
         (centre_square(cut.split()[-1].convert("RGB")), "(b) Alpha matte"),
         (centre_square(on_checker(cut)), "(c) Refined cut-out"),
         (square_rgba(frag), "(d) Fragment crop"),
         (square_rgba(obj_f), "(e) Scaled, warped, rotated"),
         (centre_square(patch), "(f) Landing patch")],
        [(s_bg, "(g) Background crop"),
         (s_clutter, "(h) Organic clutter"),
         (s_naive, "(i) Naive paste"),
         (s_shadow, "(j) Corrected paste"),
         (s_buried, "(k) Partial burial"),
         (final, "(l) Camera pass, label")],
        [(zoom(s_naive), "(m) Naive paste"),
         (zoom(s_harm), "(n) + colour harmonisation"),
         (zoom(s_sharp), "(o) + sharpness, grain"),
         (zoom(s_shadow), "(p) + contact shadow"),
         (zoom(s_buried), "(q) + occluders"),
         (zoom(vis_img), rf"(r) Visibility, $\varphi$ = {phi:.2f}")],
    ]
    row_titles = ["Object preparation", "Scene assembly",
                  "Detail of the contaminant"]

    fig, axes = plt.subplots(3, 6, figsize=(184 * MM, 101 * MM))
    fig.subplots_adjust(left=0.03, right=0.995, top=0.995, bottom=0.045,
                        wspace=0.04, hspace=0.18)
    for r, row in enumerate(rows_):
        for c, (im, title) in enumerate(row):
            ax = axes[r, c]
            ax.imshow(im, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#555555")
            ax.set_xlabel(title, fontsize=6.5, labelpad=2)
        axes[r, 0].set_ylabel(row_titles[r], fontsize=7, labelpad=3)

    # derived box (solid) against the box the object had before burial (dashed)
    ax = axes[1, 5]
    for (bx0, by0, bx1, by1), ls, col in ((full_box, "--", "#B07C52"),
                                          (vis_box, "-", "#00B050")):
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                               lw=0.9, ls=ls, ec=col))
    axes[1, 4].add_patch(Rectangle((win[0], win[1]), win[2] - win[0], win[3] - win[1],
                                   fill=False, lw=0.6, ls=":", ec="white"))

    # legend for the visibility map
    from matplotlib.patches import Patch
    axes[2, 5].legend(handles=[Patch(color="#1F3B33", label="visible"),
                               Patch(color="#B07C52", label="buried")],
                      loc="upper left", fontsize=5.5, frameon=False, handlelength=1.0,
                      borderaxespad=0.3)
    for ext in ("pdf",):
        fig.savefig(OUT / f"{args.out}.{ext}", dpi=400)
    print(f"phi={phi:.3f}  full_box={full_box}  visible_box={vis_box}")


if __name__ == "__main__":
    main()
