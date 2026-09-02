
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
FIGDIR = ROOT / "Manuscripts" / "figures"

DPI = 400
RULE = "#444444"
LABEL = "#111111"

# Shared with the TikZ figures in the manuscript, so the printed article reads
# as one visual system. Verdigris marks the method side, clay the
# contamination / discarded side; the two differ in lightness as well as hue,
# so they stay separable in greyscale and under colour-vision deficiency.
# Contrast between the pair was measured at WCAG 3.38:1, and 2.63:1 under
# simulated protanopia, with a luminance separation of 0.21. An earlier,
# softer pair (#3F6659 / #9C6B4A) measured only 1.42:1 and collapsed to
# 1.06:1 under protanopia, which would not have survived greyscale printing.
VERDIGRIS = "#1F3B33"
VERDIFILL = "#DCE8E2"
CLAY = "#B07C52"
CLAYFILL = "#F2E2D2"
INKSLATE = "#2B3A34"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # mathtext ignores font.serif and would fall back to DejaVu Sans, leaving
    # the phi in the axis label in a different face from the words around it.
    # STIX ships with matplotlib and matches the Times-family body text.
    "mathtext.fontset": "stix",
})


def grid(images, captions, out, ncols=3, panel_in=1.75, pad=0.06):
    """Lay images out on a white ground, one short caption under each."""
    n = len(images)
    if n == 0:
        print(f"  [!] nothing to draw for {out.name}")
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_in, nrows * (panel_in + 0.16)))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_axis_off()

    for ax, img, cap in zip(axes, images, captions):
        ax.set_axis_on()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(RULE)
            s.set_linewidth(0.6)
        ax.set_xlabel(cap, fontsize=6.6, color=LABEL, labelpad=2.5)

    fig.subplots_adjust(wspace=pad, hspace=0.22,
                        left=0.005, right=0.995, top=0.995, bottom=0.02)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT).as_posix()}  ({n} panels)")


def synthetic_examples(pool, n=6):
    src = sorted((ROOT / f"dataset_pool{pool}" / "preview").glob("*_boxes.jpg"))[:n]
    imgs = [Image.open(p).convert("RGB") for p in src]
    caps = [f"({chr(97 + i)})" for i in range(len(src))]
    grid(imgs, caps, FIGDIR / "Figure_3.png", ncols=3)


def false_positives(pool, n=6):
    src = sorted((ROOT / f"eval_pool{pool}" / "false_positives").glob("FP_*.jpg"))[:n]
    imgs, caps = [], []
    for i, p in enumerate(src):
        imgs.append(Image.open(p).convert("RGB"))
        # filenames carry the confidence: FP_01_conf0.83_n2_<stem>.jpg
        conf = next((t.replace("conf", "") for t in p.stem.split("_")
                     if t.startswith("conf")), "")
        caps.append(f"({chr(97 + i)}) confidence {conf}")
    grid(imgs, caps, FIGDIR / "Figure_6.png", ncols=3)


def visible_fraction(pool, threshold=0.35):
    path = ROOT / f"dataset_pool{pool}" / "visible_fraction.csv"
    if not path.exists():
        print(f"  [!] {path.name} missing")
        return
    with open(path, encoding="utf-8") as f:
        vals = np.array([float(r["visible_fraction"]) for r in csv.DictReader(f)])

    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    # Colour splits the distribution at the decision the threshold actually
    # makes: bars left of it are dropped from the annotation set, bars right
    # of it are kept. Same verdigris/clay pair used in the TikZ figures.
    bins = np.linspace(0, 1, 41)
    counts, edges = np.histogram(vals, bins=bins)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        keep = lo >= threshold
        ax.bar(lo, c, width=(hi - lo), align="edge",
               color=VERDIFILL if keep else CLAYFILL,
               edgecolor=VERDIGRIS if keep else CLAY, linewidth=0.5)
    ax.axvline(threshold, color=INKSLATE, linestyle=(0, (4, 2)), linewidth=1.0)
    ax.annotate(f"annotation threshold\n$\\varphi = {threshold}$",
                xy=(threshold, ax.get_ylim()[1] * 0.92),
                xytext=(threshold + 0.05, ax.get_ylim()[1] * 0.92),
                fontsize=7.2, va="top", ha="left", color=INKSLATE)
    ax.annotate(f"dropped\n{(vals < threshold).mean():.0%}",
                xy=(0.16, ax.get_ylim()[1] * 0.55), fontsize=7.2,
                ha="center", va="center", color=CLAY)
    ax.annotate(f"annotated\n{(vals >= threshold).mean():.0%}",
                xy=(0.68, ax.get_ylim()[1] * 0.55), fontsize=7.2,
                ha="center", va="center", color=VERDIGRIS)
    ax.set_xlabel("visible fraction $\\varphi$ after occlusion")
    ax.set_ylabel("placed objects")
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_edgecolor(RULE)
    ax.tick_params(width=0.6, length=2.5, labelsize=6.8)
    fig.tight_layout(pad=0.3)
    out = FIGDIR / "Figure_4.pdf"
    fig.savefig(out, facecolor="white")
    plt.close(fig)

    kept = (vals >= threshold).mean()
    print(f"  wrote {out.relative_to(ROOT).as_posix()}  "
          f"(n={len(vals)}, median {np.median(vals):.3f}, kept {kept:.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200)
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures -> {FIGDIR.relative_to(ROOT).as_posix()}")
    synthetic_examples(args.pool)
    false_positives(args.pool)
    visible_fraction(args.pool)


if __name__ == "__main__":
    main()
