"""
Figure: the best single detector (YOLOv11s) on withheld real photographs.

Row 1: detections on real e-waste photographs, two per category.
Row 2: the head's e-waste score at every grid location for the same
       photographs, i.e. the quantity the threshold acts on (see score_map).
Row 3: the lowest-scoring miss in each e-waste category, and the three most
       confident false alarms on organic waste. Misses carry no box: none of
       their detections reached the threshold.

Everything is shown at the threshold that holds YOLOv11s to a 10% false-alarm
rate on the organic set (read from stats.json written by paper_figures.py).
Images are chosen by rule, not by eye: detections are the hits closest to the
median confidence of hits in their category, so they are typical rather than
best-case.

Run from the project root, after paper_figures.py:
    python src/paper/best_model.py
"""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import lib_modules  # noqa: E402,F401  registers custom modules for unpickling
from pipeline_common import load_image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

FIG = ROOT / "Manuscripts" / "latex" / "figures"
WEIGHTS = ROOT / "runs" / "detect" / "pool54_yolo11s" / "weights" / "best.pt"
PER_IMAGE = ROOT / "evaluation" / "pool54_yolo11s" / "per_image.csv"
SIZE = 640
SCORE_FLOOR = 0.05      # scores below this are left uncoloured in the score maps
MM = 1 / 25.4

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "font.size": 7, "pdf.fonttype": 42})


def pad_square(img):
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (242, 242, 242))
    ox, oy = (s - w) // 2, (s - h) // 2
    out.paste(img, (ox, oy))
    return out.resize((SIZE, SIZE), Image.LANCZOS), (ox, oy, w, h, s)


def score_map(net, x):
    """
    E-waste score at every location the head predicts from: the sigmoid class
    score of the 80x80, 40x40 and 20x20 grids, each upsampled to the input,
    maximum taken across scales. This is the quantity the threshold is applied
    to, before box decoding and non-maximum suppression.
    """
    with torch.no_grad():
        scores = net(x)[0][0, 4]
    out, i = torch.zeros(SIZE, SIZE, device=scores.device), 0
    for g in (SIZE // 8, SIZE // 16, SIZE // 32):
        m = scores[i:i + g * g].reshape(1, 1, g, g)
        i += g * g
        m = torch.nn.functional.interpolate(m, size=(SIZE, SIZE), mode="bilinear",
                                            align_corners=False)[0, 0]
        out = torch.maximum(out, m)
    return out.cpu().numpy()


def main():
    stats = json.load(open(Path(__file__).parent / "stats.json"))
    thr = stats["arms"]["YOLOv11s"]["thr10"]
    rows = list(csv.DictReader(open(PER_IMAGE, encoding="utf-8")))
    for r in rows:
        r["c"] = float(r["max_conf"]) if r["max_conf"] else 0.0
    ew = [r for r in rows if r["role"] == "ewaste_test"]
    org = [r for r in rows if r["role"] == "organic_test"]
    cats = sorted({r["category"] for r in ew})

    hits, misses = [], []
    for c in cats:
        hc = [r for r in ew if r["category"] == c and r["c"] >= thr]
        med = np.median([r["c"] for r in hc])
        hits += sorted(hc, key=lambda r: abs(r["c"] - med))[:2]
        misses.append(min([r for r in ew if r["category"] == c and r["c"] < thr],
                          key=lambda r: r["c"]))
    fas = sorted(org, key=lambda r: -r["c"])[:3]

    model = YOLO(str(WEIGHTS))
    net = model.model.cuda().eval().float()

    def run(r, conf):
        img, geom = pad_square(load_image(ROOT / r["path"]))
        res = model.predict(np.array(img)[:, :, ::-1], conf=conf, imgsz=SIZE, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float().cuda() / 255
        cam = score_map(net, x)
        # every panel is the same square: the photograph letterboxed in light grey
        return np.array(img), cam, boxes, confs

    fig, axes = plt.subplots(3, 6, figsize=(184 * MM, 100 * MM))
    fig.subplots_adjust(left=0.035, right=0.94, top=0.99, bottom=0.03, wspace=0.05, hspace=0.2)
    nice = {"electrical cables": "cable", "electronic chips": "circuit board",
            "smartphones": "smartphone", "Food Organics": "food", "Vegetation": "vegetation"}

    def show(ax, img, boxes, confs, colour, title, cam=None):
        ax.imshow(img)
        if cam is not None:
            # photograph dimmed, then the score in a perceptually uniform map at a
            # fixed opacity, so the colour bar reads the same in every panel
            ax.imshow(np.zeros_like(cam), cmap="gray", alpha=0.35, vmin=0, vmax=1)
            ax.imshow(np.ma.masked_less(cam, SCORE_FLOOR), cmap="viridis", alpha=0.8,
                      vmin=0, vmax=1)
        for x0, y0, x1, y1 in boxes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.9, ec=colour))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(title, fontsize=6.2, labelpad=2)

    letters = iter("abcdefghijklmnopqr")
    shown = [run(r, thr) for r in hits]
    for j, (r, (img, cam, b, c)) in enumerate(zip(hits, shown)):
        show(axes[0, j], img, b, c, "#00B050", f"({next(letters)}) {nice[r['category']]}, {r['c']:.2f}")
    for j, (img, cam, b, c) in enumerate(shown):
        show(axes[1, j], img, [], [], None, f"({next(letters)}) score map", cam=cam)
    for j, r in enumerate(misses + fas):
        is_fa = r in fas
        img, cam, b, c = run(r, thr)
        show(axes[2, j], img, b if is_fa else [], c, "#E0302C",
             f"({next(letters)}) {'false alarm' if is_fa else 'miss'}, {nice[r['category']]}, "
             f"{r['c']:.2f}")
    pos = axes[1, 5].get_position()
    cax = fig.add_axes([0.952, pos.y0, 0.011, pos.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"), cax=cax)
    cb.set_label("e-waste score", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.ax.axhline(thr, color="white", lw=0.8)
    cb.ax.text(2.6, thr, "t", transform=cb.ax.get_yaxis_transform(), fontsize=6, va="center")
    for ax, t in zip(axes[:, 0], ["Detections", "Score map", "Errors"]):
        ax.set_ylabel(t, fontsize=7, labelpad=3)
    fig.savefig(FIG / "fig09_best_model_examples.pdf", dpi=300)
    print("threshold", thr)
    for r in hits + misses + fas:
        print(r["category"], round(r["c"], 3), r["path"])


if __name__ == "__main__":
    main()
