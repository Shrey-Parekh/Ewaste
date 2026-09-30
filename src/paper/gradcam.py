"""
HiResCAM explanations for the best single detector (YOLOv11s).

Target. The alert is raised when the highest e-waste score anywhere in the
photograph reaches the threshold, so the quantity explained is that maximum:
the sigmoid class score y of the top-scoring grid location, before box decoding
and non-maximum suppression. Gradients of y are taken at the three neck outputs
that feed the detection head (layers 16, 19 and 22; strides 8, 16 and 32).

HiResCAM (Draelos and Carin): L = ReLU(sum_k dy/dA^k * A^k), element-wise.
Each level's map is upsampled to 640 px and the three are summed; only the
level holding the top location receives gradient.

Check on synthetic validation (ground-truth boxes known):
  * energy inside boxes: share of the map's mass inside the union of the
    ground-truth boxes, against the share of the frame those boxes cover;
  * pointing game: whether the map's peak falls inside a box;
  * the same, restricted to images whose top-scoring location lies in a box.

Figure: three rows of six, every panel chosen by a fixed rule, not by eye.
  real hits       the two hits closest to the median confidence of hits, per
                  e-waste category (cable, circuit board, smartphone);
  synthetic       six validation composites at evenly spaced quantiles of the
                  energy share, so weak explanations are shown as well as good;
  false alarms    the six most confident false alarms on organic waste.
For display only, the map is smoothed with a 6-pixel Gaussian (a linear
operation) and values below 0.15 are left uncoloured; a white contour marks
half the map's maximum.

Run from the project root, after paper_figures.py:
    python src/paper/gradcam.py
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
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from best_model import FIG, MM, PER_IMAGE, ROOT, SIZE, WEIGHTS, pad_square  # noqa: E402
from pipeline_common import load_image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VAL = ROOT / "dataset_pool54"
LEVELS = (16, 19, 22)
GRIDS = ((80, 8), (40, 16), (20, 32))       # (cells per side, stride) for P3, P4, P5
FLOOR = 0.15                                # display only: lower values uncoloured


class HiResCAM:
    def __init__(self, net):
        self.net, self.acts = net, {}
        for i in LEVELS:
            net.model[i].register_forward_hook(
                lambda m, inp, out, i=i: self.acts.__setitem__(i, out))

    def __call__(self, img):
        """(normalised map, top score, centre (y, x) of the top location)."""
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float().cuda() / 255
        x.requires_grad_(True)
        scores = self.net(x)[0][0, 4]
        top = int(scores.argmax())
        acts = [self.acts[i] for i in LEVELS]
        grads = torch.autograd.grad(scores[top], acts, allow_unused=True)
        cam = torch.zeros(SIZE, SIZE, device=x.device)
        for a, g in zip(acts, grads):
            if g is not None:
                m = F.relu((g * a).sum(1, keepdim=True))
                cam += F.interpolate(m, size=(SIZE, SIZE), mode="bilinear",
                                     align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        off = 0
        for g, st in GRIDS:
            if top < off + g * g:
                k = top - off
                centre = ((k // g + 0.5) * st, (k % g + 0.5) * st)
                break
            off += g * g
        return cam / (cam.max() + 1e-12), float(scores[top].detach()), centre


def val_boxes(stem):
    out = []
    for line in (VAL / "labels" / "val" / f"{stem}.txt").read_text().splitlines():
        _, cx, cy, w, h = map(float, line.split())
        out.append(((cx - w / 2) * SIZE, (cy - h / 2) * SIZE,
                    (cx + w / 2) * SIZE, (cy + h / 2) * SIZE))
    return out


def localisation(cam_fn):
    """Energy-inside-box and pointing game over labelled synthetic validation images."""
    rec, paths, top_in = [], [], []
    for p in sorted((VAL / "images" / "val").glob("*.jpg")):
        boxes = val_boxes(p.stem)
        if not boxes:
            continue
        cam, _, (cy, cx) = cam_fn(Image.open(p).convert("RGB"))
        mask = np.zeros((SIZE, SIZE), bool)
        for x0, y0, x1, y1 in boxes:
            mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
        top_in.append(bool(mask[min(int(cy), SIZE - 1), min(int(cx), SIZE - 1)]))
        ok = cam.sum() > 0
        peak = np.unravel_index(cam.argmax(), cam.shape)
        rec.append((cam[mask].sum() / cam.sum() if ok else 0.0, bool(mask[peak]) if ok else False))
        paths.append((p, mask.mean()))
    top_in = np.array(top_in)
    e, hit = (np.array(c) for c in zip(*rec))
    out = {"n": len(paths), "box_area_mean": float(np.mean([a for _, a in paths])),
           "top_location_in_box": float(top_in.mean()), "n_top_in_box": int(top_in.sum()),
           "hirescam": {"energy_mean": float(e.mean()), "energy_median": float(np.median(e)),
                        "pointing": float(hit.mean()),
                        "energy_mean_top_in_box": float(e[top_in].mean()),
                        "pointing_top_in_box": float(hit[top_in].mean())}}
    return out, e, [p for p, _ in paths]


def main():
    stats = json.load(open(Path(__file__).parent / "stats.json"))
    thr = stats["arms"]["YOLOv11s"]["thr10"]
    net = YOLO(str(WEIGHTS)).model.cuda().eval().float()
    for p in net.parameters():
        p.requires_grad_(False)
    cam_fn = HiResCAM(net)

    loc, energy, paths = localisation(cam_fn)
    (Path(__file__).parent / "gradcam.json").write_text(
        json.dumps({"threshold": thr, **loc}, indent=1))
    print(json.dumps(loc, indent=1))

    # ---- panels, every one chosen by rule ----
    rows = list(csv.DictReader(open(PER_IMAGE, encoding="utf-8")))
    for r in rows:
        r["c"] = float(r["max_conf"]) if r["max_conf"] else 0.0
    ew = [r for r in rows if r["role"] == "ewaste_test"]
    org = [r for r in rows if r["role"] == "organic_test"]
    nice = {"electrical cables": "cable", "electronic chips": "circuit board",
            "smartphones": "smartphone", "Food Organics": "food", "Vegetation": "vegetation"}

    real = []
    for c in ("electrical cables", "electronic chips", "smartphones"):
        hc = [r for r in ew if r["category"] == c and r["c"] >= thr]
        med = np.median([r["c"] for r in hc])
        real += sorted(hc, key=lambda r: abs(r["c"] - med))[:2]
    order = np.argsort(energy)
    qs = [(2 * i + 1) / 12 for i in range(6)]
    synth = [paths[int(order[int(q * (len(order) - 1))])] for q in qs]
    fas = sorted(org, key=lambda r: -r["c"])[:6]

    groups = [
        ("Real e-waste", [(pad_square(load_image(ROOT / r["path"]))[0], nice[r["category"]],
                           [], "#00B050") for r in real]),
        ("Synthetic", [(Image.open(p).convert("RGB"), "synthetic", val_boxes(p.stem), "#00B050")
                       for p in synth]),
        ("False alarms", [(pad_square(load_image(ROOT / r["path"]))[0], nice[r["category"]],
                           [], "#E0302C") for r in fas]),
    ]

    model = YOLO(str(WEIGHTS))
    cmap = plt.get_cmap("inferno")
    fig, axes = plt.subplots(3, 6, figsize=(184 * MM, 98 * MM))
    fig.subplots_adjust(left=0.035, right=0.925, top=0.99, bottom=0.035, wspace=0.05, hspace=0.2)
    letters = iter("abcdefghijklmnopqr")
    for i, (rname, panels) in enumerate(groups):
        for j, (img, title, gt, colour) in enumerate(panels):
            ax = axes[i, j]
            cam, s, (cy, cx) = cam_fn(img)
            sm = gaussian_filter(cam, 6)
            sm = sm / (sm.max() + 1e-12)
            res = model.predict(np.array(img)[:, :, ::-1], conf=thr, imgsz=SIZE, verbose=False)[0]
            ax.imshow(img)
            ax.imshow(np.zeros_like(sm), cmap="gray", alpha=0.45, vmin=0, vmax=1)
            ax.imshow(np.ma.masked_less(sm, FLOOR), cmap=cmap, alpha=0.85, vmin=0, vmax=1)
            ax.contour(sm, levels=[0.5], colors="white", linewidths=0.6)
            for x0, y0, x1, y1 in res.boxes.xyxy.cpu().numpy():
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.8, ec=colour))
            for x0, y0, x1, y1 in gt:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       lw=0.8, ec="white", ls="--"))
            ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)     # boxes may overhang the frame
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"({next(letters)}) {title}, {s:.2f}", fontsize=6.2, labelpad=2)
        axes[i, 0].set_ylabel(rname, fontsize=7, labelpad=3)
    top, bottom = axes[0, 5].get_position(), axes[2, 5].get_position()
    cax = fig.add_axes([0.935, bottom.y0, 0.011, top.y1 - bottom.y0])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_label("HiResCAM attribution (normalised)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.ax.axhline(FLOOR, color="white", lw=0.6)
    fig.savefig(FIG / "fig13_hirescam.pdf", dpi=300)
    for r in real + fas:
        print(r["category"], round(r["c"], 3), r["path"])
    print("synthetic", [p.name for p in synth], [round(float(energy[order[int(q * (len(order) - 1))]]), 3) for q in qs])


if __name__ == "__main__":
    main()
