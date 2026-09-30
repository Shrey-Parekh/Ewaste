"""
Statistics, figures and tables for the manuscript, from the evaluation files
already on disk. Nothing is re-run: every number comes from
evaluation/pool54*/per_image.csv, evaluation/pool54*/summary.json, runs/detect/*/results.csv
and latency_pool54.json.

Operating point. Each detector is compared at the confidence threshold that
maximises its detection rate while holding the false-alarm rate on the 746
organic photographs at or below a target (5, 10 or 15%). The matched rates are
cross-checked against src/11_metrics_table.py.

Paired test. At the 10% operating point every e-waste photograph is detected
or missed by each detector, so two detectors are compared with McNemar's exact
test on the discordant photographs, Holm-corrected within each family.

Run from the project root:
    python src/paper/paper_figures.py
"""
from pathlib import Path
from math import comb, sqrt
import csv
import importlib.util
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "Manuscripts" / "latex" / "figures"
TAB = ROOT / "Manuscripts" / "latex" / "tables"
MM = 1 / 25.4
TARGETS = (0.05, 0.10, 0.15)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "mathtext.fontset": "stix", "axes.unicode_minus": False,
})

# label, eval/run tag, short name, family
ARMS = [
    ("YOLOv8s", "", "YOLOv8s", "yolo"),
    ("YOLOv11s", "yolo11s", "YOLOv11s", "yolo"),
    ("YOLOv8s+CBAM", "v8s_cbam", "YOLOv8s+CBAM", "yolo"),
    ("YOLOv11s+CBAM", "v11s_cbam", "YOLOv11s+CBAM", "yolo"),
    ("ResNet18+FPN+CBAM", "r18_fpn_cbam", "ResNet18+FPN", "tv"),
    ("ResNet18+BiFPN+CBAM", "r18_bifpn_cbam", "ResNet18+BiFPN", "tv"),
    ("GoogLeNet+FPN+CBAM", "gnet_fpn_cbam", "GoogLeNet+FPN", "tv"),
    ("GoogLeNet+BiFPN+CBAM", "gnet_bifpn_cbam", "GoogLeNet+BiFPN", "tv"),
    ("EfficientNet+FPN+CBAM", "effnet_fpn_cbam", "EfficientNet+FPN", "tv"),
    ("EfficientNet+BiFPN+CBAM", "effnet_bifpn_cbam", "EfficientNet+BiFPN", "tv"),
    ("YOLOv11s+BiFPN+CBAM", "v11s_bifpn_cbam", "YOLOv11s+BiFPN", "yolo"),
    ("Ensemble", "ensemble", "Ensemble", "ens"),
]
# One hue per backbone, shared by every figure (validated: each pair drawn
# together is >= 15 dE apart in normal vision and separable under CVD).
# The modification is carried by line style or marker, never by a new hue:
# dashed line / triangle = CBAM added to a YOLO model, dash-dot / open marker =
# BiFPN neck. The short names are the ones used in the tables.
BLUE, GREEN, VERM, PINK, GOLD, INK = "#0072B2", "#009E73", "#D55E00", "#B8559A", "#A67C00", "#1A1A1A"
COLOURS = {
    "YOLOv11s": BLUE, "YOLOv11s+CBAM": BLUE, "YOLOv11s+BiFPN+CBAM": BLUE,
    "YOLOv8s": GREEN, "YOLOv8s+CBAM": GREEN,
    "ResNet18+FPN+CBAM": VERM, "ResNet18+BiFPN+CBAM": VERM,
    "GoogLeNet+FPN+CBAM": PINK, "GoogLeNet+BiFPN+CBAM": PINK,
    "EfficientNet+FPN+CBAM": GOLD, "EfficientNet+BiFPN+CBAM": GOLD,
    "Ensemble": INK,
}
LABELS = [a[0] for a in ARMS]
SHORT = {a[0]: a[2] for a in ARMS}


def eval_dir(tag):
    return ROOT / "evaluation" / ("pool54" + (f"_{tag}" if tag else ""))


def run_dir(tag):
    return ROOT / "runs" / "detect" / ("pool54" + (f"_{tag}" if tag else ""))


def load_metrics_module():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("mt", ROOT / "src" / "11_metrics_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = np.argsort(ps)
    m, out, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


def paired_diff_ci(gain, loss, n, z=1.96):
    """Change in detection rate (variant minus reference), Wald CI for paired data."""
    d = (gain - loss) / n
    se = sqrt(max(gain + loss - (gain - loss) ** 2 / n, 0)) / n
    return d, d - z * se, d + z * se


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


# ------------------------------------------------------------------ data
def load(tag):
    org, ew = [], []
    with open(eval_dir(tag) / "per_image.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = float(r["max_conf"]) if r["max_conf"] else -1.0   # never fired
            (org if r["role"] == "organic_test" else ew).append((c, r["category"], r["path"]))
    return org, ew


def operating_point(org, ew, target):
    """Lowest threshold whose false-alarm rate stays within target."""
    oc = np.array([c for c, _, _ in org])
    ec = np.array([c for c, _, _ in ew])
    for t in np.unique(np.concatenate([oc[oc >= 0], ec[ec >= 0], [np.inf]])):
        if (oc >= t).mean() <= target:
            return float(t), ec >= t, oc >= t
    raise AssertionError("unreachable")


def roc_curve(org, ew):
    oc = np.array([c for c, _, _ in org])
    ec = np.array([c for c, _, _ in ew])
    ts = np.unique(np.concatenate([oc[oc >= 0], ec[ec >= 0]]))[::-1]
    fa = [0.0] + [(oc >= t).mean() for t in ts]
    de = [0.0] + [(ec >= t).mean() for t in ts]
    return np.array(fa), np.array(de)


def auc_partial(fa, de, limit=0.15):
    """
    Area under detection against false alarms up to `limit`, scaled to [0, 1].
    The curve is the empirical step function; the area is taken with the
    trapezoidal rule over its vertices, closed at `limit` at the last level.
    """
    keep = fa <= limit
    x = np.append(fa[keep], limit)
    y = np.append(de[keep], de[keep][-1])
    return float(np.trapezoid(y, x) / limit)


def auc_bootstrap(org, ew, reps=2000, seed=0):
    """
    Percentile 95% interval for pAUC15. Organic and e-waste photographs are
    resampled independently, with replacement, keeping each set's size.
    """
    rng = np.random.default_rng(seed)
    oc = np.array([c for c, _, _ in org])
    ec = np.array([c for c, _, _ in ew])
    vals = []
    for _ in range(reps):
        o = oc[rng.integers(0, len(oc), len(oc))]
        e = ec[rng.integers(0, len(ec), len(ec))]
        fa, de = roc_curve([(c, None, None) for c in o], [(c, None, None) for c in e])
        vals.append(auc_partial(fa, de))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main():
    mt = load_metrics_module()
    latency = json.load(open(ROOT / "latency_pool54.json"))["arms"]
    rows = {}
    for label, tag, short, fam in ARMS:
        org, ew = load(tag)
        s = json.load(open(eval_dir(tag) / "summary.json"))
        fa_, de_ = roc_curve(org, ew)
        r = {"label": label, "family": fam, "n_org": len(org), "n_ew": len(ew),
             "auc15": auc_partial(fa_, de_), "auc15_ci": auc_bootstrap(org, ew),
             "fa": fa_, "de": de_}
        for t in TARGETS:
            thr, hit, fp = operating_point(org, ew, t)
            r[f"thr{round(t*100)}"], r[f"hit{round(t*100)}"], r[f"fp{round(t*100)}"] = thr, hit, fp
        ref, _ = mt.detection_at_fa(eval_dir(tag))
        for t in TARGETS:
            assert abs(ref[t] - r[f"hit{round(t*100)}"].mean()) < 1e-9, (label, t)
        h = s["headline"]
        r["syn_thr"], r["syn_det"], r["syn_fa"] = (h["confidence"], h["ewaste_detection_rate"],
                                                   h["organic_FP_rate"])
        syn = s.get("synthetic") or {}
        r["map50"], r["map5095"] = syn.get("map50"), syn.get("map50_95")
        r["params"] = s["capacity"]["n_params"] / 1e6
        r["gflops"] = s["capacity"].get("gflops")
        r["size"] = s["capacity"]["model_size_mb"]
        r["train_min"] = syn["train_seconds"] / 60 if syn.get("train_seconds") else None
        r["latency"] = latency[tag]["latency_ms"] if tag in latency else s["capacity"]["latency_ms"]
        r["cats_ew"] = [c for _, c, _ in ew]
        r["cats_org"] = [c for _, c, _ in org]
        r["paths_ew"] = [p for _, _, p in ew]
        r["paths_org"] = [p for _, _, p in org]
        if tag != "ensemble":
            res = list(csv.DictReader(open(run_dir(tag) / "results.csv")))
            key = [k for k in res[0] if "mAP50-95" in k][0]
            r["curve"] = np.array([float(x[key]) for x in res])
        rows[label] = r

    n_ew, n_org = rows["YOLOv8s"]["n_ew"], rows["YOLOv8s"]["n_org"]
    for l in LABELS:
        assert rows[l]["paths_ew"] == rows["YOLOv8s"]["paths_ew"], "image order differs"
        assert rows[l]["paths_org"] == rows["YOLOv8s"]["paths_org"], "image order differs"

    # ------------------------------------------------------ pairwise McNemar
    pairs, raw = [], []
    for i, a in enumerate(LABELS):
        for b_ in LABELS[i + 1:]:
            ha, hb = rows[a]["hit10"], rows[b_]["hit10"]
            b, c = int((ha & ~hb).sum()), int((~ha & hb).sum())
            pairs.append((a, b_, b, c)); raw.append(mcnemar_exact(b, c))
    adj = holm(raw)
    pmat = {(a, b_): (p, q, b, c) for (a, b_, b, c), p, q in zip(pairs, raw, adj)}

    def pair(a, b_):
        """(p, holm p, a-only hits, b-only hits)."""
        if (a, b_) in pmat:
            return pmat[(a, b_)]
        p, q, b, c = pmat[(b_, a)]
        return p, q, c, b

    # ------------------------------------------------------ ablation contrasts
    ABL = [
        ("Attention", "YOLOv8s", "YOLOv8s+CBAM", "CBAM added to YOLOv8s"),
        ("Attention", "YOLOv11s", "YOLOv11s+CBAM", "CBAM added to YOLOv11s"),
        ("Neck", "ResNet18+FPN+CBAM", "ResNet18+BiFPN+CBAM", "FPN to BiFPN, ResNet18"),
        ("Neck", "GoogLeNet+FPN+CBAM", "GoogLeNet+BiFPN+CBAM", "FPN to BiFPN, GoogLeNet"),
        ("Neck", "EfficientNet+FPN+CBAM", "EfficientNet+BiFPN+CBAM", "FPN to BiFPN, EfficientNet-B0"),
        ("Neck", "YOLOv11s+CBAM", "YOLOv11s+BiFPN+CBAM", "PAN to BiFPN, YOLOv11s"),
        ("Backbone", "YOLOv11s+BiFPN+CBAM", "ResNet18+BiFPN+CBAM", "YOLOv11s backbone to ResNet18"),
        ("Backbone", "YOLOv11s+BiFPN+CBAM", "GoogLeNet+BiFPN+CBAM", "YOLOv11s backbone to GoogLeNet"),
        ("Backbone", "YOLOv11s+BiFPN+CBAM", "EfficientNet+BiFPN+CBAM", "YOLOv11s backbone to EfficientNet-B0"),
        ("Ensembling", "YOLOv11s", "Ensemble", "YOLOv11s to 11-model WBF"),
    ]
    abl_rows = []
    for fam, a, b_, desc in ABL:
        _, _, loss, gain = pair(a, b_)     # loss: reference-only hits; gain: variant-only
        d, lo, hi = paired_diff_ci(gain, loss, n_ew)
        abl_rows.append(dict(fam=fam, ref=a, var=b_, desc=desc, d=d, lo=lo, hi=hi,
                             gain=gain, loss=loss, p=mcnemar_exact(gain, loss)))
    for a, q in zip(abl_rows, holm([a["p"] for a in abl_rows])):
        a["q"] = q

    # ------------------------------------------------------ categories, correlation
    ew_cats = sorted(set(rows["YOLOv8s"]["cats_ew"]))
    org_cats = sorted(set(rows["YOLOv8s"]["cats_org"]))
    catinfo = {}
    for l in LABELS:
        r = rows[l]
        ce, co = np.array(r["cats_ew"]), np.array(r["cats_org"])
        catinfo[l] = {c: (int(r["hit10"][ce == c].sum()), int((ce == c).sum())) for c in ew_cats}
        catinfo[l].update({c: (int(r["fp10"][co == c].sum()), int((co == c).sum())) for c in org_cats})
    singles = LABELS[:-1]

    # ------------------------------------------------------ JSON for the text
    out = {"n_ew": n_ew, "n_org": n_org, "arms": {}, "ablation": abl_rows, "pairs": {},
           "categories": catinfo}
    for l in LABELS:
        r = rows[l]
        k10 = int(r["hit10"].sum())
        out["arms"][l] = {
            "det5": r["hit5"].mean(), "det10": r["hit10"].mean(), "det15": r["hit15"].mean(),
            "fa5": r["fp5"].mean(), "fa10": r["fp10"].mean(), "fa15": r["fp15"].mean(),
            "thr10": r["thr10"], "k10": k10,
            "det5_ci": wilson(int(r["hit5"].sum()), n_ew), "det10_ci": wilson(k10, n_ew),
            "det15_ci": wilson(int(r["hit15"].sum()), n_ew),
            "auc15": r["auc15"], "auc15_ci": r["auc15_ci"], "syn_thr": r["syn_thr"], "syn_det": r["syn_det"],
            "syn_fa": r["syn_fa"], "map50": r["map50"], "map5095": r["map5095"],
            "latency": r["latency"], "params": r["params"],
            "best_epoch": int(np.argmax(r["curve"])) + 1 if "curve" in r else None,
        }
    for (a, b_, b, c), p, q in zip(pairs, raw, adj):
        out["pairs"][f"{a} | {b_}"] = dict(p=p, holm=q, a_only=b, b_only=c)
    (Path(__file__).parent / "stats.json").write_text(json.dumps(out, indent=1, default=float))

    # ------------------------------------------------------ figures
    colours = COLOURS
    bifpn = {l: "BiFPN" in l for l in LABELS}

    def style(ax, grid_y=True):
        ax.grid(axis="y" if grid_y else "both", color="#e6e6e6", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    def panel(ax, letter, text, x=-0.02):
        ax.text(x, 1.02, f"({letter})", transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="right", va="bottom")
        ax.text(x + 0.01, 1.02, text, transform=ax.transAxes, fontsize=7,
                ha="left", va="bottom")

    def label_points(ax, pts, offsets, default=(5, 3)):
        for l, (x, y) in pts.items():
            dx, dy = offsets.get(l, default)
            far = abs(dx) > 9 or abs(dy) > 9
            ax.annotate(SHORT[l], (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=5.8, ha="left" if dx >= 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.35, color="#777777",
                                        shrinkA=0, shrinkB=2.5) if far else None)

    def marker(l):
        if l == "Ensemble":
            return "D"
        if rows[l]["family"] == "tv":
            return "s"
        return "^" if l in ("YOLOv8s+CBAM", "YOLOv11s+CBAM") else "o"

    # Figs 7 and 8 share one layout: six small panels, one per ablation
    # question, each with YOLOv11s as the solid blue anchor and at most three
    # other lines. Hue follows the backbone (validated palette: every pair that
    # shares a panel is >= 15 dE apart in normal vision and separable under
    # CVD); line style follows the change: dashed = CBAM added to a YOLO model,
    # dash-dot = BiFPN neck.
    DASH, DADO = (0, (4, 1.6)), (0, (5, 1.4, 1.2, 1.4))
    anchor = ("YOLOv11s", BLUE, "-")
    small = [
        ("a", "Baselines and ensemble",
         [anchor, ("YOLOv8s", GREEN, "-"), ("Ensemble", INK, "-")]),
        ("b", "Attention: adding CBAM",
         [anchor, ("YOLOv11s+CBAM", BLUE, DASH), ("YOLOv8s", GREEN, "-"),
          ("YOLOv8s+CBAM", GREEN, DASH)]),
        ("c", "Neck on YOLOv11s: BiFPN",
         [anchor, ("YOLOv11s+BiFPN+CBAM", BLUE, DADO)]),
        ("d", "ResNet18 backbone",
         [anchor, ("ResNet18+FPN+CBAM", VERM, "-"), ("ResNet18+BiFPN+CBAM", VERM, DADO)]),
        ("e", "GoogLeNet backbone",
         [anchor, ("GoogLeNet+FPN+CBAM", PINK, "-"), ("GoogLeNet+BiFPN+CBAM", PINK, DADO)]),
        ("f", "EfficientNet-B0 backbone",
         [anchor, ("EfficientNet+FPN+CBAM", GOLD, "-"), ("EfficientNet+BiFPN+CBAM", GOLD, DADO)]),
    ]
    name = {"YOLOv11s+CBAM": "YOLOv11s + CBAM", "YOLOv8s+CBAM": "YOLOv8s + CBAM",
            "YOLOv11s+BiFPN+CBAM": "YOLOv11s + BiFPN + CBAM",
            "ResNet18+FPN+CBAM": "FPN + CBAM", "ResNet18+BiFPN+CBAM": "BiFPN + CBAM",
            "GoogLeNet+FPN+CBAM": "FPN + CBAM", "GoogLeNet+BiFPN+CBAM": "BiFPN + CBAM",
            "EfficientNet+FPN+CBAM": "FPN + CBAM", "EfficientNet+BiFPN+CBAM": "BiFPN + CBAM"}

    def grid(height):
        fig, axes = plt.subplots(2, 3, figsize=(184 * MM, height * MM), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.1, wspace=0.08, hspace=0.3)
        return fig, axes

    def small_title(ax, letter, text):
        ax.set_title(f"({letter}) {text}", loc="left", fontsize=7, pad=3)

    def lw_of(l):
        return 1.7 if l == "YOLOv11s" else 1.15

    def leg(ax, **kw):
        ax.legend(fontsize=5.8, handlelength=3.0, frameon=True, framealpha=0.95,
                  edgecolor="none", borderpad=0.3, labelspacing=0.3, **kw)

    # training curves: 7-epoch running mean, peak epoch marked
    fig, axes = grid(92)
    k = np.ones(7) / 7
    for ax, (letter, title, lines) in zip(axes.flat, small):
        style(ax)
        ax.axvspan(136, 150, color="#efefef", lw=0, zorder=0)
        for l, col, ls in lines:
            if "curve" not in rows[l]:          # the ensemble is not trained
                continue
            c = rows[l]["curve"]
            e = np.arange(1, len(c) + 1)
            sm = np.convolve(np.pad(c, 3, mode="edge"), k, mode="valid")
            ax.plot(e, sm, color=col, lw=lw_of(l), ls=ls, label=name.get(l, l),
                    zorder=3 if l != "YOLOv11s" else 4, solid_capstyle="round")
            b = int(np.argmax(c))
            ax.plot(b + 1, sm[b], "o", ms=3.8, color=col, mec="white", mew=0.7, zorder=5)
        ax.set_xlim(1, 150); ax.set_ylim(0.10, 0.40)
        ax.set_xticks([1, 25, 50, 75, 100, 125, 150])
        small_title(ax, letter, title)
        leg(ax, loc="lower right", bbox_to_anchor=(0.9, 0.0))
        ax.text(143, 0.105, "no mosaic", rotation=90, ha="center", va="bottom",
                fontsize=5.3, color="#6b6b6b")
    for ax in axes[:, 0]:
        ax.set_ylabel("Synthetic val. mAP@0.50:0.95")
    for ax in axes[1]:
        ax.set_xlabel("Epoch of the main phase")
    fig.savefig(FIG / "fig07_training_curves.pdf")
    plt.close(fig)

    # operating curves: detection against false alarms, threshold swept
    fig, axes = grid(112)
    for ax, (letter, title, lines) in zip(axes.flat, small):
        style(ax, grid_y=False)
        ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
        for t in TARGETS:
            ax.axvline(t * 100, color="#9e9e9e", lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        for l, col, ls in lines:
            r = rows[l]
            ax.step(r["fa"] * 100, r["de"] * 100, where="post", color=col, ls=ls,
                    lw=lw_of(l), label=name.get(l, l), zorder=3 if l != "YOLOv11s" else 4)
            ax.plot(r["syn_fa"] * 100, r["syn_det"] * 100, "o", ms=4.0, mfc="white",
                    mec=col, mew=1.0, zorder=5)
        ax.set_xlim(0, 30); ax.set_ylim(20, 100)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        small_title(ax, letter, title)
        leg(ax, loc="lower right")
    axes[0, 0].plot([], [], "o", ms=4.0, mfc="white", mec="#555555", mew=1.0,
                    label="own synthetic threshold")
    leg(axes[0, 0], loc="lower right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Detection, real e-waste (%)")
    for ax in axes[1]:
        ax.set_xlabel("False alarms, real organic waste (%)")
    fig.savefig(FIG / "fig08_operating_curves.pdf")
    plt.close(fig)

    # ablation forest (a) and pairwise significance matrix (b)
    fig = plt.figure(figsize=(184 * MM, 96 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.36,
                          left=0.215, right=0.99, top=0.93, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    fams = ["Attention", "Neck", "Backbone", "Ensembling"]
    heads = {"Attention": "Attention (CBAM)", "Neck": "Neck topology",
             "Backbone": "Backbone (BiFPN + CBAM fixed)", "Ensembling": "Ensembling"}
    ys = np.arange(len(abl_rows))[::-1]
    for fi, f in enumerate(fams):
        idx = [y for y, a in zip(ys, abl_rows) if a["fam"] == f]
        if fi % 2 == 0:
            ax.axhspan(min(idx) - 0.5, max(idx) + 0.5, color="#f2f2f2", lw=0, zorder=0)
        ax.text(-31, max(idx) + 0.47, heads[f], ha="left", va="top", fontsize=5.6,
                fontweight="bold", color="#4a4a4a")
    for y, a in zip(ys, abl_rows):
        sig = a["q"] < 0.05
        col = INK if sig else "#8a8a8a"
        ax.plot([a["lo"] * 100, a["hi"] * 100], [y, y], color=col, lw=1.3, solid_capstyle="butt")
        ax.plot(a["d"] * 100, y, "o", ms=4.6, color=col, mfc=col if sig else "white", mew=1.1, zorder=3)
        star = ("***" if a["q"] < 0.001 else "**" if a["q"] < 0.01 else "*" if sig else "n.s.")
        ax.text(31, y, f"{100 * a['d']:+.1f} {star}", va="center", ha="right", fontsize=5.8,
                color=INK if sig else "#6b6b6b", fontweight="bold" if sig else "normal")
    ax.axvline(0, color="#444444", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([a["desc"] for a in abl_rows], fontsize=6)
    ax.set_ylim(-0.6, len(abl_rows) - 0.4)
    ax.set_xlim(-32, 32)
    ax.set_xlabel("Change in detection at 10% false alarms (points)")
    ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    panel(ax, "a", "Ablation contrasts with paired 95% intervals")

    order = sorted(LABELS, key=lambda l: -rows[l]["hit10"].mean())
    n = len(order)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b_ in enumerate(order):
            if i > j:
                M[i, j] = pair(a, b_)[1]
    ax = fig.add_subplot(gs[1])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "ylgnbu_light", plt.get_cmap("YlGnBu")(np.linspace(0, 0.62, 256)))
    val = -np.log10(np.clip(M, 1e-8, 1))
    im = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, np.ma.masked_invalid(val),
                       cmap=cmap, vmin=0, vmax=8, edgecolors="white", linewidth=0.8)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5); ax.set_aspect("equal")
    for i in range(n):
        for j in range(i):
            d = (rows[order[i]]["hit10"].mean() - rows[order[j]]["hit10"].mean()) * 100
            sig = M[i, j] < 0.05
            ax.text(j, i, f"{d:.1f}" + ("*" if sig else ""), ha="center", va="center",
                    fontsize=5.2, fontweight="bold" if sig else "normal",
                    color="#1a1a1a")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[l] for l in order], rotation=50, ha="right", fontsize=5.8,
                       rotation_mode="anchor")
    ax.set_yticklabels([SHORT[l] for l in order], fontsize=5.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, "b", "All pairs: row minus column (points)")
    cax = ax.inset_axes([0.5, 0.80, 0.46, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1.3, 4, 8])
    cb.ax.set_xticklabels(["1", "0.05", "$10^{-4}$", "$10^{-8}$"], fontsize=5.2)
    cb.ax.tick_params(length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.set_title("Holm-adjusted McNemar p  (* p < 0.05)", fontsize=5.8, pad=2)
    fig.savefig(FIG / "fig10_ablation_and_pairs.pdf")
    plt.close(fig)

    # synthetic validation against real detection
    fig, axes = plt.subplots(1, 2, figsize=(184 * MM, 66 * MM), sharey=True)
    offs = {
        "map5095": {"YOLOv8s+CBAM": (6, 1), "GoogLeNet+BiFPN+CBAM": (-6, -2),
                    "YOLOv11s+CBAM": (-6, -1), "ResNet18+BiFPN+CBAM": (-6, 0),
                    "YOLOv11s+BiFPN+CBAM": (6, -1), "EfficientNet+BiFPN+CBAM": (-5, -10),
                    "EfficientNet+FPN+CBAM": (-6, 2), "YOLOv11s": (-6, 0), "YOLOv8s": (6, 1),
                    "ResNet18+FPN+CBAM": (6, 0), "GoogLeNet+FPN+CBAM": (6, 0)},
        "map50": {"YOLOv8s+CBAM": (6, 0), "GoogLeNet+BiFPN+CBAM": (6, 3),
                  "YOLOv11s+CBAM": (-6, -3), "ResNet18+BiFPN+CBAM": (-6, 0),
                  "YOLOv11s+BiFPN+CBAM": (6, 0), "EfficientNet+BiFPN+CBAM": (-6, 3),
                  "EfficientNet+FPN+CBAM": (-6, 0), "YOLOv11s": (6, 0), "YOLOv8s": (-6, 3),
                  "ResNet18+FPN+CBAM": (-6, -3), "GoogLeNet+FPN+CBAM": (6, 0)},
    }
    for ax, key, xname, letter, ptitle in (
            (axes[0], "map5095", "Synthetic validation mAP@0.50:0.95", "a",
             "against synthetic mAP@0.50:0.95"),
            (axes[1], "map50", "Synthetic validation mAP@0.50", "b",
             "against synthetic mAP@0.50")):
        style(ax, grid_y=False)
        pts = {}
        for l in singles:
            r = rows[l]
            pts[l] = (r[key], r["hit10"].mean() * 100)
            ax.plot(*pts[l], marker(l), ms=5, color=colours[l],
                    mfc="white" if bifpn[l] else colours[l], mew=1.1, zorder=3)
        label_points(ax, pts, offs[key])
        xs = [p[0] for p in pts.values()]
        pad = (max(xs) - min(xs)) * 0.24
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(63, 88)
        ax.set_xlabel(xname)
        panel(ax, letter, f"Real detection {ptitle}")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4.5, color="#555555", label="YOLO model"),
               plt.Line2D([], [], ls="", marker="^", ms=4.8, color="#555555", label="YOLO model + CBAM"),
               plt.Line2D([], [], ls="", marker="s", ms=4.5, color="#555555",
                          label="torchvision backbone + CBAM"),
               plt.Line2D([], [], ls="", marker="o", ms=4.5, color="#555555", mfc="white",
                          label="open marker: BiFPN neck")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=5.8, frameon=True,
                   framealpha=0.95, edgecolor="none")
    axes[0].set_ylabel("Real detection at 10% false alarms (%)")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(FIG / "fig11_synthetic_vs_real.pdf")
    plt.close(fig)

    # accuracy against cost, with a broken latency axis for the ensemble
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(90 * MM, 72 * MM), sharey=True,
                                  gridspec_kw=dict(width_ratios=[4.2, 1], wspace=0.07))
    offs = {"YOLOv8s": (-6, 2), "ResNet18+FPN+CBAM": (6, -2), "YOLOv11s": (6, 1),
            "YOLOv8s+CBAM": (6, 2), "YOLOv11s+CBAM": (-6, -2), "YOLOv11s+BiFPN+CBAM": (6, -1),
            "ResNet18+BiFPN+CBAM": (-6, 0), "GoogLeNet+FPN+CBAM": (-6, 2),
            "EfficientNet+FPN+CBAM": (-6, 5), "EfficientNet+BiFPN+CBAM": (-4, -10),
            "GoogLeNet+BiFPN+CBAM": (6, 0)}
    for a_ in (ax, ax2):
        style(a_, grid_y=True)
    pts = {}
    for l in LABELS:
        r = rows[l]
        a_ = ax2 if l == "Ensemble" else ax
        pts[l] = (r["latency"], r["hit10"].mean() * 100)
        a_.scatter(*pts[l], s=10 * np.sqrt(r["params"]), marker=marker(l),
                   color="white" if bifpn[l] else colours[l], edgecolor=colours[l],
                   linewidth=1.0, zorder=3)
    label_points(ax, {l: p for l, p in pts.items() if l != "Ensemble"}, offs)
    ax2.annotate("Ensemble\n(11 models)", pts["Ensemble"], xytext=(0, -14),
                 textcoords="offset points", ha="center", va="top", fontsize=5.8)
    ax.set_xlim(9.3, 14.6); ax2.set_xlim(124, 133)
    ax2.set_xticks([125, 130])
    ax.set_ylim(63, 90)
    ax.spines["right"].set_visible(False); ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    d = 0.015
    ax.plot([1 - d, 1 + d], [-d * 1.5, d * 1.5], transform=ax.transAxes, color="black",
            lw=0.6, clip_on=False)
    ax2.plot([-d * 4.2, d * 4.2], [-d * 1.5, d * 1.5], transform=ax2.transAxes, color="black",
             lw=0.6, clip_on=False)
    fig.text(0.55, 0.02, "Latency per photograph, batch 1 (ms; axis broken)", ha="center",
             fontsize=7)
    ax.set_ylabel("Detection at 10% false alarms (%)")
    for s, lab in ((5, "5 M"), (10, "10 M"), (100, "100 M")):
        ax.scatter([], [], s=10 * np.sqrt(s), color="#dddddd", edgecolor="#777777", lw=0.6, label=lab)
    ax.legend(title="parameters", title_fontsize=5.5, fontsize=5.5, loc="upper right",
              frameon=True, framealpha=0.95, edgecolor="none", ncol=3, columnspacing=0.6,
              handletextpad=0.2, borderpad=0.5)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.14)
    fig.savefig(FIG / "fig12_cost.pdf")
    plt.close(fig)

    # dataset: composites with their derived boxes, and the visible fraction
    from PIL import Image
    prev = sorted((ROOT / "dataset_pool54" / "preview").glob("*.jpg"))[:8]
    fig, axes = plt.subplots(2, 4, figsize=(184 * MM, 92.5 * MM))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.015, hspace=0.015)
    for ax, p in zip(axes.flat, prev):
        ax.imshow(Image.open(p)); ax.axis("off")
    fig.savefig(FIG / "fig04_synthetic_examples.pdf", dpi=300)
    plt.close(fig)

    phi = np.loadtxt(ROOT / "dataset_pool54" / "visible_fraction.csv", skiprows=1)
    fig, ax = plt.subplots(figsize=(90 * MM, 56 * MM))
    style(ax)
    bins = np.linspace(0, 1, 41)
    cnt, edges = np.histogram(phi, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    ax.bar(centres, cnt, width=edges[1] - edges[0], align="center",
           color=np.where(centres < 0.35, "#E8A87C", "#2E6F95"),
           edgecolor="white", linewidth=0.4, zorder=2)
    ax.axvline(0.35, color="#333333", lw=0.8, ls="--", zorder=3)
    med = float(np.median(phi))
    ax.axvline(med, color="#1F4E6B", lw=0.8, ls=":", zorder=3)
    ymax = cnt.max()
    ax.text(0.175, ymax * 0.5, f"Excluded\n$\\varphi$ < 0.35\nn = {(phi < 0.35).sum()}",
            ha="center", va="center", fontsize=6, color="#B5651D")
    ax.text(0.5, ymax * 0.62, f"Labelled candidates\nn = {(phi >= 0.35).sum()}",
            ha="center", va="center", fontsize=6, color="#1F4E6B")
    ax.text(med + 0.012, ymax * 0.95, f"median {med:.2f}", fontsize=5.5, color="#1F4E6B",
            va="top")
    full = int((phi >= 0.999).sum())
    ax.annotate(f"fully visible\nn = {full}", (0.985, cnt[-1] * 0.8), xytext=(-24, 0),
                textcoords="offset points", ha="right", va="center", fontsize=5.5,
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#777777"))
    ax.set_xlim(0, 1.01); ax.set_ylim(0, ymax * 1.05)
    ax.set_xlabel("Visible fraction $\\varphi$ of a placed object")
    ax.set_ylabel("Objects")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig05_visible_fraction.pdf")
    plt.close(fig)
    print(f"phi: n={len(phi)} median={np.median(phi):.3f} mean={phi.mean():.3f} "
          f"p10={np.quantile(phi, .1):.3f} p90={np.quantile(phi, .9):.3f} kept={(phi >= .35).sum()} "
          f"fully visible={full}")

    # ------------------------------------------------------ table rows
    def pc(v):
        return f"{100 * v:.1f}"

    def name(l):
        return SHORT[l].replace("+", "\\,+\\,")

    best = {k: max(rows[l][k].mean() for l in LABELS) for k in ("hit5", "hit10", "hit15")}

    def bold(l, k):
        s = pc(rows[l][k].mean())
        return f"\\textbf{{{s}}}" if abs(rows[l][k].mean() - best[k]) < 1e-12 else s

    def cell(v, lo, hi):
        """Value with its interval beneath, in brackets and grey so it reads as secondary."""
        return f"\\makecell{{{v}\\\\\\textcolor{{black!60}}{{\\scriptsize [{lo},\\,{hi}]}}}}"

    def with_ci(l, k):
        lo, hi = wilson(int(rows[l][k].sum()), n_ew)
        return cell(bold(l, k), pc(lo), pc(hi))

    cal_path = Path(__file__).parent / "threshold_split.json"
    cal = json.load(open(cal_path)) if cal_path.exists() else {}

    def calib(l):
        if l not in cal:
            return "n/a"
        r = cal[l]["10"]
        return cell(f"{100 * r['det']:.1f}", f"{100 * r['det_lo']:.1f}", f"{100 * r['det_hi']:.1f}")

    # rows grouped the way the ablation reads: baselines, YOLO variants,
    # torchvision backbones, then the ensemble
    groups = [
        ("YOLO baselines", ["YOLOv8s", "YOLOv11s"]),
        ("YOLO with added modules", ["YOLOv8s+CBAM", "YOLOv11s+CBAM", "YOLOv11s+BiFPN+CBAM"]),
        ("Torchvision backbones (with CBAM)", [l for l in LABELS if rows[l]["family"] == "tv"]),
        ("Ensemble of all eleven", ["Ensemble"]),
    ]
    lines = []
    for gi, (gname, members) in enumerate(groups):
        if gi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{gname}}}}} \\\\")
        for l in members:
            r = rows[l]
            alo, ahi = r["auc15_ci"]
            label = f"\\textbf{{{name(l)}}} (proposed)" if l == "YOLOv11s" else name(l)
            lines.append(" & ".join([
                "\\hspace{2mm}" + label, with_ci(l, "hit5"), with_ci(l, "hit10"), with_ci(l, "hit15"),
                cell(f"{r['auc15']:.3f}", f"{alo:.3f}", f"{ahi:.3f}"),
                calib(l), pc(r["syn_det"]), pc(r["syn_fa"]),
                "n/a" if r["map50"] is None else f"{r['map50']:.3f}",
                "n/a" if r["map5095"] is None else f"{r['map5095']:.3f}",
            ]) + " \\\\")
    (TAB / "tab_main_results_rows.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for l in LABELS:
        r = rows[l]
        lines.append(" & ".join([
            name(l), f"{r['params']:.2f}", "n/a" if r["gflops"] is None else f"{r['gflops']:.1f}",
            f"{r['size']:.1f}", f"{r['latency']:.1f}", f"{1000 / r['latency']:.0f}",
            "n/a" if r["train_min"] is None else f"{r['train_min']:.0f}",
            "n/a" if "curve" not in r else str(int(np.argmax(r["curve"])) + 1),
        ]) + " \\\\")
    (TAB / "tab_cost_rows.tex").write_text("\n".join(lines) + "\n")

    lines, last = [], None
    for a in abl_rows:
        if last and a["fam"] != last:
            lines.append("\\addlinespace")
        first = a["fam"] != last
        last = a["fam"]
        sig = a["q"] < 0.05
        b = (lambda s: f"\\textbf{{{s}}}") if sig else (lambda s: s)
        p = "p < 0.001" if a["q"] < 0.001 else f"{a['q']:.3f}"
        lines.append(" & ".join([
            a["fam"] if first else "", a["desc"],
            pc(rows[a["ref"]]["hit10"].mean()), pc(rows[a["var"]]["hit10"].mean()),
            b(f"{100 * a['d']:+.1f}") + f" \\textcolor{{black!60}}{{\\small [{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}]}}",
            f"{a['gain']} / {a['loss']}", b(p),
        ]) + " \\\\")
    (TAB / "tab_ablation_rows.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for l in LABELS:
        cells = [name(l)]
        for c in ew_cats + org_cats:
            k, m = catinfo[l][c]
            cells.append(pc(k / m))
        lines.append(" & ".join(cells) + " \\\\")
    (TAB / "tab_categories_rows.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------ console summary
    print("category sizes:", {c: catinfo["YOLOv8s"][c][1] for c in ew_cats + org_cats})
    for a in abl_rows:
        print(f"{a['desc']:<34} {100*a['d']:+6.1f} [{100*a['lo']:+.1f},{100*a['hi']:+.1f}] "
              f"+{a['gain']}/-{a['loss']} p={a['p']:.2e} holm={a['q']:.2e}")
    for l in order:
        r = rows[l]
        print(f"{l:<26} det5 {100*r['hit5'].mean():5.1f} det10 {100*r['hit10'].mean():5.1f} "
              f"thr10 {r['thr10']:.3f} fa10 {100*r['fp10'].mean():4.1f} auc15 {r['auc15']:.3f} "
              f"[{r['auc15_ci'][0]:.3f},{r['auc15_ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
