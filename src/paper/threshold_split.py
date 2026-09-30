"""
Operating thresholds set on data the reported rates never see.

The 746 organic photographs are split at random into a calibration half and a
held-out half. The alert threshold is set on the calibration half alone with
the split-conformal rule: a photograph raises an alert when fewer than
floor(alpha * (n + 1)) calibration photographs score at or above it, which
keeps the expected false-alarm rate on new clean photographs at or below alpha
(finite-sample guarantee under exchangeability). Detection is then measured on
all 387 e-waste photographs and false alarms on the held-out organic half.
Repeated over many random splits; mean and 2.5/97.5 percentiles reported.

Run from the project root, after paper_figures.py:
    python src/paper/threshold_split.py
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import paper_figures as pf  # noqa: E402

SPLITS = 1000
TARGETS = (0.05, 0.10, 0.15)


def conformal_alerts(cal, scores, alpha):
    """Boolean alerts for `scores` given calibration scores `cal`."""
    n = len(cal)
    m = int(np.floor(alpha * (n + 1))) - 1          # calibration scores allowed >= s
    if m < 0:
        return np.zeros(len(scores), bool)
    top = np.sort(cal)[::-1]
    cut = top[m]                                     # (m+1)-th largest calibration score
    return scores > cut


def cell(v, lo, hi):
    """Percentage with its range beneath, for a LaTeX table."""
    return (r"\makecell{" + f"{100 * v:.1f}" + r"\\{\scriptsize "
            + f"{100 * lo:.1f}-{100 * hi:.1f}" + "}}")


def main():
    rng = np.random.default_rng(0)
    out = {}
    for label, tag, short, fam in pf.ARMS:
        org, ew = pf.load(tag)
        oc = np.array([c for c, _, _ in org])
        ec = np.array([c for c, _, _ in ew])
        res = {t: ([], []) for t in TARGETS}
        for _ in range(SPLITS):
            idx = rng.permutation(len(oc))
            cal, held = oc[idx[:len(oc) // 2]], oc[idx[len(oc) // 2:]]
            for t in TARGETS:
                res[t][0].append(conformal_alerts(cal, ec, t).mean())
                res[t][1].append(conformal_alerts(cal, held, t).mean())
        out[label] = {}
        for t in TARGETS:
            d, f = np.array(res[t][0]), np.array(res[t][1])
            out[label][f"{round(t*100)}"] = {
                "det": d.mean(), "det_lo": np.percentile(d, 2.5), "det_hi": np.percentile(d, 97.5),
                "fa": f.mean(), "fa_lo": np.percentile(f, 2.5), "fa_hi": np.percentile(f, 97.5)}
    Path(__file__).with_name("threshold_split.json").write_text(json.dumps(out, indent=1))
    stats = json.load(open(Path(__file__).with_name("stats.json")))

    # table rows for the detector (YOLOv11s)
    rows = []
    for t in ("5", "10", "15"):
        r = out["YOLOv11s"][t]
        rows.append(" & ".join([
            t + r"\%",
            cell(r["det"], r["det_lo"], r["det_hi"]),
            cell(r["fa"], r["fa_lo"], r["fa_hi"]),
            f"{100 * stats['arms']['YOLOv11s']['det' + t]:.1f}",
        ]) + r" \\")
    tab = Path(__file__).resolve().parents[2] / "Manuscripts" / "latex" / "tables"
    (tab / "tab_calib_rows.tex").write_text("\n".join(rows) + "\n")
    print(f"{'detector':<26}{'target':>7}{'det (split)':>22}{'FA held-out':>20}{'det (old)':>11}")
    for label in out:
        for t in ("5", "10", "15"):
            r = out[label][t]
            old = stats["arms"][label][f"det{t}"]
            print(f"{label:<26}{t+'%':>7}  {100*r['det']:5.1f} ({100*r['det_lo']:4.1f}-{100*r['det_hi']:4.1f})"
                  f"   {100*r['fa']:4.1f} ({100*r['fa_lo']:4.1f}-{100*r['fa_hi']:4.1f})   {100*old:5.1f}")


if __name__ == "__main__":
    main()
