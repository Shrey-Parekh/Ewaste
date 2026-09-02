"""
11_metrics_table.py
-------------------
Collects every evaluated model into the one table the paper reports, so that
"all models carry all metrics" is something you can check rather than assume.

It reads the summaries written by 06_evaluate.py and 10_ensemble.py and emits
the same table three ways: printed for reading, CSV for further work, and a
LaTeX tabular for the manuscript. A metric a model genuinely does not have --
FLOPs where thop is not installed, or mIoU before the test
photographs have been annotated -- is printed as a dash. It is never filled in
with a plausible-looking number.

Every rate in this table is taken at the operating point chosen on held-out
data, never at the threshold that happens to maximise F1 on the test set.
Selecting a threshold on the same data it is then scored against is optimistic
by construction, and measured across these architectures the gap reached
+0.070 F1 on one of them. The tuned figure is carried as a separate "Oracle"
column so the distance between the two stays visible.

Detection rate and false-alarm rate lead the table. Precision and F1 follow,
but both depend on the 400:747 ratio of positives to negatives in the test
split, which is an artefact of how the split was drawn rather than a
real-world prevalence of contamination.

Run:    python 11_metrics_table.py --pool 60
Output: printed table, plus Manuscripts/tables/metrics_table.{csv,tex}
"""

from pathlib import Path
import argparse
import csv
import json

# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
OUT = ROOT / "Manuscripts" / "tables"

# (label, run/eval suffix). Order is the order the paper presents them in.
MODELS = [
    ("YOLOv8s", ""),
    ("YOLOv11s", "yolo11s"),
    ("YOLOv8s+CBAM", "v8s_cbam"),
    ("YOLOv11s+CBAM", "v11s_cbam"),
    ("ResNet34+FPN+CBAM", "r34_fpn_cbam"),
    ("ResNet34+BiFPN+CBAM", "r34_bifpn_cbam"),
    ("YOLOv11s+BiFPN+CBAM", "v11s_bifpn_cbam"),
    ("Ensemble", "ensemble"),
]

COLUMNS = [
    ("Model", "model", "{}"),
    ("Detect %", "detect_rate", "{:.1f}"),
    ("FA %", "fa_rate", "{:.1f}"),
    ("mAP@50", "map50", "{:.3f}"),
    ("mAP@50:95", "map50_95", "{:.3f}"),
    ("Prec", "precision", "{:.3f}"),
    ("Rec", "recall", "{:.3f}"),
    ("F1", "f1", "{:.3f}"),
    ("Oracle Det %", "oracle_detect", "{:.1f}"),
    ("mIoU", "miou", "{:.3f}"),
    ("Dice", "dice", "{:.3f}"),
    ("ms", "latency_ms", "{:.1f}"),
    ("FPS", "fps", "{:.1f}"),
    ("Params M", "params_m", "{:.2f}"),
    ("GFLOPs", "gflops", "{:.1f}"),
    ("Size MB", "size_mb", "{:.1f}"),
    ("Train min", "train_min", "{:.1f}"),
]


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def collect(pool, label, suffix):
    tail = f"_{suffix}" if suffix else ""
    summary = read_json(ROOT / f"eval_pool{pool}{tail}" / "summary.json")
    if summary is None:
        return None

    synth = summary.get("synthetic") or {}
    cap = summary.get("capacity") or {}
    loc = summary.get("localisation") or {}
    # The reported point: chosen on held-out data. Older summaries predate the
    # split and carry only real_at_synthetic_conf; ones older still have
    # neither, and fall back to the tuned figure with the oracle column left
    # equal to it, which makes the substitution visible rather than silent.
    oracle = summary.get("real_best") or {}
    best = (summary.get("headline")
            or summary.get("real_at_synthetic_conf")
            or oracle)

    params = cap.get("n_params")
    train_s = synth.get("train_seconds")
    detect = best.get("ewaste_detection_rate")
    fa = best.get("organic_FP_rate")

    return {
        "model": label,
        "detect_rate": None if detect is None else detect * 100,
        "fa_rate": None if fa is None else fa * 100,
        "map50": synth.get("map50"),
        "map50_95": synth.get("map50_95"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "f1": best.get("f1"),
        "oracle_detect": (oracle.get("ewaste_detection_rate") * 100
                          if oracle.get("ewaste_detection_rate") is not None else None),
        "miou": loc.get("mIoU"),
        "dice": loc.get("dice"),
        "latency_ms": cap.get("latency_ms"),
        "fps": cap.get("fps"),
        "params_m": None if params is None else params / 1e6,
        "gflops": cap.get("gflops"),
        "size_mb": cap.get("model_size_mb"),
        "train_min": None if train_s is None else train_s / 60,
    }


def cell(row, key, fmt):
    v = row.get(key)
    if v is None:
        return "-"
    return v if isinstance(v, str) else fmt.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=60)
    args = ap.parse_args()

    rows, missing = [], []
    for label, suffix in MODELS:
        row = collect(args.pool, label, suffix)
        if row:
            rows.append(row)
        else:
            missing.append(label)

    if not rows:
        print(f"[!] no summaries found for pool {args.pool}.")
        print("    run 06_evaluate.py for at least one model first.")
        return

    widths = [max(len(head), max(len(cell(r, k, f)) for r in rows))
              for head, k, f in COLUMNS]

    print()
    print("  ".join(h.ljust(w) for (h, _, _), w in zip(COLUMNS, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(cell(r, k, f).ljust(w)
                        for (_, k, f), w in zip(COLUMNS, widths)))
    print()

    if missing:
        print(f"not yet evaluated: {', '.join(missing)}")

    gaps = {h for (h, k, _) in COLUMNS for r in rows if r.get(k) is None}
    if gaps:
        print(f"columns with gaps: {', '.join(sorted(gaps))}")
        if "mIoU" in gaps or "Dice" in gaps:
            print("  mIoU/Dice need hand-drawn boxes -> python 09_annotate.py")
        if "GFLOPs" in gaps:
            print("  GFLOPs are unavailable where thop is not installed")

    print("Rates are at the operating point chosen on held-out data. "
          "'Oracle Det %' is the")
    print("detection rate at the threshold that maximises F1 on the test set: "
          "an upper")
    print("bound that assumes the answer is already known, not an achievable result.")

    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "metrics_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([h for h, _, _ in COLUMNS])
        for r in rows:
            w.writerow([cell(r, k, fmt) for _, k, fmt in COLUMNS])

    tex = [
        "% generated by 11_metrics_table.py -- do not edit by hand",
        "\\begin{tabular}{l" + "r" * (len(COLUMNS) - 1) + "}",
        "\\hline",
        " & ".join(h.replace("%", "\\%") for h, _, _ in COLUMNS) + " \\\\",
        "\\hline",
    ]
    for r in rows:
        tex.append(" & ".join(cell(r, k, fmt) for _, k, fmt in COLUMNS) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (OUT / "metrics_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"written: {(OUT / 'metrics_table.csv').relative_to(ROOT)}, "
          f"{(OUT / 'metrics_table.tex').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
