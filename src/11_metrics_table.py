"""
11_metrics_table.py
-------------------
Collects every evaluated model into the one table the paper reports, so that
"all models carry all metrics" is something you can check rather than assume.

It reads the summaries written by 06_evaluate.py and 10_ensemble.py and emits
the same table three ways: printed for reading, CSV for further work, and a
LaTeX tabular for the manuscript. A metric a model genuinely does not have --
FLOPs where thop is not installed, training time for the ensemble -- is
printed as a dash. It is never filled in with a plausible-looking number.

What the paper measures is image-level screening: does a photograph that
contains e-waste raise an alarm, and does one that contains none stay quiet.
Where the box lands is not measured and not claimed.

The columns are ordered by how much weight they can bear.

  Detection at a matched false-alarm rate leads. Every arm is read at the same
  false-alarm budget, so differences are attributable to the model. This is a
  comparison protocol: the budget is met using the test set's own false-alarm
  rate, so it is not an operating point anyone could have chosen in advance.

  Detection and false-alarm rate at the synthetic threshold follow. That
  threshold is the argmax of an F1 curve which is nearly flat -- measured
  0.10-0.30 wide within 0.02 of its peak even on leak-free real data -- so
  where it lands is close to arbitrary and these columns are not comparable
  across arms. They are kept because they are what a deployment without real
  calibration data would actually get.

  The oracle column is detection at the test-set-optimal F1 threshold: an
  upper bound that assumes the answer is known, not a result.

Precision and F1 depend on the 400:747 ratio of positives to negatives in the
test split, an artefact of how it was drawn rather than a real prevalence.

Run:    python src/11_metrics_table.py --pool 54
Output: printed table, plus Manuscripts/tables/metrics_table.{csv,tex}
"""

from pathlib import Path
import argparse
import csv
import json

from lib_arms import DEFAULT_POOL, MEMBERS, eval_dir_name, run_name

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
OUT = ROOT / "Manuscripts" / "tables"

# Order is the order the paper presents them in: every arm from the one shared
# list, then the ensemble, which is not a trainable arm.
MODELS = MEMBERS + [("Ensemble", "ensemble")]

# False-alarm budgets the arms are compared at.
MATCHED_FA = (0.05, 0.10, 0.15)

COLUMNS = [
    ("Model", "model", "{}"),
    ("Det@5FA", "det_fa05", "{:.1f}"),
    ("Det@10FA", "det_fa10", "{:.1f}"),
    ("Det@15FA", "det_fa15", "{:.1f}"),
    ("Screen det % (synth thr)", "detect_rate", "{:.1f}"),
    ("FA % (synth thr)", "fa_rate", "{:.1f}"),
    ("Prec", "precision", "{:.3f}"),
    ("Rec", "recall", "{:.3f}"),
    ("F1", "f1", "{:.3f}"),
    ("Oracle det %", "oracle_detect", "{:.1f}"),
    ("mAP@50 (synth)", "map50", "{:.3f}"),
    ("mAP@50:95 (synth)", "map50_95", "{:.3f}"),
    ("ms", "latency_ms", "{:.1f}"),
    ("FPS", "fps", "{:.1f}"),
    ("Params M (fused)", "params_m", "{:.2f}"),
    ("GFLOPs", "gflops", "{:.1f}"),
    ("Size MB", "size_mb", "{:.1f}"),
    ("Epochs run", "epochs_run", "{:d}"),
    ("Best epoch", "best_epoch", "{:d}"),
    ("Train min", "train_min", "{:.1f}"),
]


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def detection_at_fa(eval_dir, targets=MATCHED_FA):
    """
    Detection rate at each target false-alarm rate, from the fine sweep.

    Takes the highest detection rate whose false-alarm rate does not exceed the
    target. The false-alarm rate used is the test set's own, so this is a way
    of comparing arms at equal cost, not a threshold a deployer could have set
    beforehand -- a deployer does not have the test set. Where no threshold in
    the sweep meets the target the entry is None, never an extrapolation.
    """
    path = eval_dir / "threshold_sweep_fine.csv"
    if not path.exists():
        return {t: None for t in targets}
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(float(r["organic_FP_rate"]), float(r["ewaste_detection_rate"]))
                for r in csv.DictReader(f)]
    out = {}
    for t in targets:
        under = [d for fa, d in rows if fa <= t]
        out[t] = max(under) if under else None
    return out


def convergence(pool, suffix):
    """(epochs run, best epoch) from the run's results.csv, or (None, None)."""
    path = ROOT / "runs" / "detect" / run_name(pool, suffix) / "results.csv"
    if not path.exists():
        return None, None
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None, None
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[key]))
    return int(float(rows[-1]["epoch"])), int(float(best["epoch"]))


def load_latency(pool):
    """
    Latency measured for every arm in one interleaved sitting, if it exists.

    The per-arm figures inside each summary.json were each timed in their own
    process on their own day, so they carry whatever the machine was doing at
    the time: two bit-identical runs of one architecture differed by 2.2x.
    14_latency.py re-measures every arm round-robin in one process, and its
    numbers replace those where available. The ensemble is not in that file --
    it is not a trained arm -- so it keeps the figure from its own run.
    """
    data = read_json(ROOT / f"latency_pool{pool}.json")
    return (data or {}).get("arms") or {}


def pct(v):
    return None if v is None else v * 100


def collect(pool, label, suffix, latency=None):
    eval_dir = ROOT / eval_dir_name(pool, suffix)
    summary = read_json(eval_dir / "summary.json")
    if summary is None:
        return None

    synth = summary.get("synthetic") or {}
    cap = summary.get("capacity") or {}
    oracle = summary.get("real_best") or {}
    at_synth = summary.get("headline") or {}

    matched = detection_at_fa(eval_dir)
    timed = (latency or {}).get(suffix)
    params = cap.get("n_params")
    train_s = synth.get("train_seconds")
    epochs_run, best_epoch = (convergence(pool, suffix) if suffix != "ensemble"
                              else (None, None))

    return {
        "model": label,
        "det_fa05": pct(matched[0.05]),
        "det_fa10": pct(matched[0.10]),
        "det_fa15": pct(matched[0.15]),
        "detect_rate": pct(at_synth.get("ewaste_detection_rate")),
        "fa_rate": pct(at_synth.get("organic_FP_rate")),
        "precision": at_synth.get("precision"),
        "recall": at_synth.get("recall"),
        "f1": at_synth.get("f1"),
        "oracle_detect": pct(oracle.get("ewaste_detection_rate")),
        "map50": synth.get("map50"),
        "map50_95": synth.get("map50_95"),
        "latency_ms": timed["latency_ms"] if timed else cap.get("latency_ms"),
        "fps": timed["fps"] if timed else cap.get("fps"),
        "params_m": None if params is None else params / 1e6,
        "gflops": cap.get("gflops"),
        "size_mb": cap.get("model_size_mb"),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "train_min": None if train_s is None else train_s / 60,
    }


def cell(row, key, fmt):
    v = row.get(key)
    if v is None:
        return "-"
    return v if isinstance(v, str) else fmt.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    args = ap.parse_args()

    latency = load_latency(args.pool)
    rows, missing = [], []
    for label, suffix in MODELS:
        row = collect(args.pool, label, suffix, latency)
        if row:
            rows.append(row)
        else:
            missing.append(label)

    if not rows:
        print(f"[!] no summaries found for pool {args.pool}.")
        print("    run src/06_evaluate.py for at least one model first.")
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

    print("Compare arms on the Det@FA columns. The synthetic-threshold columns "
          "sit at a")
    print("different point of each arm's curve and are not comparable across arms.")

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
