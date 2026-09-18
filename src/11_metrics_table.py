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

Precision and F1 depend on the ratio of positives to negatives in the test
split, an artefact of how it was drawn rather than a real prevalence.

Run:    python src/11_metrics_table.py --pool 54
Output: printed table, plus Manuscripts/tables/metrics_table.{csv,tex}
"""

from bisect import bisect_left
from pathlib import Path
import argparse
import csv
import json
import sys
import tempfile

from lib_arms import DEFAULT_POOL, MEMBERS, eval_dir_name, run_name

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
OUT = ROOT / "Manuscripts" / "tables"

# Order is the order the paper presents them in: every arm from the one shared
# list, then the ensemble, which is not a trainable arm.
MODELS = MEMBERS + [("Ensemble", "ensemble")]

# False-alarm budgets the arms are compared at, and their table columns.
MATCHED_FA = (0.05, 0.10, 0.15)
FA_KEY = {0.05: "det_fa05", 0.10: "det_fa10", 0.15: "det_fa15"}

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
    Detection rate at each target false-alarm rate, and which targets were
    censored by the inference floor.

    An image fires at threshold t when its highest detection confidence is at
    least t, so every distinct per-image confidence is a threshold at which a
    rate can change, and nothing between two of them can. Sweeping exactly
    those gives the exact answer; the 0.005-step grid used before could step
    over the threshold where the false-alarm rate crossed its budget, and read
    detection up to a point low.

    The false-alarm rate used is the test set's own, so this compares arms at
    equal cost; it is not a threshold a deployer could have set beforehand.

    A target is censored when even the lowest threshold available -- every
    image with any stored detection fires -- stays within the budget: a lower
    inference floor might then have reached more detections, so the value is
    a lower bound.
    """
    path = eval_dir / "per_image.csv"
    if not path.exists():
        return {t: None for t in targets}, []
    org, ew = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = float(r["max_conf"]) if r["max_conf"] else None
            (org if r["role"] == "organic_test" else ew).append(c)
    fired_org = sorted(c for c in org if c is not None)
    fired_ew = sorted(c for c in ew if c is not None)

    def rate(confs, n, t):
        return (len(confs) - bisect_left(confs, t)) / n

    thresholds = sorted(set(fired_org) | set(fired_ew))
    out, censored = {}, []
    for target in targets:
        # a threshold above every confidence always meets the budget, at 0
        ok = [rate(fired_ew, len(ew), t) for t in thresholds
              if rate(fired_org, len(org), t) <= target]
        out[target] = max(ok, default=0.0)
        if thresholds and rate(fired_org, len(org), thresholds[0]) <= target:
            censored.append(target)
    return out, censored


def _check():
    """detection_at_fa on a case small enough to work by hand."""
    # organic max confidences 0.9, 0.5, 0.3 and seven that never fire (n=10);
    # e-waste 0.95, 0.8, 0.6, 0.4, 0.2 and one that never fires (n=6).
    # Thresholds 0.2 0.3 0.4 0.5 0.6 0.8 0.9 0.95 give false alarms
    # .3 .3 .2 .2 .1 .1 .1 0 and detections 5 4 4 3 3 2 1 1 (of 6).
    rows = ([("organic_test", c) for c in (0.9, 0.5, 0.3)] + [("organic_test", "")] * 7
            + [("ewaste_test", c) for c in (0.95, 0.8, 0.6, 0.4, 0.2)] + [("ewaste_test", "")])
    with tempfile.TemporaryDirectory() as d:
        with open(Path(d) / "per_image.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["role", "category", "path", "n_boxes", "max_conf"])
            w.writerows([role, "x", "p", 1, c] for role, c in rows)
        got, cens = detection_at_fa(Path(d), (0.0, 0.10, 0.20, 0.30))
    assert got == {0.0: 1 / 6, 0.10: 3 / 6, 0.20: 4 / 6, 0.30: 5 / 6}, got
    # only the 30% budget is met at the lowest threshold, so only it is censored
    assert cens == [0.30], cens
    print("detection_at_fa: ok")


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

    matched, censored = detection_at_fa(eval_dir)
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
        # the columns whose value is a lower bound, rendered with a >= sign
        "censored": [FA_KEY[t] for t in censored],
    }


def cell(row, key, fmt):
    v = row.get(key)
    if v is None:
        return "-"
    text = v if isinstance(v, str) else fmt.format(v)
    return ">=" + text if key in row.get("censored", ()) else text


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

    if any(r["censored"] for r in rows):
        print("'>=' marks a lower bound: that arm stays within the false-alarm "
              "budget even at the inference floor, so a lower floor might find more.")
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
    if "--check" in sys.argv:
        _check()
    else:
        main()
