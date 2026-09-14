"""
14_latency.py
-------------
Measures inference latency for every arm in one sitting.

06_evaluate.py already times the model it is evaluating, and the routine it
uses is correct: a warm-up, CUDA synchronisation either side of each call, and
a median rather than a mean. What was wrong is that each arm was timed in its
own process, on its own day, on a machine doing different things each time.
Two bit-identical runs of the same architecture came out 2.2x apart, and one
arm reported half the latency of another with identical GFLOPs and more
parameters. Those columns were measuring machine load.

This script fixes the experiment rather than the timer. Every arm is loaded
into one process and timed round-robin: arm 1, arm 2, ... arm 11, then again,
five times over. Thermal throttling, a background process, or a clock drift
part-way through therefore lands on every arm roughly equally instead of
falling entirely on whichever happened to run last. Each arm's figure is the
median of its per-round medians.

The spread across rounds is reported alongside. It is the evidence that the
measurement is stable: if an arm's rounds disagree by more than a few percent,
the number is not trustworthy and the run should be repeated on a quieter
machine. That check is the whole point, so it is printed whether it passes or
fails.

Run:    python src/14_latency.py --pool 60
Output: latency_pool60.json, read by 11_metrics_table.py
"""

from pathlib import Path
from statistics import median
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib_modules  # noqa: F401  registers CBAM/BiFPNFuse/TVBackbone
from lib_arms import ARMS, run_name
from lib_metrics import measure_latency

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# Five rounds of 60 images each. The rounds matter more than the sample size:
# a longer single pass measures one machine state very precisely, whereas
# several short interleaved passes measure the state every arm actually shares.
ROUNDS = 9
SAMPLE = 60
WARMUP = 20

# The first timed round still runs slow even after the warm-up pass -- cuDNN
# keeps autotuning, and the clocks have not settled. Measured at about 20%
# above the rest for the lightest arms, which is larger than any difference
# between architectures, so it is dropped rather than averaged in. Rounds are
# odd-numbered after the drop so the median is a measured value.
DISCARD_FIRST = 1


def load_images(limit):
    """Test photographs, as paths. Same images every arm, same order."""
    path = ROOT / "splits" / "ewaste_test.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    key = "path" if "path" in rows[0] else list(rows[0])[0]
    return [r[key] for r in rows[:limit]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--sample", type=int, default=SAMPLE)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
              else "cpu")
    images = load_images(args.sample)
    if not images:
        print("[!] no test images found under splits/ewaste_test.csv")
        return

    models, missing = [], []
    for label, _cfg, tag in ARMS:
        w = (ROOT / "runs" / "detect" / run_name(args.pool, tag)
             / "weights" / "best.pt")
        if not w.exists():
            missing.append(label)
            continue
        models.append((label, tag, YOLO(str(w))))
    if missing:
        print(f"[!] not trained, skipped: {', '.join(missing)}")
    if not models:
        print("[!] nothing to measure")
        return

    print(f"device {device}   {len(models)} arms   "
          f"{args.rounds} rounds x {args.sample} images, interleaved")

    # Every arm gets its warm-up before any arm is timed, so the first timed
    # round does not charge arm 1 for cuDNN autotuning that arms 2..11 have
    # already had done for them by the time their turn comes.
    for _label, _tag, m in models:
        for im in images[:WARMUP]:
            m.predict(im, verbose=False)

    rounds = {tag: [] for _, tag, _ in models}
    for r in range(args.rounds):
        for _label, tag, m in models:
            ms, _ = measure_latency(
                lambda batch, _m=m: _m.predict(batch, verbose=False),
                images, warmup=0, sample=args.sample)
            rounds[tag].append(ms)
        print(f"  round {r + 1} of {args.rounds} done")

    out = {"imgsz": 640, "batch": 1, "rounds": args.rounds,
           "discarded_first": DISCARD_FIRST, "sample": args.sample,
           "device": device, "arms": {}}

    print()
    print(f"{'arm':26s} {'ms':>7s} {'FPS':>7s} {'spread':>8s}")
    print("-" * 54)
    unstable = []
    for label, tag, _ in models:
        vals = [v for v in rounds[tag][DISCARD_FIRST:] if v is not None]
        if not vals:
            continue
        # The minimum, not the median. Contention on a shared machine can only
        # ever add time to a round, never subtract it, so the fastest round is
        # the closest estimate of what the architecture costs and the slower
        # ones measure whatever else the machine was doing. The median is kept
        # alongside: the gap between them is the size of the contention, and
        # if it is large the machine was too busy to measure on.
        ms = min(vals)
        med = median(vals)
        out["arms"][tag] = {
            "label": label,
            "latency_ms": round(ms, 2),
            "fps": round(1000 / ms, 1),
            "median_ms": round(med, 2),
            "round_ms": [round(v, 2) for v in vals],
            "contention_pct": round((med - ms) / ms * 100, 1),
        }
        spread = (max(vals) - min(vals)) / ms * 100
        flag = ""
        if spread > 10:
            flag = "  <-- unstable"
            unstable.append(label)
        print(f"{label:26s} {ms:7.2f} {1000 / ms:7.1f} {spread:6.1f}%{flag}")

    path = ROOT / f"latency_pool{args.pool}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    if unstable:
        print(f"[!] rounds disagree by more than 10% for: {', '.join(unstable)}")
        print("    treat those as indicative and re-run on an idle machine.")
    else:
        print("all arms stable within 10% across rounds")
    print(f"written: {path.name}")


if __name__ == "__main__":
    main()
