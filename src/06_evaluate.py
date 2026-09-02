"""
06_evaluate.py
------------------
STEP 4 of the v3 pipeline.  Runs after 05_train.py.

Evaluates a v3 detector on the two held-out sets of REAL photographs defined
in splits/. Neither set contributed an object, an occluder or a background to
the training data, so unlike 04_eval_real.py this measures generalisation
rather than recall of objects the detector was trained on.

  A) FALSE POSITIVE RATE on organic_test (747 real photographs, no e-waste).
     Every detection is by construction a false alarm. Specificity.

  B) DETECTION RATE on ewaste_test (400 real photographs, e-waste present,
     no object shared with the training pool). Sensitivity on unseen objects.

WHAT IT STILL CANNOT MEASURE
    The true-positive rate on real *contaminated* organic waste -- e-waste
    actually buried in real wet organics. No such imagery exists publicly.

Run:   python 06_evaluate.py --pool 200
Output: eval_pool<N>/
"""

from pathlib import Path
import argparse
import csv
import json

from PIL import Image, ImageDraw

import lib_modules  # noqa: F401  binds BiFPNFuse so custom checkpoints unpickle
from lib_metrics import (count_gflops, count_parameters, localisation_summary,
                         match_ious, measure_latency, weight_size_mb)
from pipeline_common import load_image

# ------------------------- CONFIG -------------------------
# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
SPLITS = ROOT / "splits"
# Hand-drawn boxes for ewaste_test, written by 09_annotate.py. Optional: when
# absent, everything except mIoU and Dice is still reported.
ANNOTATIONS = ROOT / "annotations" / "ewaste_test"

# Reported in the sweep table -- kept coarse so the printed report stays
# readable, and unchanged from earlier runs so old reports stay comparable.
THRESHOLDS = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Searched for the operating point. Inference runs once at min(THRESHOLDS) and
# every sweep point is just a filter over the stored confidences, so a fine
# grid costs no extra GPU time -- and the coarse grid demonstrably misses the
# optimum (0.345 rather than 0.40 on the yolo11s run, +0.006 F1).
FINE_STEP = 0.005
IMG_SIZE = 640
DEVICE = 0
BATCH = 16
SAVE_WORST = 12
LATENCY_WARMUP = 20
LATENCY_SAMPLE = 200
# ----------------------------------------------------------


def load_model(weights):
    """Build a detector from a checkpoint."""
    from ultralytics import YOLO
    return YOLO(str(weights))


def read_manifest(name: str):
    with open(SPLITS / f"{name}.csv", encoding="utf-8") as f:
        return [(r["category"], ROOT / r["path"]) for r in csv.DictReader(f)]


def load_annotations(rows):
    """
    Read hand-drawn boxes for the e-waste test photographs.

    Stored in YOLO format -- class, centre x, centre y, width, height, all
    normalised -- so they are converted to pixel xyxy against each image's own
    dimensions. Only the header is read to get those, not the pixel data.
    Returns {path: [[x1,y1,x2,y2], ...]} for the images that have a file.
    """
    if not ANNOTATIONS.is_dir():
        return {}
    out = {}
    for _, path in rows:
        label = ANNOTATIONS / f"{path.stem}.txt"
        if not label.exists():
            continue
        with Image.open(path) as im:
            w, h = im.size
        boxes = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts)
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
        out[path] = boxes
    return out


def localisation_at(ew_det, truth, conf):
    """
    mIoU and Dice over detections that survive `conf`, against hand-drawn boxes,
    plus the per-image counts needed to interpret them.

    The counts matter because a bare match rate is ambiguous. A model that fires
    once on a photograph holding three annotated objects can match at most one
    of them however well it is aimed, so a low ratio of matches to ground truths
    may be arithmetic rather than mislocalisation. The ceiling below,
    sum(min(detections, ground truths)) over images, is the most matches that
    were even possible; comparing the actual total against it separates the two.
    """
    if not truth:
        return None, []
    matched, n_gt, rows = [], 0, []
    for _, path, confs, boxes in ew_det:
        gt = truth.get(path)
        if gt is None:
            continue
        n_gt += len(gt)
        kept = [(c, b) for c, b in zip(confs, boxes) if c >= conf]
        ious = match_ious([b for _, b in kept], [c for c, _ in kept], gt)
        matched += ious
        rows.append({
            "path": path,
            "n_gt": len(gt),
            "n_detections": len(kept),
            "n_matched": len(ious),
            "ceiling": min(len(kept), len(gt)),
            "best_iou": round(max(ious), 4) if ious else 0.0,
        })
    summary = localisation_summary(matched, n_gt)

    fired = [r for r in rows if r["n_detections"]]
    hit = [r for r in rows if r["n_matched"]]
    ceiling = sum(r["ceiling"] for r in rows)
    summary.update({
        "n_images": len(rows),
        "n_images_fired": len(fired),
        "n_images_with_a_match": len(hit),
        "match_ceiling": ceiling,
        # of the matches that were geometrically possible, how many landed
        "ceiling_utilisation": round(summary["n_matched"] / ceiling, 4) if ceiling else None,
        # when the model fires on an annotated photograph, how often does any
        # of its boxes actually land on an annotated object
        "hit_rate_given_fired": round(len(hit) / len(fired), 4) if fired else None,
    })
    return summary, rows


def run_inference(model, rows, label):
    """Detect once at the lowest threshold; filter afterwards for each sweep point."""
    print(f"  {label}: {len(rows)} real images")
    out = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        images = [load_image(p) for _, p in chunk]
        results = model.predict(images,
                                conf=min(THRESHOLDS), imgsz=IMG_SIZE,
                                device=DEVICE, verbose=False)
        for (category, p), r in zip(chunk, results):
            confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
            boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
            out.append((category, p, confs, boxes))
        if (i + BATCH) % 160 == 0:
            print(f"    {min(i + BATCH, len(rows))}/{len(rows)}")
    return out


def fired(detections, t):
    """(images with at least one detection, total detections) at threshold t."""
    imgs = boxes = 0
    for _, _, confs, _ in detections:
        kept = [c for c in confs if c >= t]
        if kept:
            imgs += 1
        boxes += len(kept)
    return imgs, boxes


def wilson(k, n, z=1.96):
    """
    Wilson score interval for a proportion k/n, as (low, high).

    Preferred over the normal approximation because the rates here sit near
    the ends of the range -- a false-positive rate around 5% with n=747 has
    an asymmetric interval that the normal approximation gets wrong, and can
    even push below zero.

    Used to keep the model comparison honest: differences smaller than these
    intervals are not resolvable from a single run.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def score_at(org_det, ew_det, t, n_org, n_ew):
    """Image-level metrics at threshold t. Shared by the coarse and fine sweeps."""
    o_imgs, o_boxes = fired(org_det, t)
    e_imgs, _ = fired(ew_det, t)
    precision = e_imgs / (e_imgs + o_imgs) if (e_imgs + o_imgs) else 0.0
    recall = e_imgs / n_ew if n_ew else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    d_lo, d_hi = wilson(e_imgs, n_ew)
    f_lo, f_hi = wilson(o_imgs, n_org)
    return {
        "confidence": round(t, 4),
        "organic_imgs_with_FP": o_imgs,
        "organic_total_FP": o_boxes,
        "organic_FP_rate": round(o_imgs / n_org, 4) if n_org else 0.0,
        "organic_FP_per_image": round(o_boxes / n_org, 3) if n_org else 0.0,
        "ewaste_imgs_detected": e_imgs,
        "ewaste_detection_rate": round(recall, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "detect_ci_low": round(d_lo, 4),
        "detect_ci_high": round(d_hi, 4),
        "fp_ci_low": round(f_lo, 4),
        "fp_ci_high": round(f_hi, 4),
    }


def save_worst(detections, t, out_dir, k=SAVE_WORST):
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for _, p, confs, boxes in detections:
        kept = [(c, b) for c, b in zip(confs, boxes) if c >= t]
        if kept:
            scored.append((max(c for c, _ in kept), len(kept), p, kept))
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    for rank, (maxc, cnt, p, kept) in enumerate(scored[:k], 1):
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        for c, b in kept:
            d.rectangle(b, outline=(255, 0, 0), width=4)
            d.text((b[0] + 4, max(0, b[1] - 14)), f"{c:.2f}", fill=(255, 0, 0))
        im.save(out_dir / f"FP_{rank:02d}_conf{maxc:.2f}_n{cnt}_{p.stem}.jpg", quality=90)
    return len(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--tag", type=str, default="",
                    help="run-directory suffix used by the architecture comparison")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed of the run to evaluate; 0 is the "
                         "original run and keeps the original directory names")
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ("" if args.seed == 0 else f"_s{args.seed}")
    run_dir = ROOT / "runs" / "detect" / f"pool{args.pool}{suffix}"
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"[!] weights not found: {weights}")
        return

    out = ROOT / f"eval_pool{args.pool}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    synth = {}
    synth_path = run_dir / "synthetic_summary.json"
    if synth_path.exists():
        synth = json.loads(synth_path.read_text(encoding="utf-8"))

    print(f"Loading {weights}")
    model = load_model(weights)

    organic = read_manifest("organic_test")
    ewaste = read_manifest("ewaste_test")

    print("\n=== A) SPECIFICITY: held-out real organic waste (expect nothing) ===")
    org_det = run_inference(model, organic, "organic_test")
    print("\n=== B) SENSITIVITY: held-out real e-waste, unseen objects ===")
    ew_det = run_inference(model, ewaste, "ewaste_test")

    n_org, n_ew = len(org_det), len(ew_det)

    # Cost of the detector itself. Timed on already-decoded photographs at
    # batch 1 so the number reflects the network and its NMS rather than the
    # disk, and on real images rather than synthetic ones so it matches the
    # setting every other figure in the table is measured in.
    print('\n=== C) COST: single-image latency on real photographs ===')
    warm = [load_image(path) for _, path in ewaste[:LATENCY_WARMUP + LATENCY_SAMPLE]]
    latency_ms, fps = measure_latency(
        lambda ims: model.predict(ims, conf=min(THRESHOLDS), imgsz=IMG_SIZE,
                                  device=DEVICE, verbose=False),
        warm, warmup=LATENCY_WARMUP, sample=LATENCY_SAMPLE)
    del warm
    print(f"  median {latency_ms} ms/image  ->  {fps} FPS")

    net = getattr(model, "model", None)
    capacity = {
        "n_params": count_parameters(net) if net is not None else None,
        "gflops": count_gflops(net, IMG_SIZE) if net is not None else None,
        "model_size_mb": weight_size_mb(weights),
        "latency_ms": latency_ms,
        "fps": fps,
    }

    truth = load_annotations(ewaste)

    rows = [score_at(org_det, ew_det, t, n_org, n_ew) for t in THRESHOLDS]

    # Fine search for the actual operating point. Free: no extra inference.
    lo, hi = min(THRESHOLDS), max(THRESHOLDS)
    n_steps = int(round((hi - lo) / FINE_STEP)) + 1
    fine_rows = [score_at(org_det, ew_det, lo + i * FINE_STEP, n_org, n_ew)
                 for i in range(n_steps)]

    with open(out / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # The fine sweep is what the operating point is chosen from, so persist it
    # too -- the choice stays auditable without re-running inference.
    with open(out / "threshold_sweep_fine.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fine_rows[0].keys()))
        w.writeheader()
        w.writerows(fine_rows)

    if loc_rows:
        with open(out / "localisation_per_image.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "n_gt", "n_detections",
                                              "n_matched", "ceiling", "best_iou"])
            w.writeheader()
            for r in loc_rows:
                w.writerow(dict(r, path=r["path"].relative_to(ROOT).as_posix()))

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role", "category", "path", "n_boxes", "max_conf"])
        for role, dets in (("organic_test", org_det), ("ewaste_test", ew_det)):
            for category, p, confs, _ in dets:
                w.writerow([role, category, p.relative_to(ROOT).as_posix(),
                            len(confs), round(max(confs), 4) if confs else ""])

    best = max(fine_rows, key=lambda r: r["f1"])
    best_coarse = max(rows, key=lambda r: r["f1"])

    # The reported operating point is the one carried over from synthetic
    # validation, not the one that maximises F1 on the test set. Choosing a
    # threshold on the same data the threshold is then scored on is optimistic
    # by construction: measured across these architectures the gap is small for
    # most, but reached +0.070 F1 for one, whose detection rate went from 56.2%
    # honestly reported to 77.5% tuned. The tuned figure is kept and printed,
    # clearly labelled, because the distance between the two says something
    # about how threshold-sensitive a model is.
    headline = at_synth if at_synth is not None else best
    headline_source = ("synthetic validation" if at_synth is not None
                       else "TEST SET -- no synthetic threshold available")
    localisation, loc_rows = localisation_at(ew_det, truth, headline["confidence"])
    synth_conf = synth.get("best_f1_conf")
    at_synth = None
    if synth_conf is not None:
        # Fine grid: the threshold carried over from synthetic validation is
        # matched to within FINE_STEP rather than snapped to a coarse point,
        # so "what you would actually get in deployment" is not distorted by
        # the reporting grid.
        at_synth = min(fine_rows, key=lambda r: abs(r["confidence"] - synth_conf))

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 74)
    emit(f"REAL-IMAGE EVALUATION  (object pool = {args.pool})")
    emit("=" * 74)
    emit(f"  held-out organic images (no e-waste present): {n_org}")
    emit(f"  held-out e-waste images (e-waste present)   : {n_ew}")
    emit("  no object, occluder or background in the training data came from")
    emit("  either set.")
    emit()
    emit(f"{'conf':>7} | {'FP rate':>9} {'FP/img':>8} | {'detect rate':>12} | {'F1':>6}")
    emit(f"{'':->7}-+-{'':->19}-+-{'':->14}-+-{'':->7}")
    for r in rows:
        star = "  <--" if r is best_coarse else ""
        emit(f"{r['confidence']:>7.2f} | {r['organic_FP_rate']:>9.3f} "
             f"{r['organic_FP_per_image']:>8.2f} | {r['ewaste_detection_rate']:>12.3f} "
             f"| {r['f1']:>6.3f}{star}")
    emit()
    emit("  FP rate     = fraction of held-out organic images with >=1 false alarm")
    emit("  detect rate = fraction of held-out e-waste photos with >=1 detection")
    emit("  F1          = image-level, treating each organic image as a negative")
    emit("                case and each e-waste image as a positive case")
    emit()
    emit("-" * 74)
    emit("OPERATING POINT  <-- report these numbers")
    emit("-" * 74)
    emit(f"  confidence {headline['confidence']:.3f}, chosen on {headline_source}")
    emit(f"    detection {headline['ewaste_detection_rate']:.1%} "
         f"({headline['ewaste_imgs_detected']} of {n_ew})"
         f"   95% CI [{headline['detect_ci_low']:.1%}, {headline['detect_ci_high']:.1%}]")
    emit(f"    false alarms {headline['organic_FP_rate']:.1%} "
         f"({headline['organic_imgs_with_FP']} of {n_org})"
         f"   95% CI [{headline['fp_ci_low']:.1%}, {headline['fp_ci_high']:.1%}]")
    emit(f"    F1 {headline['f1']:.3f}")
    emit()
    emit("-" * 74)
    emit("ORACLE UPPER BOUND  (not a result)")
    emit("-" * 74)
    emit(f"  confidence {best['confidence']:.3f}, chosen by maximising F1 on the")
    emit("  test set itself, so it is not achievable without already knowing the")
    emit("  answer. Reported only as a ceiling and as a measure of how sensitive")
    emit("  this model is to its threshold.")
    emit(f"    detection {best['ewaste_detection_rate']:.1%}, "
         f"false alarms {best['organic_FP_rate']:.1%}, F1 {best['f1']:.3f}")
    if at_synth is not None:
        gap = best["f1"] - at_synth["f1"]
        emit(f"    optimism over the reported point: {gap:+.3f} F1")
        if gap > 0.03:
            emit("    NOTE: a gap this large means most of the tuned figure is")
            emit("    threshold selection rather than detector quality.")
    emit()
    emit("  Intervals are Wilson score, 95%. A difference between two models")
    emit("  smaller than these intervals is not resolvable from one run each.")
    emit()
    emit("  Report DETECTION RATE and FALSE ALARM RATE as the primary numbers.")
    emit("  Precision and F1 additionally depend on the 400:747 positive-to-")
    emit("  negative ratio of this test set, which is an artefact of how the")
    emit("  split was drawn rather than a real-world prevalence.")
    emit()
    emit("  LIMITATION: this measures specificity on real organic waste and")
    emit("  detection on real isolated e-waste. It does NOT measure the true-")
    emit("  positive rate on real CONTAMINATED waste, because no such imagery")
    emit("  exists.")
    emit()
    if not truth:
        emit("  LIMITATION: detection rate counts an e-waste photograph as detected")
        emit("  when the model fires ANYWHERE in the frame. ewaste_test carries no")
        emit("  ground-truth boxes, so a detection landing on background still")
        emit("  counts. Treat the figure as an upper bound, and describe it as the")
        emit("  model firing on the image rather than localising the object.")
    else:
        emit("  Detection rate counts a photograph as detected when the model fires")
        emit("  anywhere in the frame. The mIoU below is measured against the")
        emit("  hand-drawn boxes and is what shows whether those firings actually")
        emit("  landed on the object.")
    emit()
    emit("-" * 74)
    emit("COST")
    emit("-" * 74)
    emit(f"  parameters {capacity['n_params']:,}" if capacity["n_params"]
         else "  parameters unavailable")
    emit(f"  GFLOPs     {capacity['gflops']}" if capacity["gflops"]
         else "  GFLOPs     unavailable (thop not installed in this environment)")
    emit(f"  weights    {capacity['model_size_mb']} MB")
    emit(f"  latency    {latency_ms} ms per image at batch 1, {fps} FPS")
    emit()
    emit("-" * 74)
    emit("LOCALISATION")
    emit("-" * 74)
    if localisation is None:
        emit("  no hand-drawn boxes found under annotations/ewaste_test.")
        emit("  Run 09_annotate.py to enable mIoU and Dice.")
    elif localisation["n_matched"] == 0:
        emit(f"  {localisation['n_gt']} boxes annotated, none matched at "
             f"conf {best['confidence']:.3f}")
    else:
        emit(f"  annotated boxes {localisation['n_gt']}, matched "
             f"{localisation['n_matched']} at IoU >= {localisation['iou_thr']}")
        emit(f"  mIoU {localisation['mIoU']:.4f}    Dice {localisation['dice']:.4f}")
        emit("  Averaged over matched detections only; an object the model never")
        emit("  found has no IoU and is counted as a recall failure instead.")
        emit()
        emit(f"  annotated photographs {localisation['n_images']}, "
             f"model fired on {localisation['n_images_fired']}, "
             f"landed on an object in {localisation['n_images_with_a_match']}")
        if localisation["hit_rate_given_fired"] is not None:
            emit(f"  when it fires, it hits an annotated object "
                 f"{localisation['hit_rate_given_fired']:.1%} of the time")
        if localisation["ceiling_utilisation"] is not None:
            emit(f"  matches {localisation['n_matched']} of the "
                 f"{localisation['match_ceiling']} that were geometrically "
                 f"possible ({localisation['ceiling_utilisation']:.1%})")
            emit("  A low ratio of matches to annotated boxes is partly arithmetic:")
            emit("  one detection cannot match several objects in one frame. The")
            emit("  ceiling figure is the comparison that is not confounded by that.")
        emit("  For an axis-aligned box Dice is exactly 2*IoU/(1+IoU), so it")
        emit("  ranks models identically to mIoU and adds no new evidence.")
    emit()
    emit("=" * 74)

    fp_conf = headline["confidence"]
    n_fp = save_worst(org_det, fp_conf, out / "false_positives")
    emit(f"\nSaved up to {SAVE_WORST} worst false positives at conf {fp_conf:.2f}")
    emit(f"({n_fp} held-out organic images fired) -> {out.name}/false_positives")

    summary = {
        "pool": args.pool,
        "model": synth.get("model"),
        "n_organic_test": n_org,
        "n_ewaste_test": n_ew,
        "synthetic": synth,
        "headline": headline,                 # what the paper should quote
        "headline_source": headline_source,
        "real_best": best,                    # oracle: tuned on the test set
        "real_best_coarse_grid": best_coarse,  # what earlier runs reported
        "real_at_synthetic_conf": at_synth,
        "capacity": capacity,
        "localisation": localisation,
        "fine_step": FINE_STEP,
        "sweep": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {out.name}/report.txt, threshold_sweep.csv, "
          f"threshold_sweep_fine.csv, per_image.csv, summary.json")


if __name__ == "__main__":
    main()
