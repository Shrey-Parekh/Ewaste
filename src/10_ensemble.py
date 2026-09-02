"""
10_ensemble.py
--------------
Fuses the seven single-model detectors and scores the result on the same two
sets of real photographs, by the same rules, as 06_evaluate.py.

Combination is weighted box fusion (Solovyev et al. 2021). Unlike NMS, which
keeps one box and discards the rest, WBF averages the coordinates of every box
in a cluster weighted by confidence, so agreement between models sharpens the
box rather than merely selecting one model's version of it.

The statistics are not reimplemented here. 06_evaluate.py is imported by path
-- its module name starts with a digit, so it cannot be imported by name, the
same reason 04_build_dataset.py loads lib_composite.py that way -- and its
threshold sweep, Wilson intervals and operating-point search are reused
unchanged. Two copies of that logic that drifted apart would silently make the
ensemble incomparable with its own members.

One caveat carried into the report: WBF rescales a cluster's confidence by the
fraction of models that contributed to it, so ensemble confidences do not live
on the same scale as a single model's. The operating point is therefore not
comparable across the two; the detection and false-alarm rates it produces are.

Run:   python 10_ensemble.py --pool 60
Output: eval_pool<N>_ensemble/
"""

from pathlib import Path
import argparse
import csv
import importlib.util
import json

import lib_modules  # noqa: F401  binds BiFPNFuse so custom checkpoints unpickle
from lib_metrics import best_box_f1, measure_latency
from PIL import Image

from pipeline_common import load_image

# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# (label, run-directory suffix). The empty suffix is the plain YOLOv8s run.
MEMBERS = [
    ("YOLOv8s", ""),
    ("YOLOv11s", "yolo11s"),
    ("YOLOv8s+CBAM", "v8s_cbam"),
    ("YOLOv11s+CBAM", "v11s_cbam"),
    ("ResNet34+FPN+CBAM", "r34_fpn_cbam"),
    ("ResNet34+BiFPN+CBAM", "r34_bifpn_cbam"),
    ("YOLOv11s+BiFPN+CBAM", "v11s_bifpn_cbam"),
]

IOU_THR = 0.55          # cluster membership, the WBF paper's default
LATENCY_WARMUP = 10
LATENCY_SAMPLE = 100


def synthetic_val(pool):
    """Synthetic validation images and their ground-truth boxes, in pixel xyxy."""
    d = ROOT / f"dataset_pool{pool}"
    out = []
    for img in sorted((d / "images" / "val").glob("*.jpg")):
        lab = d / "labels" / "val" / f"{img.stem}.txt"
        with Image.open(img) as im:
            w, h = im.size
        boxes = []
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = (float(v) for v in parts)
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
        out.append((img, boxes))
    return out


def ensemble_threshold(models, pool, ev, n_models):
    """
    Pick the ensemble's operating threshold on the synthetic validation split.

    Every single model gets its threshold from a box-level F1 curve over this
    same split, computed by Ultralytics. A fused detector has no such curve, so
    without this its threshold would have to come from the test set -- exactly
    the optimism the single models avoid. Choosing it here keeps the ensemble
    honest and comparable with its own members.
    """
    rows = synthetic_val(pool)
    if not rows:
        return None, None
    print(f"  choosing threshold on {len(rows)} synthetic validation images")
    per_image = []
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        images = [load_image(pth) for pth, _ in chunk]
        per_model = []
        for m in models:
            res = m.predict(images, conf=min(ev.THRESHOLDS), imgsz=ev.IMG_SIZE,
                            device=ev.DEVICE, verbose=False)
            per_model.append([
                (r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else [],
                 r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else [])
                for r in res])
        for j, (_, gt) in enumerate(chunk):
            confs, boxes = fuse([pm[j] for pm in per_model], n_models)
            per_image.append((confs, boxes, gt))
    lo, hi = min(ev.THRESHOLDS), max(ev.THRESHOLDS)
    steps = int(round((hi - lo) / ev.FINE_STEP)) + 1
    grid = [lo + k * ev.FINE_STEP for k in range(steps)]
    return best_box_f1(per_image, grid)


def load_evaluator():
    """Import 06_evaluate.py by path; its name is not a valid identifier."""
    spec = importlib.util.spec_from_file_location(
        "evaluate_single", SRC / "06_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area - inter)


def fuse(per_model, n_models, iou_thr=IOU_THR):
    """
    Weighted box fusion of one image's detections.

    ``per_model`` is a list, one entry per model, of (confidences, boxes).
    Returns fused (confidences, boxes).
    """
    entries = []
    for idx, (confs, boxes) in enumerate(per_model):
        entries.extend((c, b, idx) for c, b in zip(confs, boxes))
    entries.sort(key=lambda e: -e[0])

    clusters = []
    for conf, box, idx in entries:
        best, best_iou = None, iou_thr
        for c in clusters:
            v = iou(c["box"], box)
            if v >= best_iou:
                best, best_iou = c, v
        if best is None:
            clusters.append({"box": list(box), "members": [(conf, box, idx)]})
            continue
        best["members"].append((conf, box, idx))
        total = sum(m[0] for m in best["members"])
        best["box"] = [sum(m[0] * m[1][k] for m in best["members"]) / total
                       for k in range(4)]

    out_confs, out_boxes = [], []
    for c in clusters:
        confs = [m[0] for m in c["members"]]
        contributors = len({m[2] for m in c["members"]})
        # a box only one model found is downweighted in proportion
        out_confs.append(sum(confs) / len(confs) * contributors / n_models)
        out_boxes.append(c["box"])
    return out_confs, out_boxes


def fuse_detections(per_model_detections, n_models):
    """Fuse aligned per-model detection lists into one 06-shaped list."""
    fused = []
    reference = per_model_detections[0]
    for i, (category, path, _, _) in enumerate(reference):
        confs, boxes = fuse([(d[i][2], d[i][3]) for d in per_model_detections],
                            n_models)
        fused.append((category, path, confs, boxes))
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=60)
    args = ap.parse_args()

    ev = load_evaluator()

    weights = []
    for label, tag in MEMBERS:
        suffix = f"_{tag}" if tag else ""
        w = ROOT / "runs" / "detect" / f"pool{args.pool}{suffix}" / "weights" / "best.pt"
        if not w.exists():
            print(f"[!] missing member: {label} -> {w}")
            print("    train every member before building the ensemble.")
            return
        weights.append((label, w))

    out = ROOT / f"eval_pool{args.pool}_ensemble"
    out.mkdir(parents=True, exist_ok=True)

    organic = ev.read_manifest("organic_test")
    ewaste = ev.read_manifest("ewaste_test")

    org_runs, ew_runs, models = [], [], []
    for label, w in weights:
        print(f"\n=== {label} ===")
        model = ev.load_model(w, "ultralytics")
        models.append(model)
        org_runs.append(ev.run_inference(model, organic, "organic_test"))
        ew_runs.append(ev.run_inference(model, ewaste, "ewaste_test"))

    n = len(weights)
    print(f"\nFusing {n} models with weighted box fusion, IoU >= {IOU_THR}")
    org_det = fuse_detections(org_runs, n)
    ew_det = fuse_detections(ew_runs, n)

    n_org, n_ew = len(org_det), len(ew_det)

    print("\n=== COST: single-image latency of the whole ensemble ===")
    warm = [load_image(p) for _, p in ewaste[:LATENCY_WARMUP + LATENCY_SAMPLE]]

    def predict_all(images):
        per = []
        for m in models:
            r = m.predict(images, conf=min(ev.THRESHOLDS), imgsz=ev.IMG_SIZE,
                          device=ev.DEVICE, verbose=False)[0]
            confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
            boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
            per.append((confs, boxes))
        fuse(per, n)

    latency_ms, fps = measure_latency(predict_all, warm,
                                      warmup=LATENCY_WARMUP, sample=LATENCY_SAMPLE)
    del warm
    print(f"  median {latency_ms} ms/image  ->  {fps} FPS")

    capacity = {
        "n_params": sum(sum(p.numel() for p in m.model.parameters()) for m in models),
        "gflops": None,
        "model_size_mb": round(sum(w.stat().st_size for _, w in weights) / 1e6, 2),
        "latency_ms": latency_ms,
        "fps": fps,
    }

    truth = ev.load_annotations(ewaste)

    rows = [ev.score_at(org_det, ew_det, t, n_org, n_ew) for t in ev.THRESHOLDS]
    lo, hi = min(ev.THRESHOLDS), max(ev.THRESHOLDS)
    steps = int(round((hi - lo) / ev.FINE_STEP)) + 1
    fine_rows = [ev.score_at(org_det, ew_det, lo + i * ev.FINE_STEP, n_org, n_ew)
                 for i in range(steps)]

    best = max(fine_rows, key=lambda r: r["f1"])
    best_coarse = max(rows, key=lambda r: r["f1"])

    print()
    print("=== choosing the operating threshold on held-out synthetic data ===")
    synth_conf, synth_f1 = ensemble_threshold(models, args.pool, ev, n)
    if synth_conf is not None:
        headline = min(fine_rows, key=lambda r: abs(r["confidence"] - synth_conf))
        headline_source = "synthetic validation"
        print(f"  threshold {synth_conf:.3f} (box F1 {synth_f1:.3f} on synthetic val)")
    else:
        headline, headline_source = best, "TEST SET -- no synthetic split found"
    localisation, loc_rows = ev.localisation_at(ew_det, truth, headline["confidence"])

    for name, data in (("threshold_sweep.csv", rows),
                       ("threshold_sweep_fine.csv", fine_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    if loc_rows:
        with open(out / "localisation_per_image.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "n_gt", "n_detections",
                                              "n_matched", "ceiling", "best_iou"])
            w.writeheader()
            for r in loc_rows:
                w.writerow(dict(r, path=r["path"].relative_to(ROOT).as_posix()))

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["role", "category", "path", "n_boxes", "max_conf"])
        for role, dets in (("organic_test", org_det), ("ewaste_test", ew_det)):
            for category, p, confs, _ in dets:
                writer.writerow([role, category, p.relative_to(ROOT).as_posix(),
                                 len(confs), round(max(confs), 4) if confs else ""])

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 74)
    emit(f"ENSEMBLE OF {n} MODELS  (object pool = {args.pool})")
    emit("=" * 74)
    for label, _ in weights:
        emit(f"  member: {label}")
    emit(f"  fusion: weighted box fusion, cluster IoU >= {IOU_THR}")
    emit()
    emit(f"  held-out organic images (no e-waste present): {n_org}")
    emit(f"  held-out e-waste images (e-waste present)   : {n_ew}")
    emit()
    emit(f"{'conf':>7} | {'FP rate':>9} {'FP/img':>8} | {'detect rate':>12} | {'F1':>6}")
    emit(f"{'':->7}-+-{'':->19}-+-{'':->14}-+-{'':->7}")
    for r in rows:
        star = "  <--" if r is best_coarse else ""
        emit(f"{r['confidence']:>7.2f} | {r['organic_FP_rate']:>9.3f} "
             f"{r['organic_FP_per_image']:>8.2f} | {r['ewaste_detection_rate']:>12.3f} "
             f"| {r['f1']:>6.3f}{star}")
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
    emit(f"  confidence {best['confidence']:.3f}, maximising F1 on the test set")
    emit(f"    detection {best['ewaste_detection_rate']:.1%}, "
         f"false alarms {best['organic_FP_rate']:.1%}, F1 {best['f1']:.3f}")
    emit(f"    optimism over the reported point: {best['f1'] - headline['f1']:+.3f} F1")
    emit()
    emit("  WBF rescales a cluster's confidence by the fraction of models that")
    emit("  found it, so this confidence is not on the same scale as a single")
    emit("  model's. Compare the rates, not the threshold.")
    emit()
    emit("-" * 74)
    emit("COST")
    emit("-" * 74)
    emit(f"  parameters {capacity['n_params']:,} summed over {n} members")
    emit(f"  weights    {capacity['model_size_mb']} MB summed")
    emit(f"  latency    {latency_ms} ms per image at batch 1, {fps} FPS")
    emit(f"  Running {n} detectors costs approximately {n} times the inference")
    emit("  of one. That is the price of the accuracy reported above, and is")
    emit("  stated here so the comparison with the single models is honest.")
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
    emit("=" * 74)

    n_fp = ev.save_worst(org_det, headline["confidence"], out / "false_positives")
    emit(f"\n({n_fp} held-out organic images fired) -> {out.name}/false_positives")

    summary = {
        "pool": args.pool,
        "model": "ensemble",
        "members": [label for label, _ in weights],
        "fusion": "weighted_box_fusion",
        "fusion_iou": IOU_THR,
        "n_organic_test": n_org,
        "n_ewaste_test": n_ew,
        "headline": headline,
        "headline_source": headline_source,
        "synthetic_threshold": synth_conf,
        "real_best": best,
        "real_best_coarse_grid": best_coarse,
        "capacity": capacity,
        "localisation": localisation,
        "fine_step": ev.FINE_STEP,
        "sweep": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {out.name}/report.txt, summary.json, "
          f"threshold_sweep.csv, threshold_sweep_fine.csv, per_image.csv")


if __name__ == "__main__":
    main()
