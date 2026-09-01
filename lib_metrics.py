"""
lib_metrics.py
--------------
The model-characterisation metrics the paper reports alongside detection
accuracy: capacity, cost and localisation quality.

Kept separate from the trainer and the evaluator because both need it and
because EDNet runs in its own virtual environment. That environment has torch
and numpy but no thop, so FLOPs are reported as unavailable there rather than
silently omitted or guessed -- an absent number and a wrong number are not the
same thing.

Conventions, fixed here so every model is measured the same way:

  * FLOPs count one multiply-accumulate as two operations, at 640x640 with
    batch 1, matching how Ultralytics quotes GFLOPs for its own models.
  * Latency times the whole predict call -- preprocess, forward and NMS -- on
    already-decoded images, at batch 1. Disk decode is excluded because it
    measures the storage, not the detector.
  * mIoU and Dice are computed over matched detections only. A ground truth
    the model never found has no IoU to average; it is a recall failure and is
    counted there instead.
"""

import time

import numpy as np


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_gflops(model, imgsz=640):
    """GFLOPs at batch 1, or None when thop is unavailable (the EDNet venv)."""
    try:
        import thop
        import torch
    except ImportError:
        return None
    try:
        device = next(model.parameters()).device
        im = torch.zeros(1, 3, imgsz, imgsz, device=device)
        macs = thop.profile(model, inputs=[im], verbose=False)[0]
        return round(macs * 2 / 1e9, 2)
    except Exception:
        return None


def weight_size_mb(path):
    try:
        return round(path.stat().st_size / 1e6, 2)
    except OSError:
        return None


def measure_latency(predict, images, warmup=20, sample=200):
    """
    Median single-image latency in ms and the FPS it implies.

    ``predict`` takes a list of one image and runs the detector. The median is
    reported rather than the mean because a single scheduling stall would move
    a mean by more than any architectural difference being measured.
    """
    if not images:
        return None, None

    for im in images[:warmup]:
        predict([im])

    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    except ImportError:
        sync = None

    times = []
    for im in images[:sample]:
        if sync:
            sync()
        t0 = time.perf_counter()
        predict([im])
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)

    ms = float(np.median(times))
    return round(ms, 2), round(1000 / ms, 1)


def iou_matrix(pred, gt):
    """IoU of every predicted box against every ground-truth box, both xyxy."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = np.asarray(pred, dtype=float)[:, None, :]
    g = np.asarray(gt, dtype=float)[None, :, :]

    x1 = np.maximum(p[..., 0], g[..., 0])
    y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2])
    y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union = area_p + area_g - inter
    return np.where(union > 0, inter / union, 0.0)


def match_ious(pred, scores, gt, iou_thr=0.5):
    """
    Greedily pair predictions to ground truths, highest confidence first, and
    return the IoU of each accepted pair. One prediction per ground truth.
    """
    if len(pred) == 0 or len(gt) == 0:
        return []
    ious = iou_matrix(pred, gt)
    order = np.argsort(np.asarray(scores))[::-1]
    taken, out = set(), []
    for i in order:
        candidates = [(ious[i, j], j) for j in range(len(gt)) if j not in taken]
        if not candidates:
            break
        best, j = max(candidates)
        if best >= iou_thr:
            taken.add(j)
            out.append(float(best))
    return out


def box_counts(pred, scores, gt, conf, iou_thr=0.5):
    """True positives, false positives and false negatives for one image."""
    kept = [(c, b) for c, b in zip(scores, pred) if c >= conf]
    if not kept:
        return 0, 0, len(gt)
    boxes = [b for _, b in kept]
    confs = [c for c, _ in kept]
    tp = len(match_ious(boxes, confs, gt, iou_thr))
    return tp, len(kept) - tp, len(gt) - tp


def best_box_f1(per_image, grid, iou_thr=0.5):
    """
    Confidence maximising box-level F1 over a held-out set, and that F1.

    ``per_image`` is a list of (confidences, boxes, ground-truth boxes).

    This exists so an ensemble can choose its operating threshold the same way
    a single model does. Ultralytics hands each model a box-level F1 curve over
    the synthetic validation split and the pipeline takes its argmax; a fused
    detector has no such curve, and without one its threshold would have to be
    picked on the test set, which is the bias this is here to avoid.
    """
    best_conf, best_f1 = grid[0], -1.0
    for conf in grid:
        tp = fp = fn = 0
        for scores, boxes, gt in per_image:
            a, b, c = box_counts(boxes, scores, gt, conf, iou_thr)
            tp += a
            fp += b
            fn += c
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_conf, best_f1 = conf, f1
    return float(best_conf), float(best_f1)


def localisation_summary(all_ious, n_gt, iou_thr=0.5):
    """
    Aggregate matched IoUs into the mIoU and Dice the paper reports.

    For an axis-aligned box, Dice is exactly 2*IoU/(1+IoU) -- a monotone
    transform, so it ranks models identically to mIoU and carries no
    independent evidence. It is computed because it was asked for, and this
    relationship is stated wherever it appears.
    """
    if not all_ious:
        return {"mIoU": None, "dice": None, "n_matched": 0, "n_gt": n_gt,
                "iou_thr": iou_thr}
    a = np.asarray(all_ious, dtype=float)
    return {
        "mIoU": round(float(a.mean()), 4),
        "dice": round(float((2 * a / (1 + a)).mean()), 4),
        "n_matched": len(a),
        "n_gt": n_gt,
        "iou_thr": iou_thr,
    }
