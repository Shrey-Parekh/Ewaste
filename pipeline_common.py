"""
pipeline_common.py
------------------
Helpers shared by every trainer in the pipeline.

These live here rather than in 05_train.py because the EDNet comparison
runs in a separate virtual environment that has no Ultralytics installed --
importing 05_train.py from it would fail on its top-level
``from ultralytics import YOLO``. This module imports only NumPy, so both
trainers use exactly the same code to locate the operating point. That
matters: the F1-optimal synthetic confidence is what the paper carries into
the real-image evaluation, so if two models computed it differently the
comparison between them would be meaningless.
"""

import numpy as np


def f1_curve(metrics):
    """
    Return (confidence, f1) arrays for the box F1 curve, or (None, None).

    Ultralytics exposes this in more than one place depending on version, so
    try the direct attributes first and fall back to curves_results. Both the
    current package and the older fork EDNet is built on are covered.
    """
    box = getattr(metrics, "box", None)
    x = getattr(box, "px", None)
    y = getattr(box, "f1_curve", None)
    if x is not None and y is not None:
        y = np.asarray(y, dtype=float)
        if y.ndim > 1:                      # per-class rows; single class here
            y = y.mean(axis=0)
        return np.asarray(x, dtype=float), y

    for entry in getattr(metrics, "curves_results", []) or []:
        try:
            cx, cy, xlabel, ylabel = entry
        except (TypeError, ValueError):
            continue
        if "f1" not in str(ylabel).lower() or "confidence" not in str(xlabel).lower():
            continue
        cy = np.asarray(cy, dtype=float)
        if cy.ndim > 1:
            cy = cy.mean(axis=0)
        return np.asarray(cx, dtype=float), cy
    return None, None


def best_f1_point(metrics):
    """
    Confidence at which F1 peaks, matching the point YOLO marks on
    BoxF1_curve.png, plus the F1 there. (None, None) if unavailable.
    """
    x, y = f1_curve(metrics)
    if x is None or y is None or len(x) != len(y):
        return None, None
    i = int(np.argmax(y))
    return float(x[i]), float(y[i])


def write_f1_curve(out_dir, metrics):
    """
    Persist the raw curve so the operating point stays recoverable later
    without retraining, whatever the metrics API happens to expose.
    """
    cx, cy = f1_curve(metrics)
    if cx is None or cy is None or len(cx) != len(cy):
        return False
    (out_dir / "f1_curve.csv").write_text(
        "confidence,f1\n" + "\n".join(f"{a:.4f},{b:.6f}" for a, b in zip(cx, cy)),
        encoding="utf-8")
    return True
