"""
pipeline_common.py
------------------
Helpers shared by every trainer in the pipeline.

It also holds the defensive image decoder, which the evaluator and the
annotator both need because they read the same photograph corpus.

The trainer helpers live here rather than in 05_train.py so that anything
needing the operating point can import them without pulling in Ultralytics and
a whole training stack. The F1-optimal synthetic confidence is what the paper
carries into the real-image evaluation, so if two models computed it
differently the comparison between them would be meaningless.
"""

import numpy as np
from PIL import Image, ImageOps


def load_image(path):
    """
    Decode a photograph defensively rather than handing the path to a library.

    Ultralytics reads exif orientation via PIL.ImageOps.exif_transpose with no
    error handling, and a handful of TrashBox photographs carry a malformed
    EXIF block that raises SyntaxError there and aborts the whole batch. This
    project's corpus is not going to get its EXIF fixed upstream, so try
    exif_transpose and fall back to the untransposed image on failure. A wrong
    orientation on one in ~400 photographs is a rounding error next to the
    evaluation crashing before it finishes.
    """
    im = Image.open(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im.convert("RGB")


def f1_curve(metrics):
    """
    Return (confidence, f1) arrays for the box F1 curve, or (None, None).

    Ultralytics exposes this in more than one place depending on version, so
    try the direct attributes first and fall back to curves_results. Both the
    older forks of it are covered as well as the current release.
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
