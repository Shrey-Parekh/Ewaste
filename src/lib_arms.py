"""
lib_arms.py
-----------
The one list of models in the comparison.

Every arm is (label, model argument, run tag). The tag is what names a run
directory and an evaluation directory, so it is also what the ensemble and the
metrics table look models up by.

This exists because that list was written out separately in the ensemble and in
the metrics table, and a third copy was about to be added for the launcher.
Three hand-maintained copies of the same eleven rows is how a model gets
renamed in one place, silently dropped from another, and quietly missing from a
results table nobody re-checked.

Order is the order results are presented in: baselines, attention, then each
backbone with both necks, then the BiFPN variant of YOLO11s.
"""

ARMS = [
    ("YOLOv8s", "yolov8s.pt", ""),
    ("YOLOv11s", "yolo11s.pt", "yolo11s"),
    ("YOLOv8s+CBAM", "models/yolov8s-cbam.yaml", "v8s_cbam"),
    ("YOLOv11s+CBAM", "models/yolo11s-cbam.yaml", "v11s_cbam"),
    ("ResNet18+FPN+CBAM", "models/resnet18-fpn-cbam.yaml", "r18_fpn_cbam"),
    ("ResNet18+BiFPN+CBAM", "models/resnet18-bifpn-cbam.yaml", "r18_bifpn_cbam"),
    ("GoogLeNet+FPN+CBAM", "models/googlenet-fpn-cbam.yaml", "gnet_fpn_cbam"),
    ("GoogLeNet+BiFPN+CBAM", "models/googlenet-bifpn-cbam.yaml", "gnet_bifpn_cbam"),
    ("EfficientNet+FPN+CBAM", "models/efficientnet-fpn-cbam.yaml", "effnet_fpn_cbam"),
    ("EfficientNet+BiFPN+CBAM", "models/efficientnet-bifpn-cbam.yaml", "effnet_bifpn_cbam"),
    ("YOLOv11s+BiFPN+CBAM", "models/yolo11s-bifpn-cbam.yaml", "v11s_bifpn_cbam"),
]

# (label, tag) only, which is all the ensemble and the metrics table need.
MEMBERS = [(label, tag) for label, _, tag in ARMS]


def run_name(pool, tag):
    """Directory a run lands in, and the suffix its evaluation directory uses."""
    return f"pool{pool}" + (f"_{tag}" if tag else "")
