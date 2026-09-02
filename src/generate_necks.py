"""
src/generate_necks.py
---------------------
Emits the backbone-and-neck architecture configurations.

Three backbones, each paired with both necks, plus the BiFPN variant of
YOLO11s: seven files in all. They are generated rather than hand-written
because their neck width and depth are experimental variables, not constants,
and because every pair has to stay matched. Editing seven files by hand to
change one number is how two arms quietly drift apart and the comparison
between them stops meaning anything.

What is held equal between a backbone's FPN arm and its BiFPN arm:

  * the same backbone
  * the same neck width
  * the same number of neck passes
  * the same CBAM block on each level entering the head

so the only difference left is the fusion topology: a top-down pyramid against
bidirectional flow with learnable weights and an input-to-output skip. The
repeat count applies to both necks for that reason -- giving BiFPN more passes
than FPN would confound the topology it is meant to test with plain depth.

Backbone pyramid levels were measured, not looked up. Each backbone was run at
640 px through TVBackbone and the last feature map at each of strides 8, 16 and
32 was recorded. Two of those measurements settled real questions:

  * Inception v3 cannot be used. Its stem convolutions are unpadded, so at
    640 px it yields 77, 38 and 18 pixel maps -- strides of 8.31, 16.84 and
    35.56. Upsampling its P5 by two gives 36 against a P4 of 38, so the neck
    cannot even concatenate, and the detection head would derive fractional
    strides. GoogLeNet, which is Inception v1, gives exactly 8, 16 and 32 and
    is used instead.

  * EfficientNet's P5 is the last block at stride 32, 320 channels, not the
    1280-channel projection that follows it. That projection exists to feed a
    classifier; EfficientDet takes the block outputs, and so does this.

Run:  python src/generate_necks.py
      python src/generate_necks.py --width 160 --repeats 3
"""

from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"

# name -> torchvision model, and the (channels, split index) of P3, P4, P5.
# Indices are into the list TVBackbone returns, whose entry 0 is the input.
BACKBONES = {
    "resnet18": {
        "tv": "resnet18",
        "label": "ResNet18",
        "levels": [(128, 6), (256, 7), (512, 8)],
    },
    "googlenet": {
        "tv": "googlenet",
        "label": "GoogLeNet (Inception v1)",
        "levels": [(480, 7), (832, 13), (1024, 16)],
    },
    "efficientnet": {
        "tv": "efficientnet_b0",
        "label": "EfficientNet-B0",
        "levels": [(40, 4), (112, 6), (320, 8)],
    },
}

# YOLO11s backbone at its own scale, written out literally so every number in
# the generated file means the same thing. The scales block is the identity for
# the same reason: the parser scales the arguments of stock modules but passes
# those of CBAM, BiFPNFuse and Index through untouched, and two conventions in
# one file is a trap.
YOLO11S_BACKBONE = [
    ("Conv", "[32, 3, 2]", "P1/2"),
    ("Conv", "[64, 3, 2]", "P2/4"),
    ("C3k2", "[128, False, 0.25]", ""),
    ("Conv", "[128, 3, 2]", "P3/8"),
    ("C3k2", "[256, False, 0.25]", ""),
    ("Conv", "[256, 3, 2]", "P4/16"),
    ("C3k2", "[256, True]", ""),
    ("Conv", "[512, 3, 2]", "P5/32"),
    ("C3k2", "[512, True]", ""),
    ("SPPF", "[512, 5]", ""),
    ("C2PSA", "[512]", ""),
]
YOLO11S_LEVELS = (4, 6, 10)


class Layers:
    """Accumulates YAML layer lines and hands back the index of each."""

    def __init__(self):
        self.lines = []

    def add(self, frm, module, args, note=""):
        idx = len(self.lines)
        tail = f" # {idx}" + (f" {note}" if note else "")
        self.lines.append(f"  - [{frm}, 1, {module}, {args}]{tail}")
        return idx


def up(L, src):
    return L.add(src, "nn.Upsample", '[None, 2, "nearest"]')


def down(L, src, w):
    return L.add(src, "Conv", f"[{w}, 3, 2]")


def fuse(L, inputs, w, note=""):
    """Concat, weighted fusion, then Index to declare the fused width."""
    n = len(inputs)
    L.add("[" + ", ".join(str(i) for i in inputs) + "]", "Concat", "[1]")
    L.add(-1, "BiFPNFuse", f"[{n * w}, {w}, {n}]")
    return L.add(-1, "Index", f"[{w}, 0]", note)


def bifpn_pass(L, p3, p4, p5, w, tag):
    """One BiFPN block: top-down, then bottom-up carrying the input-to-output skip."""
    p4_td = fuse(L, [p4, up(L, p5)], w, f"P4 top-down {tag}")
    p3_out = fuse(L, [p3, up(L, p4_td)], w, f"P3 out {tag}")
    # Three inputs: the level's own input, its top-down result, and the level
    # below. That skip from input straight to output is what separates BiFPN
    # from a plain bidirectional pyramid.
    p4_out = fuse(L, [p4, p4_td, down(L, p3_out, w)], w, f"P4 out {tag}")
    p5_out = fuse(L, [p5, down(L, p4_out, w)], w, f"P5 out {tag}")
    return p3_out, p4_out, p5_out


def fpn_pass(L, p3, p4, p5, w, tag):
    """One classical top-down pass. P5 carries through as the lateral."""
    L.add("[" + f"{up(L, p5)}, {p4}" + "]", "Concat", "[1]")
    p4_out = L.add(-1, "Conv", f"[{w}, 3, 1]", f"P4 out {tag}")
    L.add("[" + f"{up(L, p4_out)}, {p3}" + "]", "Concat", "[1]")
    p3_out = L.add(-1, "Conv", f"[{w}, 3, 1]", f"P3 out {tag}")
    return p3_out, p4_out, p5


def build(backbone, neck, width, repeats):
    """backbone is a key of BACKBONES, or 'yolo11s'. neck is 'fpn' or 'bifpn'."""
    is_yolo = backbone == "yolo11s"
    spec = None if is_yolo else BACKBONES[backbone]
    L = Layers()

    if is_yolo:
        for mod, args, note in YOLO11S_BACKBONE:
            L.add(-1, mod, args, note)
        p3, p4, p5 = YOLO11S_LEVELS
    else:
        L.add(-1, "TVBackbone", f"[512, {spec['tv']}, 2]", "backbone")
        p3, p4, p5 = [
            L.add(0, "Index", f"[{c}, {i}]", f"P{n + 3}/{8 * 2 ** n}")
            for n, (c, i) in enumerate(spec["levels"])
        ]
    backbone_lines = list(L.lines)

    # lateral 1x1 projections bring every level to the common neck width
    p3 = L.add(p3, "Conv", f"[{width}, 1, 1]", "P3 lateral")
    p4 = L.add(p4, "Conv", f"[{width}, 1, 1]", "P4 lateral")
    p5 = L.add(p5, "Conv", f"[{width}, 1, 1]", "P5 lateral")

    step = bifpn_pass if neck == "bifpn" else fpn_pass
    for r in range(repeats):
        p3, p4, p5 = step(L, p3, p4, p5, width, f"(pass {r + 1})")

    a3 = L.add(p3, "CBAM", f"[{width}]")
    a4 = L.add(p4, "CBAM", f"[{width}]")
    a5 = L.add(p5, "CBAM", f"[{width}]")
    L.add(f"[{a3}, {a4}, {a5}]", "Detect", "[nc]")
    head_lines = L.lines[len(backbone_lines):]

    neck_label = "BiFPN" if neck == "bifpn" else "FPN"
    other = "FPN" if neck == "bifpn" else "BiFPN"
    back_label = "YOLO11s" if is_yolo else spec["label"]

    out = [
        f"# {back_label} + {neck_label} + CBAM",
        "#",
        "# GENERATED by src/generate_necks.py -- do not edit by hand.",
        f"# Neck width {width}, {repeats} neck pass(es).",
        "#",
        f"# Width and pass count are shared with the matching {other} configuration,",
        "# so the only difference between those two arms is the fusion topology.",
        "#",
        "# CBAM here is the open-gate variant bound in src/lib_modules.py: its gates",
        "# start uniform and open rather than as a random mask, so appending it to a",
        "# pretrained network does not corrupt the features on the first forward pass.",
    ]
    if is_yolo:
        out += [
            "#",
            "# Replacing the neck discards its COCO weights; only the backbone, layers",
            "# 0-10, is inherited.",
        ]
    else:
        levels = ", ".join(f"{c}ch at index {i}" for c, i in spec["levels"])
        out += [
            "#",
            f"# Backbone is torchvision {spec['tv']} with ImageNet weights, taken through",
            "# TVBackbone so its classifier head, and any auxiliary classifier branch,",
            "# are removed before the children are run as a sequence.",
            f"# Pyramid levels, measured at 640 px: {levels}.",
            "# Everything after the backbone is new, which is what the warm-up phase in",
            "# src/05_train.py exists to settle.",
        ]
    out += ["", "nc: 1", ""]
    if is_yolo:
        out += ["scales:",
                "  # [depth, width, max_channels] -- identity; channels below are literal",
                "  s: [1.0, 1.0, 1024]", ""]
    out += ["backbone:", "  # [from, repeats, module, args]"] + backbone_lines
    out += ["", "head:"] + head_lines
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128,
                    help="common neck width, applied to every arm")
    ap.add_argument("--repeats", type=int, default=2,
                    help="neck passes, applied to both necks so depth stays matched")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    targets = [(b, n) for b in BACKBONES for n in ("fpn", "bifpn")]
    targets.append(("yolo11s", "bifpn"))

    for backbone, neck in targets:
        name = f"{backbone}-{neck}-cbam"
        (OUT / f"{name}.yaml").write_text(build(backbone, neck, args.width, args.repeats),
                                          encoding="utf-8")
        print(f"wrote {name}.yaml  (width {args.width}, {args.repeats} pass(es))")


if __name__ == "__main__":
    main()
