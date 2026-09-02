"""
src/generate_necks.py
------------------------
Emits the three neck-based architecture configurations.

They are generated rather than hand-written because their neck width and depth
are experimental variables, not constants, and because the FPN and BiFPN arms
have to stay matched. Editing three files by hand to change one number is how
two necks quietly drift apart and the comparison between them stops meaning
anything.

What is held equal between resnet34-fpn-cbam and resnet34-bifpn-cbam:

  * the same ResNet34 backbone
  * the same neck width
  * the same number of neck passes
  * the same CBAM block on each level entering the head

so the only difference left is the fusion topology: a top-down pyramid against
bidirectional flow with learnable weights and an input-to-output skip. That is
the question models 5 and 6 exist to answer, and it is only answerable while
everything else matches.

The repeat count applies to both necks for that reason. Giving BiFPN more
passes than FPN would confound the topology it is meant to test with plain
depth.

Run:  python src/generate_necks.py
      python src/generate_necks.py --width 160 --repeats 3
"""

from pathlib import Path
import argparse

# Emits into models/ at the project root, not beside this file.
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"

# A torchvision ResNet unwrapped to its children is conv1, bn1, relu, maxpool,
# layer1..layer4. With the input prepended by split mode, list index 6 is
# layer2 at stride 8, 7 is layer3 at stride 16 and 8 is layer4 at stride 32.
# ResNet34 uses BasicBlock, so those carry 128, 256 and 512 channels.
RESNET_LEVELS = [(128, 6), (256, 7), (512, 8)]

# YOLO11s backbone at its own scale, written out literally so that every number
# in the generated file means the same thing. The scales block is the identity
# for the same reason: the parser scales the arguments of stock modules but
# passes those of CBAM, BiFPNFuse and Index through untouched, and two
# conventions in one file is a trap.
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


def build(kind, width, repeats):
    is_resnet = kind.startswith("resnet34")
    L = Layers()

    if is_resnet:
        L.add(-1, "TorchVision", "[512, resnet34, DEFAULT, True, 2, True]", "backbone")
        levels = [L.add(0, "Index", f"[{c}, {i}]", f"P{n + 3}/{8 * 2 ** n}")
                  for n, (c, i) in enumerate(RESNET_LEVELS)]
        p3, p4, p5 = levels
    else:
        for mod, args, note in YOLO11S_BACKBONE:
            L.add(-1, mod, args, note)
        p3, p4, p5 = YOLO11S_LEVELS
    backbone_lines = list(L.lines)

    # lateral 1x1 projections bring every level to the common neck width
    p3 = L.add(p3, "Conv", f"[{width}, 1, 1]", "P3 lateral")
    p4 = L.add(p4, "Conv", f"[{width}, 1, 1]", "P4 lateral")
    p5 = L.add(p5, "Conv", f"[{width}, 1, 1]", "P5 lateral")

    step = bifpn_pass if "bifpn" in kind else fpn_pass
    for r in range(repeats):
        p3, p4, p5 = step(L, p3, p4, p5, width, f"(pass {r + 1})")

    a3 = L.add(p3, "CBAM", f"[{width}]")
    a4 = L.add(p4, "CBAM", f"[{width}]")
    a5 = L.add(p5, "CBAM", f"[{width}]")
    L.add(f"[{a3}, {a4}, {a5}]", "Detect", "[nc]")
    head_lines = L.lines[len(backbone_lines):]

    neck = "BiFPN" if "bifpn" in kind else "FPN"
    other = "FPN" if neck == "BiFPN" else "BiFPN"
    back = "ResNet34" if is_resnet else "YOLO11s"

    out = [
        f"# {back} + {neck} + CBAM",
        "#",
        "# GENERATED by src/generate_necks.py -- do not edit by hand.",
        f"# Neck width {width}, {repeats} neck pass(es).",
        "#",
        f"# Width and pass count are shared with the matching {other} configuration,",
        "# so the only difference between those two arms is the fusion topology.",
        "#",
        "# CBAM here is the open-gate variant bound in lib_modules.py: its gates",
        "# start uniform and open rather than as a random mask, so appending it to a",
        "# pretrained network does not corrupt the features on the first forward pass.",
    ]
    if is_resnet:
        out += [
            "#",
            "# The backbone arrives with ImageNet weights inside the TorchVision layer.",
            "# Everything after it is new, which is what the warm-up phase in",
            "# 05_train.py exists to settle.",
        ]
    else:
        out += [
            "#",
            "# Replacing the neck discards its COCO weights; only the backbone, layers",
            "# 0-10, is inherited.",
        ]
    out += ["", "nc: 1", ""]
    if not is_resnet:
        out += ["scales:",
                "  # [depth, width, max_channels] -- identity; channels below are literal",
                "  s: [1.0, 1.0, 1024]", ""]
    out += ["backbone:", "  # [from, repeats, module, args]"] + backbone_lines
    out += ["", "head:"] + head_lines
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128,
                    help="common neck width, applied to both FPN and BiFPN")
    ap.add_argument("--repeats", type=int, default=2,
                    help="neck passes, applied to both necks so depth stays matched")
    args = ap.parse_args()

    for kind in ("resnet34-fpn-cbam", "resnet34-bifpn-cbam", "yolo11s-bifpn-cbam"):
        (OUT / f"{kind}.yaml").write_text(build(kind, args.width, args.repeats),
                                           encoding="utf-8")
        print(f"wrote {kind}.yaml  (width {args.width}, {args.repeats} pass(es))")


if __name__ == "__main__":
    main()
