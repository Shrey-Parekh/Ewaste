# Implementation report: e-waste detection in wet biodegradable waste

This report documents how every model in the study was built, trained, evaluated and explained. It covers every architecture, parameter, hyperparameter and setting, and gives the results each model produced. Every value was read from the code in `src/`, the model configurations in `models/`, the recorded run arguments (`runs/detect/*/args.yaml`), the run summaries (`synthetic_summary.json`, `eval_*/summary.json`) and the paper's statistics scripts (`src/paper/`).

---

## 1. Task and overall design

**Task.** Look at a photograph of wet organic waste (food scraps, garden waste) and raise an alert if electronic waste is present, so that an operator can inspect the load before composting or digestion. The decision is made per image. A photograph counts as flagged when the detector fires anywhere in it; whether the box lands on the object is not scored.

**Central constraint.** No public dataset contains real photographs of e-waste buried in real organic waste. Every detector is therefore trained only on synthetic composites, which are real e-waste cut-outs pasted into real organic-waste photographs. Every detector is evaluated only on real photographs that contributed nothing to training.

**Models.** Eleven single-model detectors ("arms") and one ensemble. YOLOv11s is the detector the paper proposes; the other ten arms and the ensemble form the ablation study.

| # | Label | Config / checkpoint | Run tag |
|---|---|---|---|
| 1 | YOLOv8s | `yolov8s.pt` | *(none)* |
| 2 | **YOLOv11s** (proposed) | `yolo11s.pt` | `yolo11s` |
| 3 | YOLOv8s + CBAM | `models/yolov8s-cbam.yaml` | `v8s_cbam` |
| 4 | YOLOv11s + CBAM | `models/yolo11s-cbam.yaml` | `v11s_cbam` |
| 5 | ResNet18 + FPN + CBAM | `models/resnet18-fpn-cbam.yaml` | `r18_fpn_cbam` |
| 6 | ResNet18 + BiFPN + CBAM | `models/resnet18-bifpn-cbam.yaml` | `r18_bifpn_cbam` |
| 7 | GoogLeNet + FPN + CBAM | `models/googlenet-fpn-cbam.yaml` | `gnet_fpn_cbam` |
| 8 | GoogLeNet + BiFPN + CBAM | `models/googlenet-bifpn-cbam.yaml` | `gnet_bifpn_cbam` |
| 9 | EfficientNet-B0 + FPN + CBAM | `models/efficientnet-fpn-cbam.yaml` | `effnet_fpn_cbam` |
| 10 | EfficientNet-B0 + BiFPN + CBAM | `models/efficientnet-bifpn-cbam.yaml` | `effnet_bifpn_cbam` |
| 11 | YOLOv11s + BiFPN + CBAM | `models/yolo11s-bifpn-cbam.yaml` | `v11s_bifpn_cbam` |
| 12 | Ensemble (weighted box fusion of 1–11) | `src/10_ensemble.py` | `ensemble` |

This list is defined once, in `src/lib_arms.py`, and every script reads it from there. Run directories are named `runs/detect/pool54[_<tag>]` and evaluation directories `evaluation/pool54[_<tag>]`.

**Pipeline** (all scripts in `src/`):

| Step | Script | Output |
|---|---|---|
| 1 | `01_build_splits.py` | Disjoint role manifests in `splits/` |
| 2 | `02_make_cutouts.py` (+ `lib_segment.py`) | Transparent cut-outs of e-waste objects and organic occluders |
| 3 | `03_screen_cutouts.py` | Screened e-waste cut-outs, `cutouts/ewaste_clean/` |
| 4 | `04_build_dataset.py` (+ `lib_composite.py`) | Synthetic dataset `dataset_pool54/` |
| 5 | `05_train.py` (+ `lib_modules.py`) | Trained weights and synthetic validation summary per arm |
| 6 | `06_evaluate.py` | Real-image evaluation per arm |
| 7 | `10_ensemble.py` | Weighted-box-fusion ensemble and its evaluation |
| 8 | `14_latency.py` | Latency of all arms, interleaved in one process |
| 9 | `src/paper/*.py` | Statistics, figures, calibrated thresholds, Grad-CAM |

`generate_necks.py` writes the seven FPN/BiFPN configurations.

---

## 2. Data

### 2.1 Sources and role split (`01_build_splits.py`, seed 0)

| Role | Count | Source | Used for |
|---|---|---|---|
| `ewaste_pool` | 61 (54 after screening) | Curated photographs, `raw/ewaste` | Objects pasted into the composites |
| `organic_bg` | 50 | RealWaste: *Food Organics*, *Vegetation* | Backgrounds of the composites |
| `organic_clutter` | 50 | RealWaste | Cut-outs used as clutter and occluders |
| `organic_test` | 746 | RealWaste | **Test**: false alarms (contains no e-waste) |
| `ewaste_test` | 387 | TrashBox `e-waste` (cables 175, chips/boards 145, smartphones 67) | **Test**: detection |

How the counts arise:
- **E-waste pool.** 68 curated photographs, less 7 that are the same picture as another (4 copies of test photographs and 3 pictures held twice), gives 61. Screening then rejects 7 more, leaving **54**.
- **Organic.** 847 RealWaste photographs, less 1 repeat, gives 846. These split into 50 background, 50 clutter and 746 test.
- **E-waste test.** 400 TrashBox photographs were drawn; 13 repeats were removed, leaving 387. The 52 TrashBox photographs that belonged to the curated pool were excluded.

The roles are pairwise disjoint. This was checked with a content (byte) hash across all roles. The e-waste pool was also checked with a perceptual hash against the test set and against itself: a 256-bit difference hash, with photographs counted as the same image at a Hamming distance of 10 bits or less. Re-saved copies measure 0–2 bits apart, and the closest pair of genuinely different photographs measures 34.

### 2.2 Cut-out extraction (`02_make_cutouts.py`, `lib_segment.py`)

| Setting | Value |
|---|---|
| Background removal | `rembg`, model `isnet-general-use` |
| Alpha matting | on (soft edges) |
| Edge erosion | 1 px (removes the halo left by the old background) |
| Alpha feather | Gaussian, σ = 0.8 px |
| Colour despill | on |
| Maximum input side | 1536 px (larger photographs downscaled before matting) |
| Quality gates | kept coverage 2–97% of the frame; shortest side ≥ 40 px |

### 2.3 Cut-out screening (`03_screen_cutouts.py`)

A COCO-pretrained YOLOv8s screens every e-waste cut-out, which is composited onto white for the check.

| Rule | Threshold |
|---|---|
| Screening confidence | 0.35 |
| Reject if a person is detected | conf ≥ 0.50 |
| Reject if the dominant object is non-electronic (cup, fruit, furniture …) | conf ≥ 0.50 |
| Reject as a collage | ≥ 3 distinct electronic devices |
| Reject as a matting failure | solid-alpha fraction < 0.25 |

54 cut-outs pass.

### 2.4 Synthetic scene composition (`lib_composite.py`, driven by `04_build_dataset.py`)

**Dataset.** 1500 images of 640 × 640 px, seed 0. The first 15% form the validation split, giving **1275 train and 225 validation** images. One class: `ewaste_contaminant`.

**Identity-disjoint validation.** Whole identities are split, stratified by category, with 20% going to validation:

| Kind | Train | Validation |
|---|---|---|
| E-waste objects | 44 | 10 |
| Organic occluders | 40 | 10 |
| Backgrounds | 40 | 10 |

The assignment is written to `splits/synthetic_pool54.csv`. Each kind uses its own random stream (`identity-split-<kind>-0`).

**How one composite is built.** The layers go down in order: background, then clutter, then e-waste, then occluders, then a camera pass.

| Stage | Parameter | Value |
|---|---|---|
| Background | random square crop | 60–100% of the short side, resized to 640 |
| | flips | horizontal p = 0.5, vertical p = 0.2 |
| Organic clutter (layer 1) | pieces per image | 2–6 |
| | size | 5–36% of the frame |
| | colour harmonisation strength | 0.7 × 0.45 |
| | may extend off-frame | 40% |
| E-waste objects (layer 2) | per image | 1–3 |
| | size (longest side / frame) | 0.05–0.22 |
| | fragment crop | p = 0.45, keeps 35–75% of each side |
| | rotation | uniform 0–360° |
| | perspective warp | p = 0.6, corners jittered ±6% |
| | colour harmonisation strength | 0.45 (contrast gain × 0.6·s, mean shift × s) |
| | sharpness match | Gaussian blur, radius (detail ratio − 1) × 0.45, capped at 1.6 |
| | grain match | Gaussian noise at the background's noise level (when > 0.3) |
| | contact shadow | p = 0.9, opacity 0.20–0.45, blur 10% of object size, offset dx ±10%, dy +2…14% |
| | may extend off-frame | 20% |
| Occluders (layer 3) | objects occluded | p = 0.85 per object |
| | occluders per object | 1–3 |
| | occluder size | 0.45–1.1 × the object's span |
| | placement jitter | ±30% of the object's size, around its centre |
| Labels | visibility tracking | per-pixel owner mask; occluded pixels removed |
| | minimum visible fraction φ | 0.35 (box dropped if less is visible) |
| | minimum box side | 8 px |
| | box | tight box around the **visible** pixels only |
| Camera pass | blur | σ 0–0.9 |
| | brightness / colour / contrast | 0.72–1.06 / 0.72–1.08 / 0.88–1.08 |
| | colour cast (R, G, B) | [−7, 5], [−4, 6], [−10, 3] |
| | vignette | p = 0.7, strength 0.10–0.30 |
| | sensor noise | σ 2–9 |
| | JPEG round trip | quality 72–93 (saved at quality 92) |

The visible fraction of every placed object is logged to `dataset_pool54/visible_fraction.csv`.

---

## 3. Model architectures

### 3.1 Parts shared by every arm

**Detection head.** All arms use the Ultralytics `Detect` head:
- Anchor-free and decoupled, with separate box and class branches.
- Three levels, P3/8, P4/16 and P5/32, which give 80×80, 40×40 and 20×20 grids at 640 px: 8400 locations in all.
- Distribution Focal Loss box regression with `reg_max = 16`.
- `nc = 1`. The head is rebuilt for one class in every arm, so it never inherits COCO weights.

**Loss.** These are Ultralytics `v8DetectionLoss` defaults. The recorded gains are `box = 7.5`, `cls = 0.5` and `dfl = 1.5`.
- Box term: CIoU loss.
- Class term: BCE-with-logits.
- DFL term: Distribution Focal Loss.
- Label assignment: TaskAlignedAssigner with top-k 10, α = 0.5, β = 6.0.

**Input.** 640 × 640 RGB, scaled to [0, 1].

### 3.2 CBAM with open gates (`lib_modules.CBAMOpenGate`)

Every "+CBAM" arm uses this block, which the model YAMLs call `CBAM`.

- **Channel gate** (Ultralytics form):
  - global average pool
  - 1×1 convolution, C → C, with bias
  - sigmoid
  - multiply the input

  Unlike the original CBAM, it has no reduction MLP and no max-pool branch.
- **Spatial gate**:
  - channel-wise mean and max maps, concatenated
  - 7×7 convolution, 2 → 1
  - sigmoid
  - multiply the input
- **Open-gate initialisation (this study).** Both gate convolutions get **zero weights and a bias of +2.0**.
  - At step 0 each gate outputs sigmoid(2) = 0.88 uniformly. The block then starts as a uniform gain, not a random mask over pretrained features.
  - The local sigmoid gradient there is 0.105, so the gates still learn.
  - Stock Ultralytics `SpatialAttention` has `bias=False`, so its convolution is replaced by an equal one that has a bias.
  - The stock random initialisation had cost YOLOv8s 5.8 points of detection.
- **Placement.** CBAM goes on the three feature maps entering the head, and is **appended after** the last neck layer. Every earlier layer therefore keeps its index, and COCO weights still transfer.

### 3.3 BiFPN fusion node (`lib_modules.BiFPNFuse`)

This is EfficientDet's fast normalised fusion (Tan et al. 2020, Eq. 3):

```
O = PWConv1x1( DWConv3x3( sum_i  w_i * I_i / (eps + sum_j w_j) ) ),   w_i = ReLU(learned scalar), eps = 1e-4
```

- Inputs are concatenated along channels and split back into n tensors of `w` channels each.
- The fusion weights are initialised to 1.
- Each `Conv` here is Conv2d + BatchNorm + SiLU.

### 3.4 Torchvision backbone wrapper (`lib_modules.TVBackbone`)

- Loads a torchvision classifier with **ImageNet-1k weights**. For ResNet18 and GoogLeNet these are `IMAGENET1K_V1`; the wrapper requests `weights="DEFAULT"`.
- Deletes auxiliary classifiers (GoogLeNet `aux1`/`aux2`) so the network runs as a plain sequence.
- Flattens EfficientNet's `features` Sequential.
- Drops the last two children (the pooling and classifier layers).
- Returns every intermediate map. An `Index` layer then selects the P3, P4 and P5 maps:

| Backbone | P3/8 (index, channels) | P4/16 | P5/32 |
|---|---|---|---|
| ResNet18 | 6 (layer2), 128 | 7 (layer3), 256 | 8 (layer4), 512 |
| GoogLeNet (Inception v1) | 7 (inception3b), 480 | 13 (inception4e), 832 | 16 (inception5b), 1024 |
| EfficientNet-B0 | 4 (features[3]), 40 | 6 (features[5]), 112 | 8 (features[7]), 320 |

- Inception v3 was rejected. Its unpadded stem gives strides of 8.31, 16.84 and 35.56 at 640 px, which the neck cannot fuse.
- EfficientNet's P5 is the 320-channel block, not the 1280-channel classifier projection. EfficientDet uses the block output too.

### 3.5 The two necks for torchvision backbones (`generate_necks.py`)

Both necks are **128 channels wide** and run **2 passes**. Every level is refined by a 3×3 Conv in each pass, and CBAM(128) sits on each output. The FPN and BiFPN arms of one backbone therefore differ only in fusion topology.

- **Lateral:** a 1×1 Conv maps each backbone level to 128 channels.
- **FPN (top-down only).** In each pass:
  - P5 goes through a 3×3 Conv.
  - Upsample ×2, concatenate with P4, 3×3 Conv to get P4.
  - Upsample ×2, concatenate with P3, 3×3 Conv to get P3.
  - Pass 2 repeats this on the outputs of pass 1.
- **BiFPN (bidirectional, weighted).** In each pass:
  - Top-down:
    - P4_td = Fuse(P4, up(P5))
    - P3_out = Fuse(P3, up(P4_td))
  - Bottom-up:
    - P4_out = Fuse(P4, P4_td, down(P3_out)), a three-input node with the skip from the input
    - P5_out = Fuse(P5, down(P4_out))
  - "down" is a stride-2 3×3 Conv.
  - Pass 2 repeats this on the outputs of pass 1.

### 3.6 Per-model detail

Values: parameters are from the paper's cost table; GFLOPs are at 640 px, batch 1, with a multiply-accumulate counted as 2 operations (thop); size is `best.pt` in MB; frozen layers are those held fixed in the warm-up (§4.2).

| Arm | Backbone (pretraining) | Neck | Attention | Params (M) | GFLOPs | Size (MB) | Frozen layers |
|---|---|---|---|---|---|---|---|
| YOLOv8s | CSPDarknet with C2f + SPPF (COCO) | PAN (C2f), COCO | — | 11.13 | 28.65 | 22.53 | 0–21 |
| **YOLOv11s** | C3k2 + SPPF + C2PSA (COCO) | PAN (C3k2), COCO | — (C2PSA is native) | 9.41 | 21.55 | 19.19 | 0–22 |
| YOLOv8s+CBAM | as YOLOv8s | PAN, COCO | CBAM(128/256/512) | 11.47 | 28.65\* | 23.23 | 0–21 |
| YOLOv11s+CBAM | as YOLOv11s | PAN, COCO | CBAM(128/256/512) | 9.76 | 21.55\* | 19.89 | 0–22 |
| ResNet18+FPN+CBAM | ResNet18 (ImageNet) | FPN 128×2, new | CBAM(128)×3 | 14.05 | 46.76 | 28.32 | backbone (layer 0) |
| ResNet18+BiFPN+CBAM | ResNet18 (ImageNet) | BiFPN 128×2, new | CBAM(128)×3 | 13.30 | 38.99 | 26.88 | backbone |
| GoogLeNet+FPN+CBAM | GoogLeNet (ImageNet) | FPN 128×2, new | CBAM(128)×3 | 8.66 | 42.51 | 17.63 | backbone |
| GoogLeNet+BiFPN+CBAM | GoogLeNet (ImageNet) | BiFPN 128×2, new | CBAM(128)×3 | 7.91 | 34.74 | 16.19 | backbone |
| EffNet-B0+FPN+CBAM | EfficientNet-B0 (ImageNet) | FPN 128×2, new | CBAM(128)×3 | 6.82 | 23.51 | 14.06 | backbone |
| EffNet-B0+BiFPN+CBAM | EfficientNet-B0 (ImageNet) | BiFPN 128×2, new | CBAM(128)×3 | 6.08 | 15.74 | 12.62 | backbone |
| YOLOv11s+BiFPN+CBAM | YOLOv11s backbone, layers 0–10 (COCO) | BiFPN 128×2, new | CBAM(128)×3 | 6.80 | 17.21 | 13.97 | 0–10 |
| Ensemble | all 11 | WBF | — | 105.39 | — | 214.52 | — |

\* thop does not count the custom CBAM block, so the recorded GFLOPs equal the baseline. The true extra cost is small: two small convolutions per level.

**YOLOv8s.** The stock `yolov8s` (scale s: depth 0.33, width 0.50).
- Backbone: Conv(64,3,2), Conv(128,3,2), 3×C2f(128), Conv(256,3,2), 6×C2f(256), Conv(512,3,2), 6×C2f(512), Conv(1024,3,2), 3×C2f(1024), SPPF(1024,5). All widths shown before scaling.
- Neck: a PAN neck. Top-down path: upsample + concat + C2f at P4 (layer 12) and P3 (layer 15). Bottom-up path: stride-2 Conv + concat + C2f at P4 (layer 18) and P5 (layer 21).
- Head: Detect on layers 15, 18 and 21.

**YOLOv11s** (proposed). The stock `yolo11s` (depth 0.50, width 0.50).
- Backbone: C3k2 blocks, then SPPF(1024,5) and **C2PSA** at layer 10, a position-sensitive attention block native to YOLO11.
- Neck: a PAN neck with C3k2 blocks.
- Head: Detect on layers 16 (P3), 19 (P4) and 22 (P5).

**+CBAM variants.** The same YAML as the base model, with CBAM blocks added as layers 22–24 (v8) or 23–25 (v11) on the three neck outputs. Detect then reads those. Because the channel widths are given after scaling (128/256/512), the parser passes them through untouched.

**YOLOv11s+BiFPN+CBAM.** The YOLOv11s backbone (layers 0–10, written with literal channels, scale 1.0) keeps its COCO weights. The PAN neck is replaced by the 128-wide, 2-pass BiFPN, with CBAM(128) on the P3, P4 and P5 outputs.

---

## 4. Training (`src/05_train.py`)

### 4.1 One shared schedule

No setting varies between arms, so any difference between them can be attributed to the architecture.

| Setting | Value | Note |
|---|---|---|
| Framework | Ultralytics 8.4.82, PyTorch 2.11.0, torchvision 0.26.0, CUDA 12.8 | |
| Epochs (main phase) | **150** | fixed; `patience = 0` turns early stopping off |
| Image size | 640 | |
| Batch | 16 | measured VRAM limit on an 8 GB card |
| Nominal batch (`nbs`) | 64 | gradients accumulated over 4 steps |
| Optimiser | `auto`, which resolves to **AdamW** | Ultralytics picks AdamW when iterations ≤ 10 000; here ⌈1275/64⌉ × 150 = 3000 |
| Initial LR | **0.002** | auto fit: 0.002 × 5 / (4 + nc), nc = 1 |
| β1 / β2 | 0.9 / 0.999 | |
| Weight decay | 0.0005 | scaled by batch × accumulate / nbs = 1; not applied to biases or BatchNorm |
| LR schedule | linear decay to lr0 × `lrf` (0.01) over the epochs | `cos_lr = False` |
| LR warm-up | 3 epochs; momentum 0.8 rising to 0.9; bias LR 0 (auto) | Ultralytics' built-in warm-up, separate from §4.2 |
| Mixed precision | AMP on | |
| Seed | 0, `deterministic = True` | |
| Workers | 2 | measured: host RAM 12.5 GB (against 25.1 GB at 8 workers); 5% slower |
| Cache | off | RAM caching was slower on Windows spawn |
| Checkpoint kept | `best.pt` by Ultralytics fitness = **mAP@0.50:0.95** on synthetic validation | |
| Validation NMS | IoU 0.7, max 300 detections | |

**Augmentation** (applied on top of the compositing randomisation):

| Augment | Value |
|---|---|
| Mosaic | 1.0; switched off for the last **15** epochs (`close_mosaic = 15`) |
| Scale | ±0.5 |
| Translate | ±0.2 |
| Rotation | ±15° |
| Horizontal / vertical flip | 0.5 / 0.3 |
| HSV (h, s, v) | 0.015, 0.6, 0.4 |
| Random erasing | 0.3 |
| Copy-paste, mixup, cutmix, shear, perspective | 0 |

### 4.2 Two-phase training: a frozen warm-up, then full training

1. **Weight transfer.**
   - Stock `.pt` arms load the COCO checkpoint directly.
   - YOLO-based YAML arms load matching layers from `yolov8s.pt` or `yolo11s.pt`, matched by name and shape.
   - `verify_transfer` stops the run unless every tensor in the layers to be frozen equals the checkpoint's.
   - Torchvision arms carry their ImageNet weights inside layer 0.
2. **Warm-up.** 10 epochs with the pretrained layers frozen (`freeze = depth`, depths in §3.6), on the same configuration. The new layers (head, neck, CBAM) settle against the converged backbone. `last.pt` from this phase starts the next.
3. **Main phase.** 150 epochs with everything trainable.

The 10 warm-up epochs come **in addition to** the 150.

### 4.3 Recorded training results (synthetic validation, `best.pt`)

| Arm | Best epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Best F1 | F1-optimal conf | Train time (min) |
|---|---|---|---|---|---|---|---|---|
| YOLOv8s | 124 | 0.654 | 0.490 | 0.536 | 0.346 | 0.564 | 0.536 | 38.0 |
| **YOLOv11s** | 144 | 0.634 | 0.510 | 0.541 | 0.369 | 0.574 | 0.492 | 39.9 |
| YOLOv8s+CBAM | 111 | 0.654 | 0.485 | 0.526 | 0.346 | 0.559 | 0.494 | 46.0 |
| YOLOv11s+CBAM | 91 | 0.647 | 0.522 | 0.556 | 0.365 | 0.586 | 0.562 | 50.5 |
| ResNet18+FPN+CBAM | 145 | 0.627 | 0.520 | 0.527 | 0.328 | 0.569 | 0.393 | 52.7 |
| ResNet18+BiFPN+CBAM | 91 | 0.637 | 0.524 | 0.554 | 0.340 | 0.582 | 0.454 | 57.1 |
| GoogLeNet+FPN+CBAM | 48 | 0.618 | 0.515 | 0.525 | 0.310 | 0.565 | 0.472 | 73.0 |
| GoogLeNet+BiFPN+CBAM | 72 | 0.634 | 0.549 | 0.566 | 0.340 | 0.597 | 0.514 | 64.0 |
| EffNet-B0+FPN+CBAM | 84 | 0.671 | 0.524 | 0.558 | 0.350 | 0.596 | 0.632 | 73.3 |
| EffNet-B0+BiFPN+CBAM | 72 | 0.678 | 0.515 | 0.578 | 0.364 | 0.593 | 0.600 | 155.8 |
| YOLOv11s+BiFPN+CBAM | 106 | 0.582 | 0.527 | 0.533 | 0.342 | 0.559 | 0.523 | 50.7 |

Training times include the warm-up and total 11.7 GPU-hours on one RTX 4060 Ti (8 GB). Synthetic validation separates the arms very little: 0.310–0.369 in mAP@0.5:0.95.

---

## 5. Evaluation on real photographs (`src/06_evaluate.py` and the paper scripts)

### 5.1 Protocol

- **Test sets.**
  - 746 real organic photographs with no e-waste, used for false alarms (specificity).
  - 387 real e-waste photographs of objects never seen in training, used for detection (sensitivity).
- **Inference.** Each photograph is scored once at conf 0.01, IoU 0.7, 640 px, batch 16. Every threshold is then a filter over the stored confidences.
- **Threshold grid.** A fine grid from 0.010 to 0.800 in steps of 0.005.
- **Image-level rates.**
  - Detection rate: the fraction of e-waste photographs with at least one box at or above t.
  - False-alarm rate: the fraction of organic photographs with at least one box at or above t.
- **Decoding.** Images are decoded defensively. `exif_transpose` is tried, and a malformed EXIF block falls back to the raw orientation.

### 5.2 Operating points

| Name | How t is chosen | Role |
|---|---|---|
| Synthetic-F1 | argmax of box F1 on synthetic validation | uses no real data at all |
| Matched false alarms | the lowest t giving a false-alarm rate ≤ 5 / 10 / 15% on the 746 organic photographs | compares detectors at an equal false-alarm cost |
| **Calibrated (split-conformal)** | set on one random half (373) of the organic photographs; false alarms measured on the other half (373) | what a deployment would see; reported for YOLOv11s |
| Oracle | argmax of F1 on the test set | a ceiling only, never reported as a result |

**Split-conformal rule** (`scripts/threshold_split.py`):
- With n = 373 calibration scores and target α, let m = ⌊α(n + 1)⌋.
- Set t at the m-th largest calibration score, and alert when the score is strictly above t.
- This keeps the expected false-alarm rate on new clean photographs at or below α.
- It was repeated over **1000 random splits, seed 0**. The mean and the 2.5–97.5 percentiles are reported.

### 5.3 Statistics

- **Intervals.**
  - 95% Wilson score intervals for every rate.
  - A 95% percentile bootstrap for pAUC₁₅ (the area under the detection–false-alarm curve from 0 to 15% false alarms): 2000 resamples, the two sets resampled independently, seed 0.
- **Paired comparisons.**
  - Test: McNemar's exact test on the discordant e-waste photographs at the matched 10% point.
  - Effect size: Δ = (g − ℓ)/N, with SE = √(g + ℓ − (g − ℓ)²/N)/N.
  - Correction: Holm's step-down procedure over the **10 pre-specified ablation contrasts**, which carry the paper's claims. The 66 pairs among all 12 detectors are exploratory and carry their own Holm correction.
- **Ablation contrasts.** Attention (CBAM on v8s and on v11s), neck (FPN vs BiFPN on each torchvision backbone; BiFPN vs native PAN on v11s), backbone (the YOLOv11s backbone vs each torchvision backbone, with BiFPN and CBAM fixed), and ensembling (the ensemble vs YOLOv11s).

### 5.4 Real-image results for every model

Detection is in %. The matched rows hold every model to exactly 5, 10 and 15% false alarms. The calibrated rows are split means, with the observed false-alarm rate on the unseen half close to its target.

| Arm | Det@5% | Det@10% | Det@15% | pAUC₁₅ | Calibrated 5 / 10 / 15% | Synthetic-F1 point: t, det, FA |
|---|---|---|---|---|---|---|
| YOLOv8s | 73.4 | 82.4 | 85.3 | 0.732 | 73.1 / 81.8 / 85.2 | 0.535, 74.4, 5.8 |
| **YOLOv11s** | **77.8** | **84.8** | **87.6** | **0.763** | **75.4 / 84.5 / 87.6** | 0.490, 79.1, 5.4 |
| YOLOv8s+CBAM | 55.3 | 70.5 | 77.8 | 0.585 | 54.1 / 69.9 / 77.8 | 0.495, 52.2, 4.2 |
| YOLOv11s+CBAM | 49.4 | 68.5 | 76.7 | 0.561 | 50.1 / 69.0 / 77.1 | 0.560, 47.0, 3.6 |
| ResNet18+FPN+CBAM | 68.2 | 81.9 | 86.8 | 0.684 | 64.8 / 81.0 / 86.5 | 0.395, 78.6, 8.2 |
| ResNet18+BiFPN+CBAM | 58.9 | 73.9 | 80.4 | 0.614 | 57.8 / 73.2 / 81.0 | 0.455, 58.9, 5.1 |
| GoogLeNet+FPN+CBAM | 59.4 | 75.5 | 79.8 | 0.618 | 58.7 / 74.9 / 79.8 | 0.470, 60.5, 5.2 |
| GoogLeNet+BiFPN+CBAM | 57.4 | 69.3 | 77.5 | 0.578 | 55.0 / 68.5 / 77.6 | 0.515, 44.2, 2.9 |
| EffNet-B0+FPN+CBAM | 59.7 | 76.5 | 81.7 | 0.629 | 59.7 / 75.9 / 82.0 | 0.630, 54.5, 3.9 |
| EffNet-B0+BiFPN+CBAM | 64.9 | 76.7 | 82.7 | 0.654 | 63.3 / 76.9 / 82.6 | 0.600, 48.8, 2.8 |
| YOLOv11s+BiFPN+CBAM | 46.5 | 66.4 | 74.7 | 0.532 | 46.0 / 66.1 / 74.3 | 0.525, 34.1, 2.8 |
| Ensemble | 76.0 | 86.6 | 89.9 | 0.763 | 74.2 / 86.0 / 90.3 | 0.325, 55.8, 1.3 |

YOLOv11s in more detail:
- At the matched 10% point it flags 328 of 387 photographs (84.8%, Wilson CI 80.8–88.0%). Its pAUC₁₅ is 0.763, with a bootstrap interval of 0.715–0.808.
- With a calibrated 10% threshold it flags 84.5% (split range 82.9–86.6%). False alarms on the unseen half are 9.8% (5.9–14.7%).
- By category at 10%: chips and boards 97.2%, smartphones 83.6%, cables 74.9%. False alarms are 12.4% on food and 7.6% on vegetation.

---

## 6. Ensemble (`src/10_ensemble.py`)

**Members.** All 11 arms, each run at a confidence floor of **0.10**.

**Fusion.** Weighted box fusion (Solovyev et al. 2021), written in the project's own code:
1. Sort every box from every member by confidence.
2. Assign each box to the cluster it overlaps best, at IoU ≥ **0.55**, or start a new cluster.
3. Reduce each cluster to **one opinion per member**: that member's confidence-weighted box and its mean confidence. A member firing twice therefore does not count twice.
4. Fused box = Σ wₘ·cₘ·bₘ / Σ wₘ·cₘ.
5. Fused score = (Σ wₘ·cₘ / Σ wₘ) × (Σ wₘ in the cluster / Σ wₘ over all members). The second factor is the agreement term.

**Member weights.** Each member is weighted by its synthetic-validation best F1, divided by the mean F1 (power 1.0). The weights came out nearly uniform, from 0.969 to 1.035. No real test data were used.

**Threshold.** The fused detector's box F1 is swept on synthetic validation with IoU ≥ 0.5 greedy matching, which picked **0.325**.

**Cost.** 105.4 M parameters, 214.5 MB of weights, and a latency of 128.7 ms per image (7.8 FPS). That is 11.8 times the latency of YOLOv11s. Ensemble confidences sit on a different scale from single-model ones, so only its detection and false-alarm rates are compared.

---

## 7. Cost and latency (`src/14_latency.py`)

**Method.**
- All 11 arms are loaded into one process and timed **round-robin** for 9 rounds; round 1 is discarded.
- Each round runs 60 real photographs at batch 1 and 640 px, with conf 0.25, synchronising CUDA around each call.
- The figure is the fastest of each arm's per-round medians; the median across rounds is kept as well.
- Device: RTX 4060 Ti.
- Timing all arms together puts machine load on every arm roughly equally, instead of on whichever ran last.

| Arm | Latency (ms) | FPS |
|---|---|---|
| YOLOv8s | 10.08 | 99.2 |
| **YOLOv11s** | 10.90 | 91.7 |
| YOLOv8s+CBAM | 9.81 | 101.9 |
| YOLOv11s+CBAM | 10.31 | 97.0 |
| ResNet18+FPN+CBAM | 10.89 | 91.8 |
| ResNet18+BiFPN+CBAM | 10.75 | 93.0 |
| GoogLeNet+FPN+CBAM | 11.96 | 83.6 |
| GoogLeNet+BiFPN+CBAM | 12.72 | 78.6 |
| EffNet-B0+FPN+CBAM | 12.49 | 80.1 |
| EffNet-B0+BiFPN+CBAM | 13.96 | 71.6 |
| YOLOv11s+BiFPN+CBAM | 11.42 | 87.6 |
| Ensemble | 128.74 | 7.8 |

---

## 8. Explainability: Grad-CAM for YOLOv11s (`scripts/gradcam.py`)

**What is explained.** An alert fires when the highest e-waste score anywhere in the photograph reaches t. The target is therefore **y = the sigmoid class score of the top-scoring grid location**, taken before box decoding and NMS.

**Where.**
- Gradients of y are taken at the three neck outputs that feed Detect: layers 16, 19 and 22, at strides 8, 16 and 32.
- Each map is upsampled bilinearly to 640 px, the three are summed and the result is scaled to a maximum of 1.
- Only the level that holds the top location receives gradient. It was P4 in every example shown.

**Two maps from the same gradients.**

| Method | Formula |
|---|---|
| Grad-CAM (Selvaraju et al. 2017) | L = ReLU(Σₖ αₖ Aᵏ), where αₖ = mean over positions of ∂y/∂Aᵏ |
| HiResCAM (Draelos & Carin 2020) | L = ReLU(Σₖ ∂y/∂Aᵏ ⊙ Aᵏ), element-wise, with no averaging |

A single location's score depends on a small neighbourhood of the neck map. Averaging its gradient over all positions, as Grad-CAM does, throws away where the evidence lies.

**Localisation check.** Real test photographs have no boxes, so the check uses the 225 synthetic validation composites. 210 of them contain at least one labelled object, and those boxes cover 2.5% of the frame on average.

| Measure | Grad-CAM | HiResCAM |
|---|---|---|
| Energy inside the boxes, mean (median) | 5.2% (1.6%) | **67.2% (86.6%)** |
| Pointing game: peak inside a box | 8.6% | **73.8%** |
| The same, on the 158 composites (75.2%) whose top-scoring location lies in a box: energy / pointing | 4.3% / 6.3% | **88.2% / 97.5%** |

The energy measure is from Wang et al. (2020) and the pointing game from Zhang et al. (2018).

**Reading.**
- HiResCAM locates the evidence. Its misses happen where the detector's own strongest response was elsewhere, usually on organic clutter.
- On real photographs the evidence is:
  - for a cable, the connector head;
  - for a circuit board, a cluster of components;
  - for a phone, the end of the handset.
- Both top false alarms are attributed to **adhesive produce labels**, not to the fruit.
- Plain Grad-CAM lights frame borders and background. That is an artefact of the averaging, not detector behaviour.

**Output.** `figures/fig13_gradcam.pdf` (Fig. 10 in the paper), shown in three rows: detections, Grad-CAM and HiResCAM. The numbers are in `scripts/gradcam.json`.

---

## 9. Reproducing everything

```bash
python src/01_build_splits.py
python src/02_make_cutouts.py
python src/03_screen_cutouts.py
python src/04_build_dataset.py --pool 54
python src/05_train.py --pool 54 --model yolo11s.pt --tag yolo11s
python src/06_evaluate.py --pool 54 --tag yolo11s
python src/10_ensemble.py --pool 54
python src/14_latency.py --pool 54
python src/paper/paper_figures.py
python src/paper/threshold_split.py
python src/paper/best_model.py
python src/paper/gradcam.py
```

The `05_train.py` / `06_evaluate.py` lines above are shown for YOLOv11s. Every other arm follows the same pattern with its `--model` and `--tag` from §1 (plain YOLOv8s uses `--model yolov8s.pt` and no tag): run `05_train.py`, then `06_evaluate.py`, before `10_ensemble.py`.

**Environment.**
- Software: Python 3.13.13, PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0, Ultralytics 8.4.82, NumPy 2.4.6, SciPy 1.18.0, Pillow 12.2.0, Matplotlib 3.10.7, rembg 2.0.76, ONNX Runtime 1.27.0.
- System: Windows 11, NVIDIA GeForce RTX 4060 Ti (8 GB).
- Randomness: seed 0 throughout.

**Caveats.**
- Every arm was trained once, with one seed.
- The real e-waste photographs show intact objects on clean backgrounds. Detection of e-waste actually buried in real organic waste is not measured, because no such imagery exists.
