# Handoff — e-waste contamination detection

Last updated: 2026-08-30 (multi-architecture round)

Read this first in a new session. It records what exists, what the numbers
are, what was decided and why, and what is still open. Everything here was
verified against the repository, not recalled.

---

## 1. What the project is

A research paper for Elsevier *Waste Management*: detecting electronic waste
contamination inside wet biodegradable (organic) waste.

No dataset exists of e-waste buried in organic waste, so the training data is
**built synthetically** — real e-waste objects are segmented out of
photographs and composited into real organic-waste photographs, partly buried
under organic clutter. Detectors are then trained on that synthetic set and
evaluated on **real, unedited photographs**.

### Author instructions that govern the writing
- Research paper, **not** a review paper.
- No hallucinations, no assumptions, no unnecessary information.
- Humanised prose: no AI phrasing, no AI punctuation habits, formal but natural.
- Ask rather than guess when something is unclear.
- Present metrics clearly and readably.

### Implementation instructions
- Simplest correct implementation; no over-engineering, no premature abstraction.
- Reuse existing project utilities instead of duplicating them.
- Comments only where logic is non-obvious — never line-by-line narration.
- No leftover debug code, temp files, unused imports, or partial changes.
- Commit accurately, with a message reflecting what was actually done.

### Journal requirements — Elsevier *Waste Management* (ISSN 0956-053X)
- `\documentclass[final,5p,times,twocolumn,authoryear]{elsarticle}` —
  **author–date** citations, not the numbered style.
- Abstract **≤250 words**, no references, no uncommon abbreviations.
  Currently 247.
- Highlights: **3–5 bullets, each ≤85 characters including spaces**.
  `Manuscripts/highlights.tex` carries per-bullet counts as comments; they were
  checked programmatically and must be re-checked after any edit.
- Research data: Option C, which requires a deposited DOI. The old deposit was
  deleted; a new one is needed and the DOI placeholder is still unfilled.
- AI-use declaration: keep it generic. State that AI assisted with drafting the
  manuscript and with debugging the pipeline. **Do not name the tool.**

### Corrections already received — do not repeat these
1. **The realism corrections are not decoration.** Shadow, colour
   harmonisation, blur and grain matching exist to make the object *harder* to
   find, as it would be in reality. Composited without them, an object carries
   a photometric seam and a detector can score well by finding the seam rather
   than the object. Never describe them as cosmetic.
2. **Cut-outs are composited into real organic waste**, partly buried under
   organic clutter — not pasted onto clean or plain backgrounds.
3. **When fixing an inconsistency, soften it — do not delete the section.**
   Removing whole sections was an over-correction; the proportionate response
   is to qualify the claim.
4. **Do not shade or downplay a limitation to make it read better.** State it
   plainly and briefly.

### Conventions to hold to
- **Statistics.** Wilson score intervals for proportions (better than the
  normal approximation near 0 and 1). McNemar's exact test for a *paired*
  comparison of two detectors on the same images.
- **Noise floor.** Across seeds with everything else fixed, detection spanned
  roughly 4 points. Treat any difference under ~4 points at one seed per model
  as unresolvable, and say so rather than ranking.
- **Primary metrics are detection rate and false-alarm rate.** Precision and
  F1 additionally depend on the 400:747 positive-to-negative ratio, which is an
  artefact of how the split was drawn, not a real-world prevalence.
- **Figure colours.** Green = ground truth, red = false positive, the detection
  literature convention — kept deliberately in the photo montages.
- **Palette** (accessibility-checked, 3.38:1 WCAG, 2.63:1 protanopia):
  `verdigris #1F3B33`, `verdifill #DCE8E2`, `clay #B07C52`,
  `clayfill #F2E2D2`, `inkslate #2B3A34`. A previous palette failed contrast
  at 1.42:1 and had to be replaced — re-check before changing these.
- **Figure filenames match their rendered figure numbers.** Keep that.

### How to work on this safely
- **Verify before claiming.** Run the check, read the output, then report. Do
  not state a metric, a file's contents or a fix's success from recollection.
- **Back up before any destructive rewrite.** A 1240-line manuscript was once
  rewritten across four scripts with no `.bak` and had to be reconstructed
  rather than restored. `Manuscripts/` is now tracked, so commit first.
- **Read a file before overwriting it.**
- **Do not pipe a backgrounded long-running command.** See §9.
- **Check the GPU is free before launching training.** See §9.
- **Say when something failed.** If a run crashes, a metric is unavailable or a
  step was skipped, report it with the output rather than working around it
  silently.

---

## 2. Pipeline (current, post-rename)

Scripts run in numeric order. `lib_*.py` are libraries, not steps.

| File | Role |
|---|---|
| `src/01_build_splits.py` | Partitions sources into disjoint manifests in `splits/` |
| `src/02_make_cutouts.py` | Segments e-waste objects + organic occluders → transparent PNGs |
| `src/03_screen_cutouts.py` | Auto-rejects bad mattes → final object pool |
| `src/04_build_dataset.py` | Composites objects onto backgrounds → synthetic training set |
| `src/05_train.py` | Trains a YOLO detector (`--model`, `--tag`) |
| `src/06_evaluate.py` | Scores a detector on real photographs |
| `src/07_make_figures.py` | Manuscript figures |
| `src/08_verify_integrity.py` | Asserts no training photo leaked into evaluation |
| `src/lib_segment.py` | Segmentation algorithm — imported by `02`, refuses standalone run |
| `src/lib_composite.py` | Compositing algorithm — imported by `04`, refuses standalone run |
| `src/09_annotate.py` | One-off manual pass: hand-draws boxes on the 400 e-waste test photographs |
| `src/10_ensemble.py` | Weighted box fusion over the seven detectors, scored like `06` |
| `src/11_metrics_table.py` | Collects every evaluated model into one CSV + LaTeX table |
| `src/lib_modules.py` | BiFPN node + CBAM/BiFPN registration with the YAML parser |
| `src/lib_metrics.py` | Capacity, cost and localisation metrics |
| `src/pipeline_common.py` | Operating-point logic and the defensive image decoder |

`src/lib_segment.py` and `src/lib_composite.py` deliberately raise `SystemExit` if run
directly: their module-level defaults point at the whole of `raw/`, ignoring
the split manifests, which is exactly the train/test leak the manifests exist
to prevent.

### Run order

```bash
python src/01_build_splits.py
python src/02_make_cutouts.py
python src/03_screen_cutouts.py
python src/04_build_dataset.py --pool 60

python src/05_train.py --pool 60 --model yolov8s.pt
python src/06_evaluate.py --pool 60

python src/05_train.py --pool 60 --model yolo11s.pt --tag yolo11s
python src/06_evaluate.py --pool 60 --tag yolo11s
```

`--pool 60` selects `dataset_pool60/`; it is **not** a free parameter, it must
match a dataset that exists on disk.


---

## 3. Data

| Path | Contents |
|---|---|
| `raw/ewaste/` | 68 hand-curated e-waste photographs — the object pool source |
| `raw/organic/` | 20 curated organic photographs — currently unused |
| `raw/allorganics/` | 847 general organic photographs |
| `assets/` | TrashBox + RealWaste collections, 7.7 GB, source of the test sets |
| `splits/` | The manifests. Source of truth for what may be used where |
| `cutouts/ewaste_clean/` | **60** screened cut-outs — the actual object pool |
| `dataset_pool60/` | 1500 synthetic images (1275 train / 225 val), 2374 instances |

### Splits

| Manifest | n | Role |
|---|---|---|
| `ewaste_pool` | 68 | Curated photographs → cut-outs → training |
| `ewaste_test` | 400 | Held-out real photographs, detection rate |
| `organic_bg` | 50 | Compositing backgrounds |
| `organic_clutter` | 50 | Occluders |
| `organic_test` | 747 | Held-out real photographs, false-alarm rate |

Verified pairwise disjoint. 52 TrashBox photographs are withheld from the test
source because they share a basename with a curated pool photograph — without
that exclusion the training pool would leak into the test set.

`ewaste_test` **excludes the `laptops` and `small appliances` categories
entirely**: an object that large would be removed by hand before organic waste
reached a sorting line, so it is not the contamination scenario the paper
targets. Remaining test composition: 180 electrical cables, 153 electronic
chips, 67 smartphones.

---

## 4. Results

**There are currently no results.** Every run was deleted on 2026-08-30 so that
all eight architectures could be retrained under one uniform schedule, which is
what makes the comparison fair. Nothing may be quoted from memory.

The superseded figures (YOLOv8s 78.5%/7.0%, YOLOv11s 84.2%/10.0%) are archived at `docs/archive/2026-08-30-pool60-v1/`. They came
from a different schedule and are **not** comparable with what the current code
produces.

### How to read the new numbers when they arrive

- **Expect no model to dominate on both axes.** Report the
  sensitivity/specificity trade-off; do not crown a winner on F1.
- **Differences under ~4 points are inside seed noise** at one seed per model.
- **The ensemble costs roughly seven times the inference of one model.** Its
  FPS column will look poor; report it plainly rather than omitting it.

## 5. New requirements (from the mentor, not yet implemented)

### 5.1 Metrics — 12 per model

| # | Metric | Status |
|---|---|---|
| 1 | mAP@50 | Recorded |
| 2 | mAP@50:95 | Recorded |
| 3 | Precision | Recorded |
| 4 | Recall | Recorded |
| 5 | F1-score | Recorded |
| 6 | mIoU | Implemented; needs `src/09_annotate.py` run first |
| 7 | Dice coefficient | Implemented; same dependency |
| 8 | Inference time | Implemented (`src/06_evaluate.py`, batch 1, real photographs) |
| 9 | FPS | Implemented (derives from 8) |
| 10 | Parameter count | Persisted |
| 11 | FLOPs | Persisted; unavailable where thop is not installed |
| 12 | Model size (MB) | Persisted |
| 13 | Training time | Persisted (`train_seconds`) |

For an axis-aligned box, Dice is exactly `2*IoU/(1+IoU)`. It ranks models
identically to mIoU and carries no independent evidence. It is reported because
it was asked for, and that relationship is stated wherever it appears.

`synthetic_summary.json` now also carries `n_params`, `gflops`,
`model_size_mb`, `batch` and `pretrained_from`. Latency and FPS are measured on
real photographs by `src/06_evaluate.py` into `summary.json` under `capacity`;
mIoU and Dice land under `localisation`.

### 5.2 Models — 8 total

| # | Model | Status |
|---|---|---|
| 1 | YOLOv8s | Done |
| 2 | YOLOv11s | Done |
| 3 | YOLOv8s + CBAM | Built, `models/yolov8s-cbam.yaml`, tag `v8s_cbam` |
| 4 | YOLOv11s + CBAM | Built, `models/yolo11s-cbam.yaml`, tag `v11s_cbam` |
| 5a | ResNet18 + FPN + CBAM | `models/resnet18-fpn-cbam.yaml`, tag `r18_fpn_cbam` |
| 5b | GoogLeNet + FPN + CBAM | `models/googlenet-fpn-cbam.yaml`, tag `gnet_fpn_cbam` |
| 5c | EfficientNet + FPN + CBAM | `models/efficientnet-fpn-cbam.yaml`, tag `effnet_fpn_cbam` |
| 6a | ResNet18 + BiFPN + CBAM | `models/resnet18-bifpn-cbam.yaml`, tag `r18_bifpn_cbam` |
| 6b | GoogLeNet + BiFPN + CBAM | `models/googlenet-bifpn-cbam.yaml`, tag `gnet_bifpn_cbam` |
| 6c | EfficientNet + BiFPN + CBAM | `models/efficientnet-bifpn-cbam.yaml`, tag `effnet_bifpn_cbam` |
| 7 | YOLOv11s + BiFPN + CBAM | Built, `models/yolo11s-bifpn-cbam.yaml`, tag `v11s_bifpn_cbam` |
| 8 | Ensemble | Built, `src/10_ensemble.py`, WBF over models 1-7 |

All five configurations were verified to build, forward-pass at stride
[8, 16, 32], and load their pretrained weights. Models 1 and 2 are unchanged
and still train from their stock checkpoints, so the baselines re-run without
touching any new file.


Models 5 and 6 turned out **not** to need a from-scratch detector, contrary to
the earlier estimate in this file. Ultralytics ships a `TorchVision` wrapper
whose `split=True` mode returns every child module's output, so `Index` can take
P3/P4/P5 straight out of a torchvision ResNet and the rest is an ordinary YAML.
BiFPN genuinely is absent from Ultralytics and is implemented in
`src/lib_modules.py` as a fast normalised weighted fusion node.

### 5.3 Explainable AI

Grad-CAM on whichever of models 1–7 performs best. Chosen over SHAP: SHAP has
no mature library for a detector's joint box + class output, and would be a
research effort rather than an add-on.

---

## 6. Decisions locked in

| Decision | Choice | Why |
|---|---|---|
| Object pool | Hand-curated `raw/ewaste/`, not bulk TrashBox | Automated screening cannot judge whether a segmented object still reads as e-waste |
| Backgrounds / occluders | General collection, **not** curated | Never segmented, so matting quality is irrelevant; their diversity is what holds the false-alarm rate down, and the curated 20 are all Vegetation while half the FP test set is Food Organics |
| Test set | Excludes laptops + small appliances | Items that large are removed by hand before sorting |
| Attention module | **CBAM** | Most cited in YOLO+attention work, easiest to defend |
| mIoU / Dice scope | **Annotate real test photographs with boxes** | User's explicit choice over synthetic-only |
| XAI | **Grad-CAM** | Mature tooling for CNN detectors |
| Primary metrics | Detection rate + false-alarm rate | Precision and F1 additionally depend on the arbitrary 400:747 positive-to-negative ratio |
| Backbones | **ResNet18, GoogLeNet, EfficientNet-B0** | Mentor's request. Each is paired with both necks, so FPN against BiFPN is asked three times rather than once |
| Inception variant | **GoogLeNet (Inception v1)** | Inception v3 has unpadded stem convolutions and yields 77/38/18 pixel maps at 640 px, so its P5 upsamples to 36 against a P4 of 38 and the neck cannot concatenate at all. GoogLeNet gives exactly 8/16/32 |
| EfficientNet P5 | **320-channel block output** | Not the 1280-channel projection after it, which exists to feed a classifier. EfficientDet takes the block outputs |
| Neck width | **128 for FPN and BiFPN alike** | A different width would confound neck topology with neck capacity |
| Batch size | **16, every arm** | Measured: the heaviest model peaks at 7.6 GiB of an 8 GiB card at batch 32. Ultralytics accumulates to a nominal batch of 64, so 16 and 32 optimise identically |
| CBAM placement | **After the last neck layer, before Detect** | Preserves every pre-existing layer index, so the COCO checkpoint still transfers into the whole backbone and neck |

---

## 7. Open questions

1. **Nothing is trained yet.** Eight runs plus the ensemble are queued; see
   §11 for the commands.
2. **The 400 photographs are not annotated yet.** `src/09_annotate.py` exists and
   resumes where it stopped; until it has been run, mIoU and Dice are the only
   two metrics the table cannot fill.

## 8. Manuscript status

`Manuscripts/manuscript_v3.tex` is **stale**. It was written against the old
uncurated 200-object pool and reports 88.8% detection / 6.2% false alarms —
numbers that no longer exist. Needs:

- Methods rewritten for the curated pool and the category exclusion.
- All results, tables and figures rebuilt from the current numbers.
- Abstract, highlights and framing shifted to a multi-architecture comparison.
- The pool-25 ablation dropped or re-run.
- A new data deposit — the old `zenodo_deposit/` and its zip were deleted
  along with the abandoned pipeline, and the DOI placeholder was never filled.

---

## 9. Gotchas that have already cost time

- **`expandable_segments:True` is silently unsupported** on this PyTorch build.
  It is not protecting against OOM. A previous run hit CUDA OOM at epoch 90
  purely from desktop applications competing for VRAM.
- **Do not pipe a backgrounded long-running command** (`nohup ... | grep`). It
  previously left an orphaned trainer alive; a second "restart" then wrote into
  the same directory and froze `results.csv`.
- **Training two models at once will not fit.** 8 GB card, ~4.8 GB per run.
  Evaluation alongside training is fine (inference only, no optimizer state).

---

## 10. Housekeeping

- **All run and evaluation directories were deleted on 2026-08-30**, along with
  the superseded `dataset_pool61/`. That freed 390 MB. `dataset_pool60/` is
  intact and is what every queued run trains on.
- **`docs/archive/2026-08-30-pool60-v1/`** holds the summaries of the deleted
  runs, and nothing else from them.
- **`notebook40dabb2c62.ipynb`** is the executed Kaggle notebook, kept for its
  recorded output. `kaggle_train_yolov8s.ipynb` is the clean source.
- **`dataset_pool60_kaggle.zip`** (223 MB) is the Kaggle upload artefact.
- `models/` and `annotations/` are tracked. `annotations/` is drawn by hand and
  cannot be regenerated, so it must never be added to `.gitignore`.

### Training on Kaggle

v8s was trained once on Kaggle (Tesla T4) and once locally; the local run is
what the results above use. The notebook copies the dataset into
`/kaggle/working` because `/kaggle/input` is read-only and Ultralytics needs to
write label caches. Use **Save Version → Save & Run All (Commit)** for
unattended runs, not the interactive Run All.

---

## 11. Architecture fixes applied 2026-08-31

Four changes, each its own commit. **Every model needs retraining**: the first
three alter the networks or how they are optimised, so no existing result is
comparable with anything produced after them.

1. **CBAM gates now start open.** Stock CBAM initialises both gate convolutions
   the default way, which passes only 28.7% of the feature magnitude through a
   non-uniform random mask. Appended to a pretrained neck, that corrupts a
   converged representation on the first forward pass, and it is what cost
   YOLOv8s 5.8 points of detection rate. Zeroing the gates and opening them with
   a bias of 2.0 makes the block a uniform pass-through at initialisation
   (verified: gain standard deviation 0.0000, correlation 1.0000 with the
   input) while the gates keep their gradient.

2. **Warm-up phase.** Pretrained layers are frozen for 10 epochs while the new
   ones settle, then released. Verified: all 270 pretrained tensors came back
   bit-identical from a warm-up run. The epochs are additional to the 120, not
   deducted, so a gain from warming up is not confounded with ten fewer full
   epochs.

3. **Necks are generated, with two passes instead of one.** `models/
   generate_necks.py` emits the three neck configurations from one width and
   one pass count. BiFPN had a single block where EfficientDet uses three or
   more. Both necks take the same count, because giving BiFPN more depth than
   FPN would confound the topology those two arms exist to compare.

4. **The `--workers` documentation was wrong** and is corrected. Worker count
   changes the augmentation stream, because each worker seeds its own RNG. It
   is not reproducibility-neutral.

### Still open from the diagnostic, not yet done

- **The operating threshold is tuned on the test set.** `src/06_evaluate.py` picks
  the confidence maximising F1 on the test set and reports there. Measured
  optimism is small for most arms but +0.070 F1 for YOLOv11s+BiFPN, whose
  honest detection rate at the synthetic threshold is 56.2%, not 77.5%.
  `real_at_synthetic_conf` is already computed in every summary; this is a
  reporting change, not a re-run.
- Single seed per model; no paired McNemar test; cable-skewed localisation
  ground truth; undiagnosed low box match rate. See sections above.

---

## 11. The queued run

Train one model at a time; two will not fit on the card. Evaluation is
inference-only, so it can overlap the next training if you want to.

Tags are not cosmetic. `src/10_ensemble.py` and `src/11_metrics_table.py` locate run
directories by them, so they must match exactly as written.

### Baselines (unchanged architectures)

```bash
python src/05_train.py    --pool 60 --model yolov8s.pt
python src/06_evaluate.py --pool 60

python src/05_train.py    --pool 60 --model yolo11s.pt --tag yolo11s
python src/06_evaluate.py --pool 60 --tag yolo11s
```

### Attention variants

```bash
python src/05_train.py    --pool 60 --model models/yolov8s-cbam.yaml --tag v8s_cbam
python src/06_evaluate.py --pool 60 --tag v8s_cbam

python src/05_train.py    --pool 60 --model models/yolo11s-cbam.yaml --tag v11s_cbam
python src/06_evaluate.py --pool 60 --tag v11s_cbam
```

### Backbones and necks

Three backbones, each paired with both necks. Everything about a pair is held
equal except the fusion topology, so the six runs answer the FPN-against-BiFPN
question three times over.

```bash
python src/05_train.py    --pool 60 --model models/resnet18-fpn-cbam.yaml --tag r18_fpn_cbam
python src/06_evaluate.py --pool 60 --tag r18_fpn_cbam

python src/05_train.py    --pool 60 --model models/resnet18-bifpn-cbam.yaml --tag r18_bifpn_cbam
python src/06_evaluate.py --pool 60 --tag r18_bifpn_cbam

python src/05_train.py    --pool 60 --model models/googlenet-fpn-cbam.yaml --tag gnet_fpn_cbam
python src/06_evaluate.py --pool 60 --tag gnet_fpn_cbam

python src/05_train.py    --pool 60 --model models/googlenet-bifpn-cbam.yaml --tag gnet_bifpn_cbam
python src/06_evaluate.py --pool 60 --tag gnet_bifpn_cbam

python src/05_train.py    --pool 60 --model models/efficientnet-fpn-cbam.yaml --tag effnet_fpn_cbam
python src/06_evaluate.py --pool 60 --tag effnet_fpn_cbam

python src/05_train.py    --pool 60 --model models/efficientnet-bifpn-cbam.yaml --tag effnet_bifpn_cbam
python src/06_evaluate.py --pool 60 --tag effnet_bifpn_cbam

python src/05_train.py    --pool 60 --model models/yolo11s-bifpn-cbam.yaml --tag v11s_bifpn_cbam
python src/06_evaluate.py --pool 60 --tag v11s_bifpn_cbam
```

### Ensemble and the metrics table

Both need all seven single models present first.

```bash
python src/10_ensemble.py --pool 60
python src/11_metrics_table.py --pool 60
```

`src/11_metrics_table.py` can be run at any point; it prints what exists and names
what is still missing.

### Annotation

Independent of training, and resumable — it opens at the first photograph
without a label file.

```bash
python src/09_annotate.py
python src/09_annotate.py --check
```

Until this has been run, mIoU and Dice are the only cells in the table that
cannot be filled.
