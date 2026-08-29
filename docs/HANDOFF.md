# Handoff — e-waste contamination detection

Last updated: 2026-08-30

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

---

## 2. Pipeline (current, post-rename)

Scripts run in numeric order. `lib_*.py` are libraries, not steps.

| File | Role |
|---|---|
| `01_build_splits.py` | Partitions sources into disjoint manifests in `splits/` |
| `02_make_cutouts.py` | Segments e-waste objects + organic occluders → transparent PNGs |
| `03_screen_cutouts.py` | Auto-rejects bad mattes → final object pool |
| `04_build_dataset.py` | Composites objects onto backgrounds → synthetic training set |
| `05_train.py` | Trains a YOLO detector (`--model`, `--tag`) |
| `05b_train_ednet.py` | Trains EDNet (separate venv, VisDrone-pretrained) |
| `06_evaluate.py` | Scores a detector on real photographs |
| `07_make_figures.py` | Manuscript figures |
| `08_verify_integrity.py` | Asserts no training photo leaked into evaluation |
| `lib_segment.py` | Segmentation algorithm — imported by `02`, refuses standalone run |
| `lib_composite.py` | Compositing algorithm — imported by `04`, refuses standalone run |
| `pipeline_common.py` | F1-curve / operating-point logic shared by `05` and `05b` |

`lib_segment.py` and `lib_composite.py` deliberately raise `SystemExit` if run
directly: their module-level defaults point at the whole of `raw/`, ignoring
the split manifests, which is exactly the train/test leak the manifests exist
to prevent.

### Run order

```bash
python 01_build_splits.py
python 02_make_cutouts.py
python 03_screen_cutouts.py
python 04_build_dataset.py --pool 60

python 05_train.py --pool 60 --model yolov8s.pt
python 06_evaluate.py --pool 60

python 05_train.py --pool 60 --model yolo11s.pt --tag yolo11s
python 06_evaluate.py --pool 60 --tag yolo11s

external/.venv-ednet/Scripts/python.exe 05b_train_ednet.py --pool 60
external/.venv-ednet/Scripts/python.exe 06_evaluate.py --pool 60 --tag ednets --backend ednet
```

`--pool 60` selects `dataset_pool60/`; it is **not** a free parameter, it must
match a dataset that exists on disk.

**EDNet's tag is `ednets`, not `ednet`.** Using the wrong one produces
`[!] weights not found`.

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

## 4. Results so far

All three trained on the identical `dataset_pool60` and scored on the
identical real test sets (400 e-waste / 747 organic photographs).

### Synthetic validation

| Metric | YOLOv8s | YOLOv11s | EDNet-S |
|---|---|---|---|
| Pretrained on | COCO | COCO | **VisDrone** |
| mAP@0.50 | 0.836 | **0.850** | 0.799 |
| mAP@0.50:0.95 | 0.572 | **0.590** | 0.537 |
| Precision | 0.844 | **0.881** | 0.798 |
| Recall | **0.778** | 0.742 | 0.725 |
| F1 | **0.813** | 0.809 | 0.767 |
| Train time | 35.1 min | 38.9 min | 161.1 min |

### Real-photo evaluation (the numbers that matter)

| Metric | YOLOv8s | YOLOv11s | EDNet-S |
|---|---|---|---|
| Operating confidence | 0.490 | 0.345 | 0.275 |
| Detection rate | 78.5% [74.2, 82.2] | **84.2%** [80.4, 87.5] | 82.5% [78.5, 85.9] |
| False-alarm rate | **7.0%** [5.3, 9.0] | 10.0% [8.1, 12.4] | 11.0% [8.9, 13.4] |
| F1 | 0.820 | **0.830** | 0.813 |
| Detected / 400 | 314 | 337 | 330 |
| False alarms / 747 | 52 | 75 | 82 |

Intervals are 95% Wilson score.

### How to read this honestly

- **No model dominates on both axes.** v8s has the best specificity, v11s the
  best sensitivity. Their intervals do not overlap on either metric, so that
  trade-off is real, not noise.
- **The F1 ranking is not conclusive.** 0.830 / 0.820 / 0.813 sit inside each
  model's own confidence interval. Report the trade-off, do not crown a winner
  on F1.
- **EDNet is confounded.** Its backbone is VisDrone-pretrained (aerial imagery)
  while the YOLOs are COCO-pretrained. If EDNet looks worse, part of that is
  the domain mismatch, not the architecture. This must be stated wherever
  EDNet appears in a comparison.

---

## 5. New requirements (from the mentor, not yet implemented)

### 5.1 Metrics — 12 per model

| # | Metric | Status |
|---|---|---|
| 1 | mAP@50 | Recorded |
| 2 | mAP@50:95 | Recorded |
| 3 | Precision | Recorded |
| 4 | Recall | Recorded |
| 5 | F1-score | Recorded |
| 6 | mIoU | **Not implemented** |
| 7 | Dice coefficient | **Not implemented** |
| 8 | Inference time | **Not implemented** |
| 9 | FPS | **Not implemented** (derives from 8) |
| 10 | Parameter count | **Not persisted** (Ultralytics prints it; not saved) |
| 11 | FLOPs | **Not persisted** (same) |
| 12 | Model size (MB) | **Not persisted** |
| 13 | Training time | Recorded (`train_seconds`) |

`synthetic_summary.json` currently holds only:
`pool, seed, model, epochs, dataset, precision, recall, map50, map50_95,
best_f1, best_f1_conf, train_seconds`. Items 8–12 need adding there.

> The user wrote "model sy" in the original list. Read as **model size**.
> Confirm if that was meant to be something else.

### 5.2 Models — 8 total

| # | Model | Status |
|---|---|---|
| 1 | YOLOv8s | Done |
| 2 | YOLOv11s | Done |
| 3 | YOLOv8s + CBAM | To do |
| 4 | YOLOv11s + CBAM | To do |
| 5 | ResNet + FPN + CBAM | To do — build from scratch |
| 6 | ResNet + BiFPN + CBAM | To do — build from scratch |
| 7 | YOLOv11s + BiFPN + CBAM | To do — neck replacement |
| 8 | Ensemble | To do — composition undecided |

EDNet-S is trained and evaluated but is **not** in the mentor's numbered list.
Decide whether it stays in the paper as an extra baseline or is dropped.

Models 5–7 are not configuration flags. Ultralytics ships neither BiFPN nor a
ResNet+FPN detector, so each is a genuine architecture build, debug and train
cycle. Scope accordingly.

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

---

## 7. Open questions

1. **The user's last message was cut off mid-sentence**: *"then from the
   models"*. Ask what the rest was before assuming.
2. **Annotation plan for mIoU/Dice.** The user chose to annotate real test
   photographs with boxes. Nothing has been annotated yet. 1147 photographs is
   a large manual task — scope it (all of them? a stratified subset? which
   tool?) before starting.
3. **Ensemble composition** — which of models 1–7, and fused how (e.g.
   weighted box fusion)?
4. **EDNet's place** in the final paper, given it is outside the numbered list.
5. **Whether the pool-25 diversity ablation returns.** It compared 25 vs 200
   objects from the *old uncurated* pool and is no longer valid. Drop it, or
   re-run within the curated pool.

---

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
- **EDNet lives in `external/.venv-ednet/`** with torch 2.0.1+cu118, separate
  from the main environment's 2.11.0+cu128. Run it with that interpreter.
- **One TrashBox photograph has a malformed EXIF block**
  (`electronic chips/electronic_chip 485.jpg`). `06_evaluate.py` decodes
  defensively because of it; do not "simplify" that back to passing paths
  straight to Ultralytics.
- **`06_evaluate.py` searches a fine confidence grid** (step 0.005) for the
  operating point and reports Wilson intervals. The coarse 9-point table is
  kept only for readability. Re-running an old evaluation with the current code
  changes the reported operating point — this is expected and correct.

---

## 10. Housekeeping

- **`runs/detect/pool60_ednets2/`** is a failed EDNet run — no weights, no
  `results.csv`. Safe to delete; left in place pending confirmation.
- **`runs/detect/val/`, `val-2/`** are bare Ultralytics validation dumps.
- **`dataset_pool61/`** is superseded by `dataset_pool60/` and unused.
- **`notebook40dabb2c62.ipynb`** is the executed Kaggle notebook, kept for its
  recorded output. `kaggle_train_yolov8s.ipynb` is the clean source.
- **`dataset_pool60_kaggle.zip`** (223 MB) is the Kaggle upload artefact.

### Training on Kaggle

v8s was trained once on Kaggle (Tesla T4) and once locally; the local run is
what the results above use. The notebook copies the dataset into
`/kaggle/working` because `/kaggle/input` is read-only and Ultralytics needs to
write label caches. Use **Save Version → Save & Run All (Commit)** for
unattended runs, not the interactive Run All.
