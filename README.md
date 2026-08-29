# Detecting electronic waste contamination in wet biodegradable waste

Code for the paper *Detecting electronic waste contamination in wet
biodegradable waste with synthetically composited training data*.

No public imagery exists of electronic waste actually buried in wet organic
waste, so the training set is built by compositing e-waste cut-outs into real
organic waste photographs, and the detector is then scored on real photographs
it has never seen.

## Setup

```bash
pip install -r requirements.txt
```

EDNet is the exception: it pins `torch==2.0.1` and `numpy<2.0`, which are
incompatible with the environment above. It gets its own virtual environment:

```bash
bash external/setup_ednet.sh
```

## Pipeline

Run in order. Every step is seeded, so the whole thing is reproducible.

| Step | Script | Produces |
|---|---|---|
| 1 | `01_build_splits.py` | `splits/*.csv` — partitions both collections into disjoint roles |
| 2 | `02_make_cutouts.py` | `cutouts/` — alpha-matted objects, plus `extraction_log.csv` |
| 3 | `03_screen_cutouts.py` | `cutouts/ewaste_clean/` — rejects matting failures and collages |
| 4 | `04_build_dataset.py --pool 200` | `dataset_pool200/` — 1500 composited images with occlusion-aware labels |
| 5 | `05_train.py --pool 200` | `runs/detect/pool200/` |
| 6 | `06_evaluate.py --pool 200` | `eval_pool200/` — sweep over the withheld real sets |
| 7 | `07_make_figures.py` | `Manuscripts/figures/` |

`08_verify_integrity.py` audits the whole thing and can be run at any time. It
exits non-zero if any training photograph has reached an evaluation set, so it
can gate a rebuild:

```bash
python 08_verify_integrity.py
```

## The split discipline

This is the part most worth understanding before changing anything.

An earlier version of this pipeline drew its object bank from the same
photographs it later measured detection rate on, so that result was partly
measured on training material. The fix is that roles are assigned **before**
anything is extracted, and every downstream step reads the manifests rather
than the raw directories:

| Manifest | Count | Role |
|---|---|---|
| `ewaste_pool.csv` | 350 | objects composited into training images |
| `organic_bg.csv` | 50 | backgrounds for training images |
| `organic_clutter.csv` | 50 | occluders placed over objects |
| `ewaste_test.csv` | 400 | withheld, measures detection rate |
| `organic_test.csv` | 747 | withheld, measures false alarm rate |

The five are pairwise disjoint. If you add a step, read a manifest — never
`raw/` directly.

## Model comparison

All arms train on the identical dataset with identical augmentation, so only
the architecture differs.

```bash
python 05_train.py    --pool 200 --model yolo11s.pt --tag yolo11s
python 06_evaluate.py --pool 200 --tag yolo11s

external/.venv-ednet/Scripts/python.exe 05b_train_ednet.py --pool 200
external/.venv-ednet/Scripts/python.exe 06_evaluate.py --pool 200 --tag ednets --backend ednet
```

One caveat belongs in any write-up of these numbers: the Ultralytics models
start from COCO weights, whereas the published EDNet checkpoints are trained
on VisDrone. A difference between them confounds architecture with pretraining
corpus.

A second caveat: across seeds 0-2 with everything else fixed, detection rate
spanned 85.2-89.8% and false alarms 5.9-8.8%. Differences smaller than roughly
four points are inside that noise and should not be ranked.

## Layout

```
splits/         role manifests -- the source of truth for what may be used where
raw/ewaste/     the hand-curated e-waste photographs the object pool is drawn from
cutouts/        extracted objects and occluders
backgrounds/    the 50 background photographs, materialised from the manifest
dataset_pool*/  generated training sets
runs/detect/    trained detectors
eval_pool*/     evaluation results
logs/           build logs
docs/           design records
Manuscripts/    LaTeX source, figures, highlights
external/       EDNet clone and its isolated virtual environment
```

Generated directories are gitignored: they are reproducible from `splits/`,
which is tracked.

### Library modules

`lib_segment.py` and `lib_composite.py` hold the segmentation and compositing
algorithms. They are **not pipeline steps** — `02_make_cutouts.py` and
`04_build_dataset.py` import them by path and override their configuration
rather than duplicating the algorithms, so the two cannot drift apart. Do not
delete them.

Their module-level defaults still point at the whole of `raw/`, ignoring the
split manifests, which is why the pipeline scripts override every input path
before calling in — and why both refuse to run standalone.
