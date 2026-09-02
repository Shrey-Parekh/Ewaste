# Detecting electronic waste contamination in wet biodegradable waste

Code for the paper *Detecting electronic waste contamination in wet
biodegradable waste with synthetically composited training data*.

No public imagery exists of electronic waste actually buried in wet organic
waste, so the training set is built by compositing e-waste cut-outs into real
organic waste photographs, and the detectors are then scored on real
photographs they have never seen.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

Run in order. Every step is seeded, so the whole thing is reproducible.

| Step | Script | Produces |
|---|---|---|
| 1 | `src/01_build_splits.py` | `splits/*.csv` — partitions both collections into disjoint roles |
| 2 | `src/02_make_cutouts.py` | `cutouts/` — alpha-matted objects, plus `extraction_log.csv` |
| 3 | `src/03_screen_cutouts.py` | `cutouts/ewaste_clean/` — rejects matting failures and collages |
| 4 | `04_build_dataset.py --pool 60` | `dataset_pool60/` — 1500 composited images with occlusion-aware labels |
| 5 | `05_train.py --pool 60` | `runs/detect/pool60[_tag]/` |
| 6 | `06_evaluate.py --pool 60` | `eval_pool60[_tag]/` — sweep over the withheld real sets |
| 7 | `src/07_make_figures.py` | `Manuscripts/figures/` |

`src/08_verify_integrity.py` audits the whole thing and can be run at any time. It
exits non-zero if any training photograph has reached an evaluation set, so it
can gate a rebuild:

```bash
python src/08_verify_integrity.py
```

Three further steps sit outside the numbered run order:

| Script | Role |
|---|---|
| `src/09_annotate.py` | One-off manual pass: hand-draws boxes on the 400 held-out e-waste photographs, which is what makes mIoU and Dice computable |
| `src/10_ensemble.py` | Fuses the seven trained detectors with weighted box fusion and scores the result |
| `src/11_metrics_table.py` | Collects every evaluated model into one table, in CSV and LaTeX |

## The split discipline

This is the part most worth understanding before changing anything.

An earlier version of this pipeline drew its object bank from the same
photographs it later measured detection rate on, so that result was partly
measured on training material. The fix is that roles are assigned **before**
anything is extracted, and every downstream step reads the manifests rather
than the raw directories:

| Manifest | Count | Role |
|---|---|---|
| `ewaste_pool.csv` | 68 | hand-curated photographs the object pool is cut from |
| `organic_bg.csv` | 50 | backgrounds for training images |
| `organic_clutter.csv` | 50 | occluders placed over objects |
| `ewaste_test.csv` | 400 | withheld, measures detection rate |
| `organic_test.csv` | 747 | withheld, measures false alarm rate |

The five are pairwise disjoint. 52 TrashBox photographs are withheld from the
test source because they share a basename with a curated pool photograph. If
you add a step, read a manifest — never `raw/` directly.

`ewaste_test` excludes the `laptops` and `small appliances` categories: an item
that large would be removed by hand before organic waste reached a sorting
line, so it is not the contamination scenario the paper targets.

## Model comparison

Eight architectures train on the identical dataset through the identical
schedule, so only the architecture differs. That schedule lives in exactly one
place, `TRAIN_CFG` in `src/05_train.py`, and nothing is set per model. Batch size
is 16 for every arm, chosen by measuring peak VRAM across all of them: the
heaviest reaches 7.6 GiB of an 8 GiB card at batch 32, which leaves nothing for
the desktop.

Every arm also gets a warm-up phase: whatever inherited pretrained weights is
frozen while the newly initialised layers settle against it, then everything is
released. One rule for all of them, though the frozen set necessarily differs,
since only some arms have new layers to settle. It is on by default and
disabled with `--warmup-epochs 0`.

The neck configurations in `models/` are generated, not hand-written:

```bash
python src/generate_necks.py --width 128 --repeats 2
```

Width and pass count apply to the FPN and the BiFPN arms together. Those two
exist to compare fusion topology, which only works while everything else about
them matches.

```bash
python src/05_train.py    --pool 60 --model yolov8s.pt
python src/06_evaluate.py --pool 60

python src/05_train.py    --pool 60 --model yolo11s.pt --tag yolo11s
python src/06_evaluate.py --pool 60 --tag yolo11s

python src/05_train.py    --pool 60 --model models/yolov8s-cbam.yaml --tag v8s_cbam
python src/06_evaluate.py --pool 60 --tag v8s_cbam
```

...and so on for the remaining configurations in `models/`. See
`docs/HANDOFF.md` for the full command list and the tag each model must use.

One caveat belongs in any write-up of these numbers: across seeds with
everything else fixed, detection rate has spanned roughly four points, so
differences smaller than that should not be ranked.

## Layout

```
src/            every Python file: the numbered pipeline steps and the libraries
splits/         role manifests -- the source of truth for what may be used where
models/         architecture configurations for the attention and BiFPN variants
annotations/    hand-drawn boxes on the held-out e-waste photographs
raw/ewaste/     the hand-curated e-waste photographs the object pool is drawn from
cutouts/        extracted objects and occluders
backgrounds/    the 50 background photographs, materialised from the manifest
dataset_pool*/  generated training sets
runs/detect/    trained detectors
eval_pool*/     evaluation results
logs/           build logs
docs/           design records, the handoff, and archived superseded results
Manuscripts/    LaTeX source, figures, tables, highlights
```

Generated directories are gitignored: they are reproducible from `splits/` and
`models/`, which are tracked. `annotations/` is tracked as well — it is drawn
by hand and cannot be regenerated.

### Library modules

`src/lib_segment.py` and `src/lib_composite.py` hold the segmentation and compositing
algorithms. They are **not pipeline steps** — `src/02_make_cutouts.py` and
`src/04_build_dataset.py` import them by path and override their configuration
rather than duplicating the algorithms, so the two cannot drift apart. Do not
delete them.

Their module-level defaults still point at the whole of `raw/`, ignoring the
split manifests, which is why the pipeline scripts override every input path
before calling in — and why both refuse to run standalone.

`src/lib_modules.py` defines the BiFPN fusion node and registers it, along with
Ultralytics' own CBAM, with the YAML parser. Anything that builds, trains or
loads a model from `models/` must import it first, including the evaluator: a
checkpoint containing a custom module cannot be unpickled without it.

`src/lib_metrics.py` holds the capacity, cost and localisation measurements, and
`src/pipeline_common.py` the operating-point logic and the defensive image decoder
that both the evaluator and the annotator need.
