# Detection of electronic waste contamination in wet biodegradable waste using segmentation and explainable AI

Code for the paper *Detection of electronic waste contamination in wet biodegradable waste using segmentation and explainable AI* (submitted to *Waste Management*).

No public imagery exists of electronic waste actually buried in wet organic
waste, so the training set is built by compositing e-waste cut-outs into real
organic waste photographs, and the detectors are then scored on real
photographs they have never seen.

## Setup

Tested with Python 3.13 on Windows 11, CUDA 12.8 and one NVIDIA RTX 4060 Ti
(8 GB). Install the CUDA builds of PyTorch first, then the rest:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data

The source photographs are not redistributed with this code. Download both
collections and place them where the split manifests expect them:

| Collection | Licence | Expected location |
|---|---|---|
| [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) | CC BY 4.0 | `assets/trash/realwaste-main/RealWaste/` |
| [TrashBox](https://github.com/nikhilvenkatkumsetty/TrashBox) | none stated | `assets/trash/TrashBox/TrashBox_train_dataset_subfolders/` |

The 68 hand-curated training photographs in `raw/ewaste/` were copied out of
TrashBox; `splits/ewaste_pool.csv` lists them. Every manifest in `splits/`
records the relative path of each photograph and its role, so the partition
used in the paper is reproduced exactly once both collections are in place.
Because TrashBox carries no licence, the cut-outs and synthetic images derived
from it are not shared either; steps 2 to 4 regenerate them from the
manifests with the same seeds.

## Pipeline

Run in order. Every step is seeded, so the whole thing is reproducible.

| Step | Script | Produces |
|---|---|---|
| 1 | `src/01_build_splits.py` | `splits/*.csv` — partitions both collections into disjoint roles |
| 2 | `src/02_make_cutouts.py` | `cutouts/` — alpha-matted objects, plus `extraction_log.csv` |
| 3 | `src/03_screen_cutouts.py` | `cutouts/ewaste_clean/` — rejects matting failures and collages |
| 4 | `src/04_build_dataset.py --pool 54` | `dataset_pool54/` — 1500 composited images with occlusion-aware labels |
| 5 | `src/05_train.py --pool 54` | `runs/detect/pool54[_tag]/` |
| 6 | `src/06_evaluate.py --pool 54` | `evaluation/pool54[_tag]/` — sweep over the withheld real sets |
| 7 | `src/paper/*.py` | figures, table rows and statistics for the paper (see below) |

`src/08_verify_integrity.py` audits the whole thing and can be run at any time. It
exits non-zero if any training photograph has reached an evaluation set, so it
can gate a rebuild:

```bash
python src/08_verify_integrity.py
```

Further steps sit outside the numbered run order:

| Script | Role |
|---|---|
| `src/10_ensemble.py` | Fuses the eleven trained detectors with weighted box fusion and scores the result |
| `src/11_metrics_table.py` | Collects every evaluated model into one table, in CSV and LaTeX (`results/`) |
| `src/12_gui.py` | Launcher: pick models, queue training and evaluation |
| `src/13_excel.py` | Writes the same table as a formatted workbook with charts (`results/`) |
| `src/14_latency.py` | Times every arm in one interleaved sitting, so latency is comparable |

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

The five are pairwise disjoint. Repeated pictures inside the test draws, found by
perceptual hashing, are scored once, which leaves the 387 e-waste and 746
organic photographs reported in the paper. 52 TrashBox photographs are withheld from the
test source because they share a basename with a curated pool photograph. If
you add a step, read a manifest — never `raw/` directly.

`ewaste_test` excludes the `laptops` and `small appliances` categories: an item
that large would be removed by hand before organic waste reached a sorting
line, so it is not the contamination scenario the paper targets.

## Model comparison

Eleven architectures train on the identical dataset through the identical
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
python src/05_train.py    --pool 54 --model yolov8s.pt
python src/06_evaluate.py --pool 54

python src/05_train.py    --pool 54 --model yolo11s.pt --tag yolo11s
python src/06_evaluate.py --pool 54 --tag yolo11s

python src/05_train.py    --pool 54 --model models/yolov8s-cbam.yaml --tag v8s_cbam
python src/06_evaluate.py --pool 54 --tag v8s_cbam
```

...and so on for the remaining configurations in `models/`. Three backbones,
ResNet18, GoogLeNet and EfficientNet-B0, are each paired with both necks. Each
configuration must use this tag, because the evaluation and paper scripts find
runs by it:

| Model config | `--tag` |
|---|---|
| `yolov8s.pt` | *(none)* |
| `yolo11s.pt` | `yolo11s` |
| `models/yolov8s-cbam.yaml` | `v8s_cbam` |
| `models/yolo11s-cbam.yaml` | `v11s_cbam` |
| `models/yolo11s-bifpn-cbam.yaml` | `v11s_bifpn_cbam` |
| `models/resnet18-fpn-cbam.yaml` | `r18_fpn_cbam` |
| `models/resnet18-bifpn-cbam.yaml` | `r18_bifpn_cbam` |
| `models/googlenet-fpn-cbam.yaml` | `gnet_fpn_cbam` |
| `models/googlenet-bifpn-cbam.yaml` | `gnet_bifpn_cbam` |
| `models/efficientnet-fpn-cbam.yaml` | `effnet_fpn_cbam` |
| `models/efficientnet-bifpn-cbam.yaml` | `effnet_bifpn_cbam` |

Then fuse the eleven and time them:

```bash
python src/10_ensemble.py --pool 54
python src/14_latency.py  --pool 54
```

## Paper figures and tables

`src/paper/` turns the evaluation outputs into everything the manuscript
shows. Run from the project root, in this order:

```bash
python src/paper/paper_figures.py       # statistics, figures, table rows; writes stats.json
python src/paper/threshold_split.py     # split-conformal calibration; writes threshold_split.json
python src/paper/composition_stages.py  # the stage-by-stage compositing figure
python src/paper/best_model.py          # example detections of YOLOv11s
python src/paper/gradcam.py             # HiResCAM maps and localisation check (GPU)
```

Figures go to `Manuscripts/latex/figures/` and table rows to
`Manuscripts/latex/tables/`; the JSON files next to the scripts hold the
numbers quoted in the text.

One caveat belongs in any write-up of these numbers: across seeds with
everything else fixed, detection rate has spanned roughly four points, so
differences smaller than that should not be ranked.

## Layout

```
src/            the numbered pipeline steps and the libraries
src/paper/      statistics, figures and table rows for the manuscript
models/         architecture configurations, generated by src/generate_necks.py
splits/         role manifests, the source of truth for what may be used where
docs/           design notes and the full implementation report
results/        summary tables across all models (CSV, LaTeX, Excel)
latency_pool54.json  interleaved latency measurements

raw/            source photographs (not redistributed)
assets/         TrashBox and RealWaste downloads (not redistributed)
cutouts/        extracted objects and occluders
backgrounds/    the 50 background photographs, materialised from the manifest
dataset_pool54/ generated training set
runs/detect/    trained detectors
evaluation/     evaluation results, one folder per detector
Manuscripts/    LaTeX source (latex/) and compiled PDFs (pdf/); not tracked
```

Generated directories are gitignored: they are reproducible from `splits/` and
`models/`, which are tracked. `Manuscripts/` is deliberately not tracked:
the paper lives with its author, this repository holds the code that produced
the numbers in it.

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

`src/lib_metrics.py` holds the capacity and cost measurements, and
`src/pipeline_common.py` the operating-point logic and the defensive image decoder
the evaluator needs.
