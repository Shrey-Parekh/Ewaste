# Curated e-waste object pool

Date: 2026-08-26
Status: approved, implementation pending additional curated photographs

## Problem

The v3 pipeline drew its e-waste training pool by globbing the TrashBox
collection: `01_build_splits.py` listed every category directory under
`assets/trash/TrashBox/.../e-waste`, stratified-sampled 350 candidates,
extracted cut-outs from all of them, and kept the 200 that passed automated
screening.

Nothing in that path consulted human judgement about whether a given
photograph yields a cut-out that still reads as electronic waste. TrashBox is
web-scraped, so the pool accumulated matting failures, product collages and
images whose subject is ambiguous once segmented. The automated screen in
`03_screen_cutouts.py` rejects the grossest failures — solidity, haze,
person detection — but it cannot judge whether a successfully segmented
object is *recognisably* e-waste.

A hand-curated set of 46 photographs exists at `raw/ewaste/`, selected
precisely on that criterion. The v3 pipeline never read it.

## Decision

Draw the e-waste object pool from the curated directory. Leave the organic
side unchanged.

### E-waste

- **Pool**: every photograph in `raw/ewaste/`, replacing the stratified
  TrashBox draw. `N_EWASTE_POOL` stops being a sample size; the pool is
  whatever the curated directory holds, less automated-screen rejects.
- **Test**: still drawn from TrashBox, but **excluding by basename any
  photograph present in the curated pool**. The curated files were copied out
  of TrashBox, so 9 of the 46 currently appear in `ewaste_test.csv`. Without
  the exclusion those 9 become a train/test leak of exactly the kind
  `01_build_splits.py` was written to prevent.

### Organic — deliberately unchanged

Backgrounds and occluders keep coming from the general RealWaste collection
via `organic_bg` / `organic_clutter`.

Curation earns its place for cut-outs because a bad matting result is an
unrecognisable blob presented to the detector as a positive example.
Backgrounds and occluders are used as whole photographs and never segmented,
so matting quality does not apply to them. What they contribute is diversity,
and diversity in the negative context is what holds the false-positive rate
down.

The curated organic directory holds 20 photographs, all Vegetation. The
false-positive test set is 747 photographs, 49% Food Organics. Compositing
only on curated organic would leave half the false-alarm test set drawn from
a category never seen during training, and would degrade the very metric the
organic photographs exist to protect.

## Known consequence: category coverage

The curated pool does not cover the test distribution evenly.

| Category | Curated pool | Share of `ewaste_test` |
|---|---|---|
| electronic chips | 19 | 21% |
| electrical cables | 10 | 23% |
| generic "e-waste" | 7 | — |
| ad-hoc downloads | 6 | — |
| smartphones | 2 | 9% |
| miscellaneous | 2 | — |
| **laptops** | **0** | **17%** |
| **small appliances** | **0** | **31%** |

Roughly 48% of the evaluation set belongs to two categories absent from
training. The pool-25 ablation already established that object-pool diversity
drives performance, so detection rate is expected to fall relative to the
uncurated 200-object pool.

That fall would be a data-coverage artefact, not a property of the curation.
Reporting it without the caveat would misattribute the cause.

**Mitigation**: additional curated photographs of laptops and small
appliances are to be added to `raw/ewaste/` before the rebuild. The
implementation lands first; the rebuild waits.

## Invalidated artefacts

Everything downstream of the uncurated pool is deleted, not kept for
comparison — the object pool is the independent variable, so results from the
old pool are not comparable to results from the new one.

- `cutouts/ewaste/`, `cutouts/ewaste_clean/`
- `dataset_pool200/`, `dataset_pool25/`
- `runs/detect/pool200{,_s1,_s2}`, `runs/detect/pool25`
- `eval_pool200{,_s1,_s2}`, `eval_pool25`

`cutouts/organic/` is retained; the organic side is unchanged.

## Rebuild sequence

Held until the curated directory is extended.

```
python 01_build_splits.py
python 02_make_cutouts.py
python 03_screen_cutouts.py --pool <survivors>
python 04_build_dataset.py --pool <survivors>
python 05_train.py --pool <n> --model yolov8s.pt
python 05_train.py --pool <n> --model yolo11s.pt --tag yolo11s
```

Then the three-model comparison table.

## Consequences for the manuscript

The methods section currently describes a stratified draw from TrashBox
followed by automated screening. That description becomes wrong and must be
rewritten to state that the object pool is hand-curated, give the selection
criterion, and report the category coverage gap in the limitations.

The pool-25 diversity ablation loses its basis — it compared 25 against 200
objects drawn from the uncurated pool. Either it is dropped or it is re-run
within the curated pool at whatever sizes the curated set supports.
