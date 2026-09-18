"""
01_build_splits.py
------------------
STEP 0 of the v3 pipeline.  Runs ONCE.

WHY THIS EXISTS
    In the v2 pipeline the e-waste training cut-outs were segmented from the
    same 46 photographs that 04_eval_real.py scored detection rate on, so the
    reported detection rate was measured on objects the detector had already
    been trained on.  This script removes that overlap by partitioning the
    source collections BEFORE anything is extracted or composited.

    Every downstream stage reads these manifests instead of globbing whole
    directories, so the disjointness is enforced by construction rather than
    by convention.

PARTITION
    e-waste
        ewaste_pool  all of raw/ewaste  -> cut-outs -> training data
        ewaste_test  <=400 from TrashBox -> held-out real photographs,
                                           detection rate

        ewaste_test excludes the "laptops" and "small appliances" categories
        entirely (see EWASTE_TEST_EXCLUDE_CATEGORIES): an object that large
        would be removed by hand before organic waste reached a sorting line,
        so it is not the contamination the paper targets. This also matches
        the curated pool, which has never contained either category.

        The pool is HAND-CURATED, not sampled. raw/ewaste holds photographs
        selected by eye for yielding a cut-out that still reads as electronic
        waste once segmented -- a judgement the automated screen in
        03_screen_cutouts.py cannot make, since it can reject a matting
        failure but not an ambiguous subject.

        Those files were copied out of TrashBox, so the test draw excludes
        any TrashBox photograph whose basename matches a curated one.
        Without that exclusion the curated pool would leak straight into the
        test set.

    organic  (RealWaste Food Organics + Vegetation, stratified)
        organic_bg      50 -> compositing backgrounds
        organic_clutter 50 -> clutter / occluder cut-outs
        organic_test   747 -> held-out real photographs, false-positive rate

        Deliberately NOT curated. Backgrounds and occluders are used as whole
        photographs and never segmented, so matting quality does not apply to
        them; what they contribute is diversity, and diversity in the negative
        context is what holds the false-positive rate down. Half of
        organic_test is Food Organics, a category the curated organic
        directory does not cover at all.

    The three organic roles partition all 847 images with no remainder, so
    the false-positive set is fully disjoint from every training input.

Run:   python 01_build_splits.py
Output: splits/*.csv, splits/summary.json
"""

from pathlib import Path
import csv
import hashlib
import json
import random

from pipeline_common import perceptual_hash, same_image

# ------------------------- CONFIG -------------------------
# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# The training pool is this hand-curated directory, taken whole. It is NOT
# sampled from TrashBox: the selection criterion is whether a photograph still
# reads as electronic waste after segmentation, which no automated screen can
# decide.
EWASTE_POOL_SRC = ROOT / "raw" / "ewaste"

# The test set is drawn from the TrashBox collection, so detection rate is
# measured against a wide range of unseen objects.
EWASTE_TEST_SRC = ROOT / "assets" / "trash" / "TrashBox" / "TrashBox_train_dataset_subfolders" / "e-waste"

# Excluded from the test source entirely: an item this large would be pulled
# out by hand before organic waste ever reached a sorting line, so asking the
# detector to find one is not the contamination scenario the paper targets.
# This also removes the category-coverage gap against the curated pool, which
# has never contained laptops or small appliances.
EWASTE_TEST_EXCLUDE_CATEGORIES = {"laptops", "small appliances"}

ORGANIC_SRC = ROOT / "assets" / "trash" / "realwaste-main" / "RealWaste"
ORGANIC_CATEGORIES = ["Food Organics", "Vegetation"]

OUT = ROOT / "splits"

# No N_EWASTE_POOL: the pool is however many photographs raw/ewaste holds,
# less whatever 03_screen_cutouts.py rejects for matting quality.
N_EWASTE_TEST = 400
N_ORGANIC_BG = 50
N_ORGANIC_CLUTTER = 50
# organic_test takes everything that is left

SEED = 0

# Categories for curated photographs whose filenames carry none. The label
# comes from the filename prefix elsewhere, and these sixteen were saved as
# "e-waste NNN", "images (N)" and similar, or under a misleading name:
# "laptops 198" is an Intel desktop CPU. Assigned by looking at each image.
# Batteries have no counterpart category in the test set; they stay in the
# pool as genuine e-waste but are reported as their own stratum.
CATEGORY_OVERRIDES = {
    "e-waste 1222.jpg": "electronic chip",      # optical drive pickup assembly
    "e-waste 2825.jpg": "smartphones",
    "e-waste 2843.jpg": "smartphones",
    "e-waste 2966.jpg": "smartphones",
    "e-waste 3006.jpg": "smartphones",
    "e-waste 783.jpg": "electronic chip",       # circuit board
    "e-waste 913.jpg": "electronic chip",       # speed-controller board
    "images (1).jpg": "batteries",
    "images (2).jpg": "smartphones",
    "images (3).jpg": "electronic chip",        # motherboard
    "images (4).jpg": "electronic chip",        # circuit board
    "images (5).jpg": "electrical cables",
    "images.jpg": "batteries",
    "istockphoto-2246139796-612x612.jpg": "batteries",
    "laptops 198.jpg": "electronic chip",       # desktop CPU, not a laptop
    "Miscellaneous Trash_315.jpg": "electrical cables",
}
# ----------------------------------------------------------

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT)


def stratified_take(by_category: dict, n: int, rng: random.Random):
    """
    Draw n items spread proportionally across categories, consuming them from
    the pools so later draws cannot collide with earlier ones.

    Proportional allocation with largest-remainder rounding, so the split
    mirrors the composition of the source collection rather than flattening it.
    """
    sizes = {c: len(v) for c, v in by_category.items()}
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"asked for {n} images, only {total} remain")

    exact = {c: n * s / total for c, s in sizes.items()}
    alloc = {c: int(v) for c, v in exact.items()}
    remainder = n - sum(alloc.values())
    order = sorted(exact, key=lambda c: exact[c] - alloc[c], reverse=True)
    for c in order[:remainder]:
        alloc[c] += 1

    taken = []
    for c in sorted(by_category):
        k = min(alloc[c], len(by_category[c]))
        rng.shuffle(by_category[c])
        taken.extend((c, p) for p in by_category[c][:k])
        by_category[c] = by_category[c][k:]
    return taken


def content_hash(path: Path) -> str:
    """Identity of a photograph by its bytes, not its name."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_manifest(name: str, rows):
    path = OUT / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role", "category", "path"])
        for category, p in rows:
            w.writerow([name, category, p.relative_to(ROOT).as_posix()])
    print(f"  {name:<16} {len(rows):>4}  -> {path.relative_to(ROOT).as_posix()}")
    return len(rows)


def curated_category(path: Path) -> str:
    """
    Category label for a curated photograph, taken from its filename stem.

    The curated directory is flat -- the category structure TrashBox provides
    through directory names is lost when files are copied out of it -- so the
    label is recovered from the filename prefix, except where the filename
    carries none and CATEGORY_OVERRIDES gives one assigned by eye.
    """
    if path.name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[path.name]
    stem = path.stem
    for sep in (" ", "_"):
        if sep in stem:
            head = stem.rsplit(sep, 1)
            if head[-1].isdigit():
                return head[0].replace("_", " ").strip()
    return stem.replace("_", " ").strip()


def main():
    rng = random.Random(SEED)
    OUT.mkdir(exist_ok=True)

    # ---- gather sources ----
    curated = list_images(EWASTE_POOL_SRC)
    if not curated:
        print(f"[!] no curated e-waste photographs under {EWASTE_POOL_SRC}")
        return

    ewaste = {d.name: list_images(d) for d in sorted(EWASTE_TEST_SRC.iterdir())
              if d.is_dir() and d.name not in EWASTE_TEST_EXCLUDE_CATEGORIES} \
        if EWASTE_TEST_SRC.exists() else {}
    if not ewaste:
        print(f"[!] no e-waste categories under {EWASTE_TEST_SRC}")
        return

    # The curated files were copied out of TrashBox, so the same photograph can
    # sit in both places under the same name. Drop those from the test source
    # BEFORE drawing, or the training pool leaks into the test set.
    curated_names = {p.name for p in curated}
    excluded = 0
    for c in ewaste:
        before = len(ewaste[c])
        ewaste[c] = [p for p in ewaste[c] if p.name not in curated_names]
        excluded += before - len(ewaste[c])

    organic = {c: list_images(ORGANIC_SRC / c) for c in ORGANIC_CATEGORIES}
    if not any(organic.values()):
        print(f"[!] no organic images under {ORGANIC_SRC}")
        return

    print("Source collections")
    print(f"  e-waste / curated pool     {len(curated):>5}  ({EWASTE_POOL_SRC.name}/)")
    for c, v in sorted(ewaste.items()):
        print(f"  e-waste / {c:<20} {len(v):>5}")
    print(f"  ({excluded} TrashBox photographs withheld from the test source "
          f"because they also appear in the curated pool)")
    for c, v in sorted(organic.items()):
        print(f"  organic / {c:<20} {len(v):>5}")

    # ---- partition (order matters: each draw consumes from the pool) ----
    print("\nManifests")
    counts = {}
    ewaste_test = stratified_take(ewaste, N_EWASTE_TEST, rng)

    # The name-based exclusion above misses a photograph copied out of
    # TrashBox and renamed, and a byte comparison misses one that was also
    # re-saved. Four curated photographs were one or the other: copies of test
    # photographs under different names, three of them re-encoded. Compare by
    # perceptual hash, which recognises the same picture whatever its bytes.
    #
    # A clash is resolved by dropping the curated copy, not the test one:
    # removing a test candidate would change the seeded draw and so every
    # evaluation split, whereas the pool is never drawn with the RNG and can
    # lose a member without disturbing anything else. The curated directory
    # also holds three pictures twice under different names; only the first
    # of each is kept.
    test_ph = [perceptual_hash(p) for _, p in ewaste_test]
    kept_ph, pool = [], []
    for p in curated:
        h = perceptual_hash(p)
        match = next((t for (_, t), th in zip(ewaste_test, test_ph)
                      if same_image(h, th)), None)
        if match is not None:
            print(f"  dropped from pool, same image as test photograph "
                  f"{match.name}: {p.name}")
            continue
        twin = next((q for q, qh in zip(pool, kept_ph) if same_image(h, qh)), None)
        if twin is not None:
            print(f"  dropped from pool, same image as curated {twin.name}: {p.name}")
            continue
        kept_ph.append(h)
        pool.append(p)

    counts["ewaste_pool"] = write_manifest(
        "ewaste_pool", [(curated_category(p), p) for p in pool])
    counts["ewaste_test"] = write_manifest("ewaste_test", ewaste_test)

    counts["organic_bg"] = write_manifest(
        "organic_bg", stratified_take(organic, N_ORGANIC_BG, rng))
    counts["organic_clutter"] = write_manifest(
        "organic_clutter", stratified_take(organic, N_ORGANIC_CLUTTER, rng))
    remaining = [(c, p) for c in sorted(organic) for p in organic[c]]
    counts["organic_test"] = write_manifest("organic_test", remaining)

    # ---- verify disjointness rather than assume it ----
    # By content, not path: two paths can hold the same photograph, and a
    # path-only check passed while one did.
    sets = {}
    for name in counts:
        with open(OUT / f"{name}.csv", encoding="utf-8") as f:
            sets[name] = {content_hash(ROOT / r["path"]) for r in csv.DictReader(f)}

    print("\nDisjointness check")
    names = sorted(sets)
    clashes = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                clashes += 1
                print(f"  [!] {a} and {b} share {len(overlap)} images")
    if clashes == 0:
        print("  all manifests pairwise disjoint")

    summary = {
        "seed": SEED,
        "counts": counts,
        "ewaste_pool_source": EWASTE_POOL_SRC.relative_to(ROOT).as_posix(),
        "ewaste_pool_curated": True,
        "ewaste_test_source": EWASTE_TEST_SRC.relative_to(ROOT).as_posix(),
        "ewaste_test_excluded_as_curated": excluded,
        "organic_source": ORGANIC_SRC.relative_to(ROOT).as_posix(),
        "organic_categories": ORGANIC_CATEGORIES,
        "organic_curated": False,
        "pairwise_disjoint": clashes == 0,
        "disjointness_checked_by": "content hash across all roles; the pool "
                                   "also by perceptual hash against the test "
                                   "set and itself",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSTEP 0 complete -> {OUT}")
    print("Then run:  python 02_make_cutouts.py")


if __name__ == "__main__":
    main()
