

from pathlib import Path
import csv
import json
import sys

# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
SPLITS = ROOT / "splits"
CUTOUTS = ROOT / "cutouts"
BACKGROUNDS = ROOT / "backgrounds"
ANNOTATIONS = ROOT / "annotations" / "ewaste_test"
SUBSET = ROOT / "annotations" / "subset.json"

TRAIN_ROLES = ["ewaste_pool", "organic_bg", "organic_clutter"]
TEST_ROLES = ["ewaste_test", "organic_test"]
ALL_ROLES = TRAIN_ROLES + TEST_ROLES

BACKSLASH = chr(92)


def norm(p: str) -> str:
    """Compare paths case- and separator-insensitively; this is Windows."""
    return p.replace(BACKSLASH, "/").strip().lower()


def load_role(name: str):
    path = SPLITS / (name + ".csv")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return {norm(r["path"]) for r in csv.DictReader(f)}


def category_of(name):
    """Map each test photograph's stem to its category, from the manifest."""
    out = {}
    path = SPLITS / (name + ".csv")
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stem = norm(r["path"]).rsplit("/", 1)[-1].rsplit(".", 1)[0]
            out[stem] = r["category"]
    return out


def check_annotations(failures):
    """
    Audit the hand-drawn boxes.

    This section exists because a real problem went unnoticed without it. A
    stratified sample of 100 photographs was drawn for annotation, but 27 files
    from an earlier pass were still on disk, and the evaluator reads every label
    file it finds. All 27 happened to be electrical cables, so the ground truth
    the localisation metrics were computed against was 57% cables where the test
    set is 45% -- a skew nothing in the pipeline would have reported.

    A malformed or orphaned label is a failure. A composition skew is not: the
    annotations are legitimate work and it is a judgement call whether to
    restrict the evaluation or widen the sample. It is printed loudly instead.
    """
    print()
    print("5. hand-drawn annotations")
    if not ANNOTATIONS.is_dir():
        print("   -- skipped, no annotations/ewaste_test")
        return

    labels = sorted(ANNOTATIONS.glob("*.txt"))
    if not labels:
        print("   -- skipped, no label files yet")
        return

    cats = category_of("ewaste_test")
    organic = category_of("organic_test")

    orphans, malformed, boxes = [], [], 0
    for f in labels:
        if f.stem not in cats:
            orphans.append(f.stem)
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            boxes += 1
            ok = len(parts) == 5
            if ok:
                try:
                    vals = [float(v) for v in parts[1:]]
                    ok = all(0.0 <= v <= 1.0 for v in vals) and vals[2] > 0 and vals[3] > 0
                except ValueError:
                    ok = False
            if not ok:
                malformed.append(f.name)

    print("   {} label files, {} boxes".format(len(labels), boxes))

    on_organic = [f.stem for f in labels if f.stem in organic]
    for label, bad, why in (
        ("annotate a photograph outside ewaste_test", orphans, "orphan"),
        ("are malformed", sorted(set(malformed)), "malformed"),
        ("annotate an organic_test photograph, which holds no e-waste",
         on_organic, "wrong split"),
    ):
        n = len(bad)
        if n:
            failures.append("{} label file(s) {}".format(n, label))
        print("   {}  {}: {}".format("ok " if not n else "FAIL", why, n))
        for x in sorted(bad)[:5]:
            print("        " + x)

    if not SUBSET.exists():
        print("   -- no subset.json, so every annotated photograph is in scope")
        return

    spec = json.loads(SUBSET.read_text(encoding="utf-8"))
    wanted = set(spec["stems"])
    have = {f.stem for f in labels}
    outside = sorted(have - wanted)
    missing = len(wanted - have)

    print("   stratified subset: {} of {} annotated, {} still to do".format(
        len(wanted & have), len(wanted), missing))
    if outside:
        print("   WARNING  {} annotated photographs are outside the subset.".format(
            len(outside)))
        print("            06_evaluate.py reads every label file, so these are")
        print("            included in mIoU whether or not they were sampled.")

    designed, actual = {}, {}
    for stem in wanted:
        designed[cats.get(stem, "?")] = designed.get(cats.get(stem, "?"), 0) + 1
    for stem in have:
        actual[cats.get(stem, "?")] = actual.get(cats.get(stem, "?"), 0) + 1
    total_d, total_a = sum(designed.values()), sum(actual.values())
    if total_a:
        print("   composition of the ground truth actually used:")
        for c in sorted(set(designed) | set(actual)):
            pd = 100 * designed.get(c, 0) / total_d if total_d else 0
            pa = 100 * actual.get(c, 0) / total_a
            flag = "  <-- skewed" if abs(pa - pd) > 5 else ""
            print("     {:<24} sampled {:>5.1f}%   in use {:>5.1f}%{}".format(
                c, pd, pa, flag))


def main():
    roles = {}
    for name in ALL_ROLES:
        s = load_role(name)
        if s is None:
            print("[!] missing manifest: splits/" + name + ".csv")
            print("    run 01_build_splits.py first")
            return 1
        roles[name] = s

    failures = []
    print("=" * 66)
    print("PIPELINE INTEGRITY AUDIT")
    print("=" * 66)
    print()
    print("Split sizes")
    for name in ALL_ROLES:
        kind = "train" if name in TRAIN_ROLES else "eval"
        print("  {:<18} {:>5}   ({})".format(name, len(roles[name]), kind))

    print()
    print("1. every pair of roles disjoint")
    for i, a in enumerate(ALL_ROLES):
        for b in ALL_ROLES[i + 1:]:
            n = len(roles[a] & roles[b])
            if n:
                failures.append("{} and {} share {} photographs".format(a, b, n))
            print("   {}  {:<16} vs {:<16} {:>4}".format(
                "ok " if n == 0 else "FAIL", a, b, n))

    train = set().union(*(roles[r] for r in TRAIN_ROLES))
    test = set().union(*(roles[r] for r in TEST_ROLES))
    shared = train & test
    if shared:
        failures.append("{} photographs are in both a training and an "
                        "evaluation role".format(len(shared)))
    print()
    print("2. training union ({}) vs evaluation union ({})".format(
        len(train), len(test)))
    print("   {}  shared photographs: {}".format(
        "ok " if not shared else "FAIL", len(shared)))

    print()
    print("3. extraction consumed only authorised photographs")
    log = CUTOUTS / "extraction_log.csv"
    if not log.exists():
        print("   -- skipped, no cutouts/extraction_log.csv")
    else:
        with open(log, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        used = {norm(r["source_path"]) for r in rows}
        unauth = used - train
        leaked = used & test
        if unauth:
            failures.append("{} extracted photographs are outside every "
                            "training role".format(len(unauth)))
        if leaked:
            failures.append("{} extracted photographs come from an evaluation "
                            "set".format(len(leaked)))
        print("   consumed {} photographs".format(len(used)))
        print("   {}  outside any training role: {}".format(
            "ok " if not unauth else "FAIL", len(unauth)))
        print("   {}  drawn from an evaluation set: {}".format(
            "ok " if not leaked else "FAIL", len(leaked)))
        for r in sorted(leaked)[:5]:
            print("        " + r)

    print()
    print("4. materialised backgrounds match the manifest")
    if not BACKGROUNDS.exists():
        print("   -- skipped, no backgrounds/")
    else:
        n_disk = sum(1 for p in BACKGROUNDS.iterdir() if p.is_file())
        n_manifest = len(roles["organic_bg"])
        if n_disk != n_manifest:
            failures.append("backgrounds holds {} images but the manifest "
                            "lists {}".format(n_disk, n_manifest))
        print("   {}  on disk {}, manifest {}".format(
            "ok " if n_disk == n_manifest else "FAIL", n_disk, n_manifest))

    check_annotations(failures)

    print()
    print("=" * 66)
    if failures:
        print("FAILED  {} problem(s)".format(len(failures)))
        for f in failures:
            print("  - " + f)
        print("=" * 66)
        return 1
    print("PASSED  no training photograph appears in either evaluation set")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
