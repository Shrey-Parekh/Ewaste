

from pathlib import Path
import csv
import hashlib
import sys

from pipeline_common import perceptual_hash, same_image

# This file lives in src/; the data it reads and writes lives beside src/, not
# inside it. SRC is used for loading sibling modules by path, ROOT for anything
# on disk.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
SPLITS = ROOT / "splits"
CUTOUTS = ROOT / "cutouts"
BACKGROUNDS = ROOT / "backgrounds"

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


def role_hashes(name: str):
    """{content hash: path} for one role, or None if the manifest is missing."""
    path = SPLITS / (name + ".csv")
    if not path.exists():
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(hashlib.md5((ROOT / r["path"]).read_bytes()).hexdigest(),
                           []).append(r["path"])
    return out


def check_content(failures):
    """
    The same photograph under two paths. Checks 1 and 2 compare paths, and
    passed while a training photograph was a renamed, byte-identical copy of a
    test photograph. This compares the bytes.
    """
    print()
    print("5. no photograph shared by content, across or within roles")
    hashes = {name: role_hashes(name) for name in ALL_ROLES}
    for name, h in hashes.items():
        dup = [v for v in h.values() if len(v) > 1]
        if dup:
            failures.append("{} holds {} photographs twice under different "
                            "names".format(name, len(dup)))
        print("   {}  {:<16} duplicates within: {}".format(
            "ok " if not dup else "FAIL", name, len(dup)))
    for i, a in enumerate(ALL_ROLES):
        for b in ALL_ROLES[i + 1:]:
            both = set(hashes[a]) & set(hashes[b])
            if both:
                failures.append("{} and {} share {} photographs by content".format(
                    a, b, len(both)))
                for h in sorted(both)[:3]:
                    print("        {} == {}".format(hashes[a][h][0], hashes[b][h][0]))
            print("   {}  {:<16} vs {:<16} {:>4}".format(
                "ok " if not both else "FAIL", a, b, len(both)))


def check_near_duplicates(failures):
    """
    The same photograph saved twice with different bytes. Check 5 compares
    bytes and passed while three training photographs were re-encoded copies
    of test photographs. This compares perceptual hashes, pairwise within the
    e-waste pool and between every training role and every evaluation role.
    """
    print()
    print("6. no photograph shared as a re-saved copy (perceptual hash)")
    ph = {}
    for name in ALL_ROLES:
        with open(SPLITS / (name + ".csv"), encoding="utf-8") as f:
            ph[name] = [(r["path"], perceptual_hash(ROOT / r["path"]))
                        for r in csv.DictReader(f)]

    pool = ph["ewaste_pool"]
    twins = [(a, b) for i, (a, ha) in enumerate(pool)
             for b, hb in pool[i + 1:] if same_image(ha, hb)]
    if twins:
        failures.append("ewaste_pool holds {} pictures twice".format(len(twins)))
    print("   {}  ewaste_pool       near-duplicates within: {}".format(
        "ok " if not twins else "FAIL", len(twins)))
    for a, b in twins[:3]:
        print("        {} ~ {}".format(a, b))

    for tr in TRAIN_ROLES:
        for te in TEST_ROLES:
            hits = [(a, b) for a, ha in ph[tr] for b, hb in ph[te] if same_image(ha, hb)]
            if hits:
                failures.append("{} and {} share {} pictures as re-saved "
                                "copies".format(tr, te, len(hits)))
            print("   {}  {:<16} vs {:<16} {:>4}".format(
                "ok " if not hits else "FAIL", tr, te, len(hits)))
            for a, b in hits[:3]:
                print("        {} ~ {}".format(a, b))


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
        # Split into train/ and val/ by 04_build_dataset.py, so that synthetic
        # validation never reuses a training background.
        n_disk = sum(1 for p in BACKGROUNDS.rglob("*") if p.is_file())
        n_manifest = len(roles["organic_bg"])
        if n_disk != n_manifest:
            failures.append("backgrounds holds {} images but the manifest "
                            "lists {}".format(n_disk, n_manifest))
        print("   {}  on disk {}, manifest {}".format(
            "ok " if n_disk == n_manifest else "FAIL", n_disk, n_manifest))
        sides = [{hashlib.md5(p.read_bytes()).hexdigest()
                  for p in (BACKGROUNDS / d).glob("*") if p.is_file()}
                 for d in ("train", "val")]
        shared = sides[0] & sides[1]
        if shared:
            failures.append("{} backgrounds are in both synthetic train and "
                            "val".format(len(shared)))
        print("   {}  synthetic train/val backgrounds shared: {}".format(
            "ok " if not shared else "FAIL", len(shared)))

    check_content(failures)
    check_near_duplicates(failures)

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
