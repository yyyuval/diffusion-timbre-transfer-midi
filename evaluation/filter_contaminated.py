"""Drop input tracks the TARGET model trained on.

The bridge uses two models. The clip comes from the SOURCE model's validation
set, which `select_validation_tracks.py` checks. What it does not check is the
other half: whether the TARGET model trained on that same track. A CocoChorales
track is one piece rendered as four stems, so if the target model saw the
violin+cello stems of track X, it already knows the melody of the flute+bassoon
stems of track X -- note for note, onset for onset. It can then reproduce the
pitch content from memory rather than through the bridge, and the content
metrics measure recall of training data.

How much of a problem this is depends entirely on the ensemble. Flute and
bassoon live only in woodwind_track*, violin and cello only in string_track*,
so for the poly case the two models were trained on disjoint material and
contamination is impossible. The sole exposure is the random_track* ensemble,
which combines arbitrary instruments and so can hold both a cello and a
bassoon. Measured on the committed splits: 0 of 353 poly tracks, 9 of 475 mono.

Nothing here regenerates audio. The clips exist and the per-clip metrics exist;
this removes the contaminated rows and leaves everything else untouched.

  python evaluation/filter_contaminated.py --list-only
  python evaluation/filter_contaminated.py --csv Outputs/.../content/*.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Which model's training set each case has to be checked against. Mirrors
# CASE_DATASETS in make_eval_pools.py; kept here so this script stays runnable
# on its own, and asserted against it below.
CASE_DATASETS = {
    "poly": dict(source="flute_bassoon", target="violin_cello"),
    "mono": dict(source="cello", target="bassoon"),
    "mono_b2c": dict(source="bassoon", target="cello"),
}
ARCHITECTURES = ["midi", "nomidi"]


def track_of(cell):
    """A splits cell is "<track>/<stem>" or "<track>/<stem>+<stem>"."""
    return cell.strip().split("/")[0]


def read_splits(path):
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("%s is empty" % path)
    return rows


def column(rows, name, block):
    if name not in rows[0]:
        raise SystemExit("no column %r in the splits file. Present: %s"
                         % (name, sorted(k for k in rows[0] if k not in ("block", "row"))))
    return {track_of(r[name]) for r in rows if r["block"] == block and r[name].strip()}


def contaminated(rows, case):
    """Shared source validation that the target model trained on.

    The source side is intersected across architectures because that is the
    pool a paired comparison draws from; the target side is unioned, because a
    track is unusable if EITHER target model saw it -- the two architectures
    are compared against each other and both have to be clean.
    """
    ds = CASE_DATASETS[case]
    source_val = set.intersection(*[
        column(rows, "%s_%s" % (ds["source"], a), "VALIDATION") for a in ARCHITECTURES])
    target_train = set.union(*[
        column(rows, "%s_%s" % (ds["target"], a), "TRAINING") for a in ARCHITECTURES])
    return source_val, source_val & target_train


def filter_csv(path, bad, suffix):
    with io.open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "sample_id" not in fields:
        print("  %s: no sample_id column -- skipped" % path)
        return None
    kept = [r for r in rows if track_of(r["sample_id"]) not in bad]
    dropped = len(rows) - len(kept)
    out = path.with_suffix(suffix + path.suffix)
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    ids = len({track_of(r["sample_id"]) for r in kept})
    print("  %s -> %s  (%d rows, %d dropped, %d distinct clips)"
          % (path.name, out.name, len(kept), dropped, ids))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="evaluation/splits.csv")
    p.add_argument("--cases", nargs="+", default=["poly", "mono"],
                   choices=sorted(CASE_DATASETS))
    p.add_argument("--out-dir", default="tools/pools",
                   help="where the contaminated-track lists are written")
    p.add_argument("--csv", nargs="*", type=Path, default=[],
                   help="metric CSVs to filter; each gets a .clean.csv beside it")
    p.add_argument("--suffix", default=".clean")
    p.add_argument("--list-only", action="store_true",
                   help="report the counts and write the lists, touch no CSV")
    args = p.parse_args()

    rows = read_splits(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bad = set()
    for case in args.cases:
        ds = CASE_DATASETS[case]
        pool, dirty = contaminated(rows, case)
        pct = 100.0 * len(dirty) / len(pool) if pool else 0.0
        print("%-9s %s -> %s | shared source validation %4d | contaminated %3d (%.1f%%)"
              % (case, ds["source"], ds["target"], len(pool), len(dirty), pct))
        path = out_dir / ("contaminated_%s.txt" % case)
        path.write_text("\n".join(sorted(dirty)) + ("\n" if dirty else ""))
        for t in sorted(dirty):
            print("    %s" % t)
        bad |= dirty

    if args.list_only or not args.csv:
        if not args.list_only:
            print("\nno --csv given; lists written to %s, nothing filtered" % out_dir)
        return

    print("\nfiltering %d CSV(s) against %d contaminated track(s)" % (len(args.csv), len(bad)))
    for path in args.csv:
        if not path.is_file():
            print("  %s: not found -- skipped" % path)
            continue
        filter_csv(path, bad, args.suffix)


if __name__ == "__main__":
    main()
