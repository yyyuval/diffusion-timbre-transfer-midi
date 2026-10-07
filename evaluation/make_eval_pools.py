"""Turn splits.csv into the candidate pools every metric draws from.

`dump_splits.py` recovered, per (dataset, architecture), which recordings went
to validation and which to training. This converts that into the pools the
evaluation is allowed to use, and it is the only place the contamination rules
live.

**The rules, and why.** The official CocoChorales test split was extracted into
the training pool before anyone noticed, so no test set survives. What does
survive is an accident of the unseeded `torch.randperm` in
`fractional_random_split`: a MIDI model and a no-MIDI model trained on the same
data land on *different* splits, because model construction consumes the global
RNG first and the two architectures have different parameter counts. A
recording held out by both was trained on by neither.

For a bridge, two models touch a clip and both can leak:

  * the SOURCE model reverse-diffuses the input. If it trained on that clip it
    can reconstruct from memory, and the bridge looks better than it is.
  * the TARGET model generates the output. If it trained on the clip, nothing
    stops it regurgitating the version it memorised.

So a clip is usable for a condition when the source model held it out AND the
target model never saw it. That is `pool_<case>_<arch>.txt`.

sigma_max changes no parameter count, so `<run>` and `<run>_sigma100` share a
split: one pool serves both, and their numbers are directly comparable.

Three kinds of list come out:

  pool_<case>_<arch>.txt   clips usable by that architecture, both sigmas.
                           Big (hundreds to ~1.5k). Feeds FAD, which needs n
                           well above its embedding dimension and does not pair
                           clips with anything.

  shared_<case>.txt        clips in every architecture's pool for that case.
                           Small. Feeds the content metrics, which compare
                           MIDI against no-MIDI clip by clip -- a paired test
                           is only valid on identical inputs.

  reference_<case>.txt     real TARGET-instrument recordings both target
                           architectures held out. The FAD reference, shared by
                           all conditions so the comparison between them is
                           fair.

    python evaluation/make_eval_pools.py --splits tools/splits.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# case -> which dataset supplies the input, which generates the output. The
# names are splits.csv column stems; "_midi"/"_nomidi" is appended.
CASE_DATASETS = {
    "poly":     dict(source="flute_bassoon", target="violin_cello"),
    "mono":     dict(source="cello",         target="bassoon"),
    "mono_b2c": dict(source="bassoon",       target="cello"),
}
ARCHITECTURES = ["midi", "nomidi"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="tools/splits.csv",
                   help="the table dump_splits.py wrote")
    p.add_argument("--out-dir", default="tools/pools")
    p.add_argument("--cases", nargs="+", default=list(CASE_DATASETS),
                   choices=list(CASE_DATASETS))
    return p.parse_args()


def read_splits(path):
    """{column: {"VALIDATION": {tracks}, "TRAINING": {tracks}}}.

    Cells are "<track>/<stem>" or "<track>/<stem>+<stem>"; only the track
    matters, since a clip is one track's stems summed. Blank cells are padding
    -- the columns are different lengths and the writer squared them off.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames if c not in ("block", "row")]
        out = {c: {"VALIDATION": set(), "TRAINING": set()} for c in columns}
        for row in reader:
            block = row["block"]
            for c in columns:
                cell = (row.get(c) or "").strip()
                if cell:
                    out[c][block].add(cell.split("/", 1)[0])
    return out


def write(path, tracks):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(tracks)) + "\n", encoding="utf-8")
    return len(tracks)


def main():
    args = parse_args()
    splits = read_splits(args.splits)
    out_dir = Path(args.out_dir)

    missing = []
    for case in args.cases:
        for role in ("source", "target"):
            for arch in ARCHITECTURES:
                col = "%s_%s" % (CASE_DATASETS[case][role], arch)
                if col not in splits:
                    missing.append(col)
    if missing:
        raise SystemExit("%s has no column(s): %s\nfound: %s"
                         % (args.splits, ", ".join(sorted(set(missing))),
                            ", ".join(sorted(splits))))

    print("%-10s %-8s %9s %9s %9s %9s" %
          ("case", "arch", "src_val", "-tgt_seen", "= pool", "shared"))
    print("-" * 60)

    for case in args.cases:
        datasets = CASE_DATASETS[case]
        pools = {}
        for arch in ARCHITECTURES:
            source_val = splits["%s_%s" % (datasets["source"], arch)]["VALIDATION"]
            target_train = splits["%s_%s" % (datasets["target"], arch)]["TRAINING"]
            # Only tracks carrying both instruments can collide; in CocoChorales
            # that is the `random` ensemble, where the families mix.
            pool = source_val - target_train
            pools[arch] = pool
            n = write(out_dir / ("pool_%s_%s.txt" % (case, arch)), pool)
            print("%-10s %-8s %9d %9d %9d" %
                  (case, arch, len(source_val), len(source_val) - n, n), end="")
            print()

        shared = set.intersection(*pools.values())
        write(out_dir / ("shared_%s.txt" % case), shared)

        # The FAD reference: real target audio neither target model trained on.
        reference = set.intersection(*[
            splits["%s_%s" % (datasets["target"], arch)]["VALIDATION"]
            for arch in ARCHITECTURES])
        write(out_dir / ("reference_%s.txt" % case), reference)

        print("%-10s %-8s %9s %9s %9s %9d" % (case, "shared", "", "", "", len(shared)))
        print("%-10s %-8s %9s %9s %9s %9d" % (case, "reference", "", "", "", len(reference)))
        print("-" * 60)

    print("\nwrote %s/{pool,shared,reference}_*.txt" % out_dir)
    print("""
How to read the numbers:

  pool       feeds FAD. CLAP embeddings are 512-D and EnCodec's are 128-D; a
             Frechet distance fit on fewer samples than dimensions is singular
             and the estimate is biased upward. Want >= ~1000 for CLAP, >= ~400
             for EnCodec.
  shared     feeds the paired content metrics. 200 is enough there -- the
             paired test gains far more from identical inputs than from n.
  reference  the real target audio FAD measures against. Same caveat on size;
             if it falls short of what CLAP needs, pad it with
             --pad-reference-to and accept that the padding was seen in
             training (it biases every condition equally, so the ranking
             survives even though the absolute value does not).""")


if __name__ == "__main__":
    main()
