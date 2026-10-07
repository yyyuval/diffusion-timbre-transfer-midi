"""Recover which recordings went to validation and which to training.

Nothing was ever written down. The split comes from `torch.randperm` inside
`fractional_random_split` (audio_data_pytorch/utils.py:48), which has no
generator of its own and so draws from the global RNG -- *after* model
construction has already consumed it. That is why a MIDI model and a no-MIDI
model trained on identical data land on different splits.

It is reproducible, though, because it is deterministic: same seed, same file
list, same number of draws before it. This replays `train.py`'s startup in the
same order and reads the split back out.

**The reproduction is unverified.** If anything consumed RNG between
`seed_everything` and `setup()` that is not replayed here, the answer is wrong
with no outward sign. Two guards:

  * `--check` runs the whole thing twice and asserts the split is identical,
    catching nondeterminism in file discovery (single-instrument mode uses an
    *unsorted* `os.scandir`, so the file ORDER is not guaranteed stable across
    machines or NFS states -- if it moved, the indices point elsewhere).
  * `--with-callbacks` additionally instantiates the callbacks, as train.py
    does, and reports whether that changes the answer. If it does not, they
    consume no RNG and the shorter replay is sound.

sigma_max does not affect any of this: it changes no parameter count, so
`<run>` and `<run>_sigma100` always share a split. Eight distinct splits, not
sixteen.

    python evaluation/dump_splits.py --check
    python evaluation/dump_splits.py --out splits.csv
"""

import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

# The eight distinct (dataset, architecture) pairs. The run directory is only
# used to read back the exact overrides that run was launched with.
CONFIGS = [
    ("cello_midi",          "cello_gt_ms_85ep"),
    ("cello_nomidi",        "cello_nomidi_85ep"),
    ("bassoon_midi",        "bassoon_gt_ms_85ep"),
    ("bassoon_nomidi",      "bassoon_nomidi_85ep"),
    ("violin_cello_midi",   "violin_cello_gt_ms_85ep"),
    ("violin_cello_nomidi", "violin_cello_nomidi_85ep"),
    ("flute_bassoon_midi",  "flute_bassoon_gt_ms_85ep"),
    ("flute_bassoon_nomidi","flute_bassoon_nomidi_85ep"),
]


def overrides_of(run_dir):
    """The exact CLI overrides a run was launched with, as it recorded them."""
    import yaml
    path = os.path.join(REPO, "our_checkpoints", run_dir, "runs", ".hydra", "overrides.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError("no overrides.yaml for %s -- has it trained?" % run_dir)
    with open(path) as f:
        ovr = yaml.safe_load(f)
    # exp_tag only names the output directory; dropping it avoids writing there.
    return [o for o in ovr if not o.startswith("exp_tag=")]


def split_for(run_dir, with_callbacks=False):
    """Replay train.py's startup and read the split back.

    Order matters and mirrors train.py:33-91 exactly -- seed, datamodule, model,
    then (optionally) callbacks. setup() is what Lightning calls inside fit(),
    and it is where randperm runs.
    """
    import hydra
    from hydra import compose, initialize_config_dir
    import pytorch_lightning as pl

    with initialize_config_dir(config_dir=REPO, version_base=None):
        cfg = compose(config_name="config",
                      overrides=overrides_of(run_dir) + ["exp_tag=_splitprobe"])

    pl.seed_everything(cfg.seed)                                    # train.py:58
    datamodule = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")  # the big RNG consumer

    if with_callbacks and "callbacks" in cfg:
        for _, cb in cfg["callbacks"].items():
            if "_target_" in cb:
                hydra.utils.instantiate(cb, _convert_="partial")

    datamodule.setup()
    ds = datamodule.dataset

    def label(i):
        """<track>/<stem>, or <track>/<stem+stem> for a mixture."""
        paths = ds.groups[i] if getattr(ds, "groups", None) else [ds.wavs[i]]
        track = os.path.basename(os.path.dirname(os.path.dirname(paths[0])))
        stems = "+".join(os.path.splitext(os.path.basename(p))[0] for p in paths)
        return "%s/%s" % (track, stems)

    val = [label(int(i)) for i in datamodule.data_val.indices]
    train = [label(int(i)) for i in datamodule.data_train.indices]
    del model, datamodule
    return val, train, len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/splits.csv")
    ap.add_argument("--check", action="store_true",
                    help="run each split twice and assert they match")
    ap.add_argument("--with-callbacks", action="store_true",
                    help="also instantiate callbacks, as train.py does")
    ap.add_argument("--only", nargs="+", help="limit to these column names")
    args = ap.parse_args()

    wanted = [(n, d) for n, d in CONFIGS if not args.only or n in args.only]
    results = {}

    for name, run_dir in wanted:
        print("\n=== %s  (%s)" % (name, run_dir))
        val, train, total = split_for(run_dir, args.with_callbacks)
        print("    %d recordings -> %d validation / %d training" % (total, len(val), len(train)))

        if args.check:
            val2, train2, _ = split_for(run_dir, args.with_callbacks)
            if val == val2 and train == train2:
                print("    reproducible: identical on a second run")
            else:
                print("    *** NOT REPRODUCIBLE -- the file order or the RNG replay moved.")
                print("    *** Single-instrument discovery uses an unsorted os.scandir;")
                print("    *** if that is the cause, the recovered split is not the real one.")
        results[name] = (val, train, total)

    # ---- the table -----------------------------------------------------------
    # Validation on top, training below, with the boundary on one row across all
    # columns. Columns are different lengths, so each block is padded to the
    # widest of its kind; the blanks are the shorter datasets, not missing data.
    names = [n for n, _ in wanted]
    max_val = max(len(results[n][0]) for n in names)
    max_train = max(len(results[n][1]) for n in names)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block", "row"] + names)
        for r in range(max_val):
            w.writerow(["VALIDATION", r] +
                       [results[n][0][r] if r < len(results[n][0]) else "" for n in names])
        for r in range(max_train):
            w.writerow(["TRAINING", r] +
                       [results[n][1][r] if r < len(results[n][1]) else "" for n in names])

    print("\n%s" % ("-" * 64))
    print("%-22s %9s %9s %9s" % ("column", "total", "validation", "training"))
    for n in names:
        val, train, total = results[n]
        print("%-22s %9d %9d %9d" % (n, total, len(val), len(train)))
    print("\nwrote %s  (%d validation rows, then %d training rows)"
          % (args.out, max_val, max_train))

    # ---- what this was for ---------------------------------------------------
    # A recording held out by BOTH architectures was trained on by neither, so it
    # is usable for evaluation even though no test split survives.
    print("\nheld out by BOTH architectures (usable as an evaluation set):")
    for data in ["cello", "bassoon", "violin_cello", "flute_bassoon"]:
        a, b = data + "_midi", data + "_nomidi"
        if a in results and b in results:
            both = set(results[a][0]) & set(results[b][0])
            print("  %-16s %5d of %d" % (data, len(both), results[a][2]))
            with open("evaluation/heldout_%s.txt" % data, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(both)) + "\n")


if __name__ == "__main__":
    main()
