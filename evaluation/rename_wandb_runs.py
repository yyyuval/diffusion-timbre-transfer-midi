"""Give the W&B runs names that say what they are.

The tags written at launch use shorthand that only makes sense to whoever was
in the conversation: `gt` is the MIDI that shipped with the dataset, `bp` is
MIDI we transcribed from the audio ourselves, `ms` is the multiscale injection.
Six months from now none of that reads.

The vocabulary here:
    realmidi    the score that came with the dataset
    automidi    transcribed from the audio with basic-pitch (has errors)
    wrongmidi   another recording's notes -- the control condition
    nomidi      no conditioning at all
    onelevel    the OLD injection, MIDI enters once at the top

Multiscale injection is not named, because it is now the only way MIDI is
injected -- marking the norm in every name is noise. `onelevel` stays, because
those four runs are the exception and that is the one thing worth flagging.

Only the W&B display name changes. Checkpoint directories come from `exp_tag`
and are untouched, which is what you want: a long stable name on disk, a
readable one in the charts.

    python evaluation/rename_wandb_runs.py --dry-run
    python evaluation/rename_wandb_runs.py
"""

import argparse

import wandb

PROJECT = "diffusion-timbre-transfer-Proj/diffusion-timbre-transfer"

RENAME = {
    # sigma_max=100
    "cello_gt_ms_85ep_sigma100":          "cello_realmidi_sigma100_85ep",
    "bassoon_gt_ms_85ep_sigma100":        "bassoon_realmidi_sigma100_85ep",
    "violin_cello_gt_ms_85ep_sigma100":   "violin+cello_realmidi_sigma100_85ep",
    "violin_cello_nomidi_85ep_sigma100":  "violin+cello_nomidi_sigma100_85ep",
    "flute_bassoon_gt_ms_85ep_sigma100":  "flute+bassoon_realmidi_sigma100_85ep",
    "flute_bassoon_nomidi_85ep_sigma100": "flute+bassoon_nomidi_sigma100_85ep",

    # sigma_max=5
    "cello_gt_ms_85ep":          "cello_realmidi_sigma5_85ep",
    "bassoon_gt_ms_85ep":        "bassoon_realmidi_sigma5_85ep",
    "violin_cello_gt_ms_85ep":   "violin+cello_realmidi_sigma5_85ep",
    "violin_cello_nomidi_85ep":  "violin+cello_nomidi_sigma5_85ep",
    "flute_bassoon_gt_ms_85ep":  "flute+bassoon_realmidi_sigma5_85ep",
    "flute_bassoon_nomidi_85ep": "flute+bassoon_nomidi_sigma5_85ep",

    # The 40-epoch batch -- the only runs using the old single injection.
    "bassoon_gt_40ep":        "bassoon_realmidi_onelevel_sigma5_40ep",
    "bassoon_bp_40ep":        "bassoon_automidi_onelevel_sigma5_40ep",
    "bassoon_scrambled_40ep": "bassoon_wrongmidi_onelevel_sigma5_40ep",
    "bassoon_nomidi_40ep":    "bassoon_nomidi_sigma5_40ep",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would change, write nothing")
    args = ap.parse_args()

    api = wandb.Api()
    runs = list(api.runs(args.project))
    print("%d runs in %s\n" % (len(runs), args.project))

    renamed, already = set(), []
    for run in runs:
        if run.name in RENAME:
            new = RENAME[run.name]
            print("  %-40s -> %s" % (run.name, new))
            if not args.dry_run:
                run.name = new
                run.update()
            renamed.add(RENAME[run.name] if args.dry_run else new)
        elif run.name in RENAME.values():
            already.append(run.name)

    if already:
        print("\nalready renamed, left alone:")
        for n in sorted(already):
            print("  %s" % n)

    # The old tags are reconstructed from memory, so say plainly which ones
    # were never found rather than silently doing less than asked.
    missing = [old for old, new in RENAME.items()
               if new not in renamed and new not in already]
    if missing:
        print("\nNOT FOUND in this project (check the exact tag):")
        for old in sorted(missing):
            print("  %s" % old)

    if args.dry_run:
        print("\n(dry run -- nothing was written)")


if __name__ == "__main__":
    main()
