"""Pick the evaluation clips, one list per (case, architecture).

  poly: woodwind tracks with 1_flute + 4_bassoon   (flute_bassoon -> violin_cello)
  mono: string tracks with 4_cello                  (cello -> bassoon)

Each list holds --n-fad tracks and is written **shared clips first**, so a
prefix is a valid paired set:

  tracks[:--n-content]   the clips every architecture held out. Identical in
                         every list, so the content metrics compare MIDI
                         against no-MIDI on the same audio and a paired test
                         means something. `run_bridge_batched.py --n 200`
                         takes exactly this prefix.
  tracks[:]              the whole list. Feeds FAD, which needs n above its
                         embedding dimension and pairs nothing, so it may use
                         clips private to one architecture.

Candidates come from `make_eval_pools.py` (--pool-dir), which encodes the
contamination rules -- read its docstring before changing anything here. With
no --pool-dir the old behaviour returns: one list per case, sampled from the
whole ensemble family. That samples training data and is kept only for
reproducing the first 20-clip runs.

Deterministic: candidates are sorted, then shuffled with --seed, and the first
that pass the checks are kept. A track passes when every input stem has a
non-empty wav AND its stems_midi/*.mid, and the MIDI has at least one note in
the 17 s window the bridge actually sees (a clip with no reference notes makes
the content metrics degenerate).

  python evaluation/make_track_lists.py --pool-dir tools/pools
  python evaluation/make_track_lists.py --pool-dir tools/pools --cases mono
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bridge_200 import (  # noqa: E402
    CASES,
    CLIP_LENGTH,
    DATASET_ROOT,
    SAMPLING_RATE,
    list_family_tracks,
    midi_path_for,
    pick_tracks,
    read_track_list,
    stem_paths,
    track_has_files,
)

ARCHITECTURES = ["midi", "nomidi"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default=DATASET_ROOT)
    p.add_argument("--pool-dir", default=None,
                   help="where make_eval_pools.py wrote pool_*/shared_*.txt")
    p.add_argument("--n-content", type=int, default=200,
                   help="paired prefix, drawn from shared_<case>.txt")
    p.add_argument("--n-fad", type=int, default=1000,
                   help="full list length, topped up from pool_<case>_<arch>.txt")
    p.add_argument("--n", type=int, default=None,
                   help="legacy single-tier size; implies no pools")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cases", nargs="+", default=["poly", "mono"], choices=list(CASES))
    p.add_argument("--out-dir", default=str(ROOT / "evaluation"))
    p.add_argument("--skip-midi-check", action="store_true",
                   help="only check that the files exist, do not parse the MIDI")
    return p.parse_args()


def midi_has_notes(midi_path: str, max_time: float) -> bool:
    import pretty_midi

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return False
    return any(note.start < max_time for inst in pm.instruments for note in inst.notes)


def make_accept(dataset_root, stems, check_midi):
    """A filter plus the rejection log, so a thin pool explains itself."""
    max_time = CLIP_LENGTH / SAMPLING_RATE
    rejected = []

    def accept(track):
        if not track_has_files(dataset_root, track, stems):
            rejected.append((track, "missing or empty stem/MIDI file"))
            return False
        if check_midi:
            for audio in stem_paths(dataset_root, track, stems):
                if not midi_has_notes(midi_path_for(audio), max_time):
                    rejected.append((track, "no notes in the first %.1f s of %s"
                                     % (max_time, Path(audio).stem)))
                    return False
        return True

    return accept, rejected


def report(label, picked, wanted, rejected):
    short = " SHORT BY %d" % (wanted - len(picked)) if len(picked) < wanted else ""
    print("  %-22s %4d of %4d wanted, %d rejected%s"
          % (label, len(picked), wanted, len(rejected), short))
    for track, reason in rejected[:5]:
        print("      rejected %s: %s" % (track, reason))


def legacy_lists(args, out_dir):
    """One list per case, sampled from the whole family. Pre-pools behaviour."""
    n = args.n or 200
    for case_name in args.cases:
        case = CASES[case_name]
        accept, rejected = make_accept(args.dataset_root, case["stems"], not args.skip_midi_check)
        picked = pick_tracks(list_family_tracks(args.dataset_root, case["family"]),
                             n, args.seed, accept=accept)
        out = out_dir / ("track_list_%d_%s.txt" % (n, case_name))
        out.write_text("\n".join(picked) + "\n")
        report(case_name, picked, n, rejected)
        print("    -> %s" % out)
    print("\nNOTE: sampled from the family at large, so these clips were almost all\n"
          "      trained on. Pass --pool-dir for sets the models never saw.")


def pooled_lists(args, out_dir):
    pool_dir = Path(args.pool_dir)
    for case_name in args.cases:
        case = CASES[case_name]
        print("=" * 70)
        print("  %s: %s -> %s" % (case_name, case["source"], case["target"]))
        print("=" * 70)

        # The paired prefix is a property of the case, not of an architecture:
        # computed once, then prepended to every list so `--n` slices the same
        # audio out of each one.
        shared_path = pool_dir / ("shared_%s.txt" % case_name)
        accept, rejected = make_accept(args.dataset_root, case["stems"], not args.skip_midi_check)
        shared = pick_tracks(read_track_list(shared_path), args.n_content, args.seed,
                             accept=accept, allow_short=True)
        report("shared (paired)", shared, args.n_content, rejected)
        if len(shared) < args.n_content:
            print("      the paired content metrics will run on %d clips, not %d."
                  % (len(shared), args.n_content))

        for arch in ARCHITECTURES:
            pool_path = pool_dir / ("pool_%s_%s.txt" % (case_name, arch))
            accept, rejected = make_accept(args.dataset_root, case["stems"],
                                           not args.skip_midi_check)
            fill = pick_tracks(read_track_list(pool_path), args.n_fad - len(shared),
                               args.seed, accept=accept, exclude=shared, allow_short=True)
            tracks = shared + fill
            out = out_dir / ("track_list_%s_%s.txt" % (case_name, arch))
            out.write_text("\n".join(tracks) + "\n")
            report("%s (+%d fill)" % (arch, len(fill)), tracks, args.n_fad, rejected)
            print("    -> %s" % out)
            if len(tracks) < 512:
                print("      under 512 clips: CLAP-FAD (512-D) will be singular here.\n"
                      "      Read EnCodec-FAD (128-D) instead, or widen the pool.")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.pool_dir:
        pooled_lists(args, out_dir)
    else:
        legacy_lists(args, out_dir)


if __name__ == "__main__":
    main()
