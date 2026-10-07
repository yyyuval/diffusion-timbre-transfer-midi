"""Pick the fixed evaluation tracks for the 200-clip runs.

  poly: woodwind tracks with 1_flute + 4_bassoon   (flute_bassoon -> violin_cello)
  mono: string tracks with 4_cello                  (cello -> bassoon)

Deterministic: candidates are sorted, then shuffled with --seed, and the first
--n that pass the checks are kept. A track passes when every input stem has a
non-empty wav AND its stems_midi/*.mid, and the MIDI has at least one note in
the 17 s window the bridge actually sees (a clip with no reference notes makes
the content metrics degenerate).

  python evaluation/make_track_lists.py
  python evaluation/make_track_lists.py --n 50 --cases mono
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
    stem_paths,
    track_has_files,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default=DATASET_ROOT)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cases", nargs="+", default=["poly", "mono"], choices=["poly", "mono"])
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


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_name in args.cases:
        case = CASES[case_name]
        candidates = list_family_tracks(args.dataset_root, case["family"])
        accept, rejected = make_accept(args.dataset_root, case["stems"], not args.skip_midi_check)
        picked = pick_tracks(candidates, args.n, args.seed, accept=accept)

        out = out_dir / ("track_list_%d_%s.txt" % (args.n, case_name))
        out.write_text("\n".join(picked) + "\n")
        print("%s: %d %s_track* folders, picked %d (seed %d), rejected %d on the way"
              % (case_name, len(candidates), case["family"], len(picked), args.seed, len(rejected)))
        for track, reason in rejected[:10]:
            print("    rejected %s: %s" % (track, reason))
        print("  -> %s" % out)


if __name__ == "__main__":
    main()
