"""Merge several MIDI files into one -- the MIDI a score actually gives you.

The polyphonic models have so far been conditioned on one roll per instrument,
stacked, which tells them which instrument plays which note. Nobody has that
outside this dataset: a score, or a transcription of a recording, is one MIDI
for the whole ensemble. This builds that MIDI from the dataset's per-stem
ground truth, so a model can be trained on what would really be available.

Every note of every input goes into a single instrument track. Two notes of the
same pitch that overlap in time become one note spanning both; notes that only
touch stay separate, so a re-articulation is kept. The rendered roll is then
exactly the element-wise OR of the inputs' rolls, and dataset mode checks that
on every track.

Two inputs, one output:

    python midi_preprocessing/merge_midi.py --inputs 1_flute.mid 4_bassoon.mid --out merged.mid

A whole ensemble -- writes <track>/merged_flute_bassoon.mid under --midi_out_root
and <track>/merged_flute_bassoon_pianoroll.npy under --cache_root, which is what
training reads with midi_merged=true:

    python midi_preprocessing/merge_midi.py --dataset_root $D --instrument flute bassoon \\
        --cache_root midi_cache_wood_fps75_gt_merged --midi_out_root midi_gt_merged/wood
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Note = Tuple[int, float, float, int]  # pitch, start (s), end (s), velocity


def merge_notes(notes: Iterable[Note]) -> List[Note]:
    """Union of notes. Same pitch and overlapping in time -> one note.

    Overlap is strict (the next note starts before the current one ends), so
    two notes that merely touch remain two. Zero-length notes are dropped:
    they render to nothing and only confuse a reader of the file. The fused
    note keeps the louder velocity.
    """
    by_pitch: Dict[int, List[Tuple[float, float, int]]] = {}
    for pitch, start, end, velocity in notes:
        if end <= start:
            continue
        by_pitch.setdefault(int(pitch), []).append((float(start), float(end), int(velocity)))

    merged: List[Note] = []
    for pitch in sorted(by_pitch):
        current = None
        for start, end, velocity in sorted(by_pitch[pitch]):
            if current is not None and start < current[1]:
                current = (current[0], max(current[1], end), max(current[2], velocity))
            else:
                if current is not None:
                    merged.append((pitch,) + current)
                current = (start, end, velocity)
        merged.append((pitch,) + current)
    merged.sort(key=lambda n: (n[1], n[0]))
    return merged


def read_notes(pm) -> List[Note]:
    notes: List[Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend((n.pitch, n.start, n.end, n.velocity) for n in inst.notes)
    return notes


def merge_midi(paths: Sequence[Path]):
    """Read the inputs, return one PrettyMIDI with a single merged track.

    Built on a copy of the first input so its tempo map and resolution carry
    over: note times then go back to the same ticks they were read from, and a
    rewritten file renders to the same frames rather than drifting by one where
    a note sits near a frame boundary.
    """
    import pretty_midi

    pms = [pretty_midi.PrettyMIDI(str(p)) for p in paths]
    merged = copy.deepcopy(pms[0])
    merged.instruments = []
    track = pretty_midi.Instrument(program=0, name="merged")
    for pitch, start, end, velocity in merge_notes(n for pm in pms for n in read_notes(pm)):
        track.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    merged.instruments.append(track)
    return merged, pms


def render_roll(pm, fps: int, velocity_threshold: float) -> np.ndarray:
    """The same two lines every cache builder here uses."""
    return (pm.get_piano_roll(fs=fps) > velocity_threshold).astype(np.uint8)


def or_of_rolls(rolls: Sequence[np.ndarray]) -> np.ndarray:
    width = max(r.shape[1] for r in rolls)
    out = np.zeros((128, width), dtype=np.uint8)
    for r in rolls:
        out[:, : r.shape[1]] |= r
    return out


def rolls_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal once padded to a common width -- trailing silence is not a difference."""
    width = max(a.shape[1], b.shape[1])
    pad = lambda r: np.pad(r, ((0, 0), (0, width - r.shape[1])))
    return bool(np.array_equal(pad(a), pad(b)))


# ------------------------------------------------------------------ pair mode
def merge_pair(args) -> None:
    merged, _ = merge_midi([Path(p) for p in args.inputs])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(out))
    n = len(merged.instruments[0].notes)
    print("merged %d files -> %s  (%d notes)" % (len(args.inputs), out, n))
    if args.roll:
        roll = render_roll(merged, args.fps, args.velocity_threshold)
        Path(args.roll).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.roll, roll)
        print("roll %s -> %s" % (roll.shape, args.roll))


# --------------------------------------------------------------- dataset mode
def merge_dataset(args) -> None:
    from audio_data_pytorch.datasets.wav_dataset import (
        get_paired_stem_filenames,
        merged_midi_name,
    )

    # Enumerate exactly as training does, so the cache covers what will be
    # asked for -- including random_track* folders that number their stems
    # differently from the ensemble ones.
    groups = get_paired_stem_filenames([args.dataset_root], args.instrument)
    if args.limit is not None:
        groups = groups[: args.limit]
    if not groups:
        sys.exit("No track has all of %s under %s." % (args.instrument, args.dataset_root))

    name = merged_midi_name(args.instrument)
    cache_root = Path(args.cache_root)
    midi_out_root = Path(args.midi_out_root) if args.midi_out_root else None
    print("%d tracks with all of %s -> %s_pianoroll.npy  (fps %d, threshold %g)"
          % (len(groups), args.instrument, name, args.fps, args.velocity_threshold))

    written = skipped = missing = failed = 0
    mismatched: List[str] = []
    missing_examples: List[str] = []

    for i, group in enumerate(groups, start=1):
        stem_wavs = [Path(p) for p in group]
        track = stem_wavs[0].parent.parent.name
        roll_path = cache_root / track / (name + "_pianoroll.npy")
        if roll_path.exists() and not args.overwrite:
            skipped += 1
            continue

        midi_paths = [w.parent.parent / "stems_midi" / (w.stem + ".mid") for w in stem_wavs]
        absent = [m for m in midi_paths if not m.exists()]
        if absent:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(str(absent[0]))
            continue

        try:
            merged, pms = merge_midi(midi_paths)
            roll = render_roll(merged, args.fps, args.velocity_threshold)

            if not args.no_check:
                expected = or_of_rolls([render_roll(pm, args.fps, args.velocity_threshold)
                                        for pm in pms])
                if not rolls_equal(roll, expected):
                    mismatched.append(track)
                    continue

            roll_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(roll_path, roll)
            if midi_out_root is not None:
                mid = midi_out_root / track / (name + ".mid")
                mid.parent.mkdir(parents=True, exist_ok=True)
                merged.write(str(mid))
            written += 1
        except Exception as e:  # one bad file must not end a 9,000-track run
            failed += 1
            print("[%d/%d] FAILED %s: %s" % (i, len(groups), track, e))

        if i % 500 == 0 or i == len(groups):
            print("[%d/%d] %s" % (i, len(groups), track))

    print("\nDone")
    print("  written    : %d" % written)
    print("  skipped    : %d  (already cached)" % skipped)
    print("  missing    : %d  (a stem had no stems_midi/*.mid)" % missing)
    print("  failed     : %d" % failed)
    if not args.no_check:
        print("  mismatched : %d  (merged roll != OR of the stems' rolls -- NOT written)"
              % len(mismatched))
        for t in mismatched[:5]:
            print("    %s" % t)
    for m in missing_examples:
        print("  missing e.g. %s" % m)

    covered = written + skipped
    print("\ncoverage: %d/%d tracks (%.1f%%)" % (covered, len(groups), 100.0 * covered / len(groups)))
    if covered < len(groups):
        print("Every track without a roll trains on an all-zero roll. Fix this before "
              "launching a run on this cache.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", help="pair mode: the .mid files to merge")
    p.add_argument("--out", help="pair mode: the merged .mid to write")
    p.add_argument("--roll", help="pair mode: also write the rendered roll as .npy")
    p.add_argument("--dataset_root", help="dataset mode: CocoChorales main_dataset")
    p.add_argument("--instrument", nargs="+", help="dataset mode: e.g. flute bassoon")
    p.add_argument("--cache_root", help="dataset mode: where the merged rolls go")
    p.add_argument("--midi_out_root", default=None,
                   help="dataset mode: also keep the merged .mid per track here")
    p.add_argument("--fps", type=int, default=75,
                   help="must match midi_fps in exp/datamodule/base.yaml")
    p.add_argument("--velocity_threshold", type=float, default=0.0,
                   help="CocoChorales velocities are a flat 64, so 0 keeps every note")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_check", action="store_true",
                   help="skip comparing each merged roll with the OR of the stems' rolls")
    args = p.parse_args()

    if args.inputs:
        if len(args.inputs) < 2 or not args.out:
            p.error("pair mode needs --inputs with two or more files and --out")
        merge_pair(args)
    else:
        if not (args.dataset_root and args.instrument and args.cache_root):
            p.error("dataset mode needs --dataset_root, --instrument and --cache_root "
                    "(or use --inputs/--out for a single pair)")
        if len(args.instrument) < 2:
            p.error("--instrument needs two or more names; one instrument has nothing to merge")
        merge_dataset(args)


if __name__ == "__main__":
    main()
