"""
Compare the dataset's ground-truth MIDI against what basic-pitch transcribed.

Both are .mid files, so they are trivially "the same format". The question that
actually matters is whether they are interchangeable *in this pipeline* --
same layout, same timing, same pitch range, and above all the same velocity
semantics, because --velocity_threshold is applied to velocities.

Run it on one track. It prints the two side by side and flags anything that
would make a swap unsafe.

    python debug_midi/compare_midi_sources.py \
      --track /dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/cocochorales_tiny_v1_zipped/main_dataset/string_track000534 \
      --instrument cello
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pretty_midi


FPS = 75                 # must match midi_fps in exp/datamodule/base.yaml
THRESHOLD = 40.0         # the --velocity_threshold the caches were built with


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def rule(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def describe_midi(label, midi_path: Path):
    """Everything about a .mid that could break a swap."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        print(f"{label}: NO NOTES in {midi_path}")
        return pm, None

    pitches = np.array([n.pitch for n in notes])
    vels = np.array([n.velocity for n in notes])
    durs = np.array([n.end - n.start for n in notes])

    print(f"{label}")
    print(f"  file            {midi_path}")
    print(f"  instruments     {len(pm.instruments)}"
          f"   programs={[i.program for i in pm.instruments]}"
          f"   drums={[i.is_drum for i in pm.instruments]}")
    print(f"  notes           {len(notes)}")
    print(f"  duration        {pm.get_end_time():.2f} s")
    print(f"  pitch           {pitches.min()}-{pitches.max()}"
          f"   ({pretty_midi.note_number_to_name(int(pitches.min()))}"
          f"-{pretty_midi.note_number_to_name(int(pitches.max()))})")
    print(f"  velocity        min={vels.min()}  mean={vels.mean():.1f}  max={vels.max()}")
    print(f"  note length     median={np.median(durs):.3f} s"
          f"   shortest={durs.min():.3f} s")

    below = int((vels < THRESHOLD).sum())
    print(f"  velocity < {THRESHOLD:.0f}     {below}/{len(notes)} notes"
          f"  ({100.0 * below / len(notes):.1f}%)  <- dropped by the threshold")

    return pm, vels


def render(pm, fps=FPS, threshold=THRESHOLD):
    """Same two lines the cache builders use."""
    velocity_roll = pm.get_piano_roll(fs=fps)
    return (velocity_roll > threshold).astype(np.uint8)


def describe_roll(label, roll):
    if roll is None or roll.size == 0:
        print(f"{label}: empty roll")
        return
    per_frame = roll.sum(axis=0)
    active = int((per_frame > 0).sum())
    print(f"{label}")
    print(f"  shape           {roll.shape}   ({roll.shape[1] / FPS:.2f} s at {FPS} fps)")
    print(f"  frames w/ notes {active}/{roll.shape[1]}"
          f"  ({100.0 * active / roll.shape[1]:.1f}%)")
    print(f"  notes per frame mean={per_frame.mean():.2f}  max={int(per_frame.max())}")
    print(f"  distinct pitches {int((roll.sum(axis=1) > 0).sum())}")


def compare_rolls(a, b, label_a, label_b):
    """How much would the model's input actually change if we swapped?"""
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        print("  cannot compare: one roll is empty")
        return
    a, b = a[:, :n].astype(bool), b[:, :n].astype(bool)

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = inter / union if union else 1.0

    only_a = int(np.logical_and(a, ~b).sum())
    only_b = int(np.logical_and(b, ~a).sum())

    print(f"  overlap (IoU)   {iou:.3f}   1.000 would mean identical")
    print(f"  only in {label_a:<9} {only_a} cells")
    print(f"  only in {label_b:<9} {only_b} cells")
    if a.shape[1] != b.shape[1]:
        print(f"  NOTE lengths differ, compared first {n} frames only")
    return iou


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True,
                   help="one track folder in the dataset")
    p.add_argument("--instrument", default="cello")
    p.add_argument("--basic_pitch_midi", default=None,
                   help="the basic-pitch .mid for this stem "
                        "(default: look under midi_outputs/cache_temp_midis*)")
    p.add_argument("--cached_npy", default=None,
                   help="the .npy the cache builder produced "
                        "(default: midi_cache_<instrument>_fps75/<track>/...)")
    p.add_argument("--repo", default="/home/shared_workspace/diffusion-timbre-transfer")
    args = p.parse_args()

    track = Path(args.track)
    repo = Path(args.repo)
    inst = args.instrument
    track_name = track.name

    # ---- what is actually on disk -------------------------------------
    rule("1. LAYOUT  -- do the filenames line up?")

    audio_dir = track / "stems_audio"
    midi_dir = track / "stems_midi"

    print(f"stems_audio/  {sorted(f.name for f in audio_dir.glob('*')) if audio_dir.is_dir() else 'MISSING'}")
    print(f"stems_midi/   {sorted(f.name for f in midi_dir.glob('*')) if midi_dir.is_dir() else 'MISSING'}")

    if not midi_dir.is_dir():
        sys.exit("\nNo stems_midi/ in this track -- nothing to compare.")

    wavs = sorted(audio_dir.glob(f"*_{inst}.wav"))
    gts = sorted(midi_dir.glob(f"*_{inst}*.mid")) or sorted(midi_dir.glob(f"*{inst}*.mid"))

    print(f"\n{inst} audio stems : {[w.name for w in wavs] or 'NONE'}")
    print(f"{inst} midi  stems : {[m.name for m in gts] or 'NONE'}")

    if not gts:
        print(f"\nNo '{inst}' in the stem MIDI filenames. All of stems_midi/ is listed")
        print("above -- the naming may differ (e.g. by part number only).")
        sys.exit(1)
    if not wavs:
        sys.exit(f"\nNo {inst} wav in this track.")

    gt_midi = gts[0]
    wav = wavs[0]
    print(f"\npairing: {wav.name}  <->  {gt_midi.name}")

    # ---- the two MIDI files -------------------------------------------
    rule("2. THE TWO .mid FILES")

    gt_pm, gt_vels = describe_midi("GROUND TRUTH (dataset)", gt_midi)

    bp_path = Path(args.basic_pitch_midi) if args.basic_pitch_midi else None
    if bp_path is None:
        for parent in ("cache_temp_midis_fps75", "cache_temp_midis",
                       "cache_temp_midis_bassoon_fps75"):
            hits = sorted((repo / "midi_outputs" / parent / track_name)
                          .glob(f"{wav.stem}*_basic_pitch.mid"))
            if hits:
                bp_path = hits[0]
                break

    bp_pm = None
    print()
    if bp_path and bp_path.exists():
        bp_pm, _ = describe_midi("BASIC-PITCH (transcribed)", bp_path)
    else:
        print("BASIC-PITCH (transcribed)")
        print("  not found -- pass --basic_pitch_midi, or skip (the cached .npy")
        print("  below is still compared, which is what training actually reads)")

    # ---- rendered to the grid the model sees --------------------------
    rule(f"3. RENDERED TO PIANO ROLLS  (fps={FPS}, threshold={THRESHOLD:.0f})")

    gt_roll = render(gt_pm)
    describe_roll("GROUND TRUTH", gt_roll)

    bp_roll = None
    if bp_pm is not None:
        print()
        bp_roll = render(bp_pm)
        describe_roll("BASIC-PITCH", bp_roll)

    npy = Path(args.cached_npy) if args.cached_npy else (
        repo / f"midi_cache_{inst}_fps75" / track_name / f"{wav.stem}_pianoroll.npy")
    cached = None
    print()
    if npy.exists():
        cached = np.load(npy)
        describe_roll(f"CACHED .npy  ({npy.name})", cached)
    else:
        print(f"CACHED .npy       not found at {npy}")

    # ---- would swapping change the input? -----------------------------
    if cached is not None:
        rule("4. WOULD A SWAP CHANGE WHAT THE MODEL SEES?")
        print("ground truth  vs  the cached roll training reads today:")
        compare_rolls(gt_roll, cached, "truth", "cached")

    # ---- the thing most likely to bite --------------------------------
    rule("5. VERDICT")

    if gt_vels is not None:
        lost = int((gt_vels < THRESHOLD).sum())
        pct = 100.0 * lost / len(gt_vels)
        print(f"velocity threshold {THRESHOLD:.0f} would delete {lost}/{len(gt_vels)}"
              f" ground-truth notes ({pct:.1f}%).")
        if pct > 1.0:
            print("  ^ THIS MATTERS. Ground-truth velocities are musical dynamics,")
            print("    not confidence scores. Quiet notes are real notes. Rebuild")
            print("    from ground truth with --velocity_threshold 0.")
        else:
            print("  (low -- but still prefer threshold 0 for ground truth)")

    gt_dur = gt_pm.get_end_time()
    print(f"\nground-truth MIDI duration: {gt_dur:.2f} s")
    print("compare against the wav duration; a large gap means the two are not")
    print("aligned and the crop offset maths would be wrong (see Bug 10).")

    print("\nIf the pairing above is right, the rolls broadly agree, and the only")
    print("real difference is the velocity threshold, the swap is safe.")


if __name__ == "__main__":
    main()
