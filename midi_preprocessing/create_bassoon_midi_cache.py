from pathlib import Path
import argparse
import subprocess
import numpy as np
import pretty_midi


def midi_to_pianoroll(midi_path: Path, out_path: Path, fps: int, velocity_threshold: float):
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    velocity_roll = midi_data.get_piano_roll(fs=fps)
    piano_roll = (velocity_roll > velocity_threshold).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, piano_roll)

    return piano_roll.shape


def run_basic_pitch(wav_path: Path, temp_midi_dir: Path) -> Path:
    temp_midi_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "basic-pitch",
        str(temp_midi_dir),
        str(wav_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    midi_files = list(temp_midi_dir.glob(f"{wav_path.stem}*_basic_pitch.mid"))

    if not midi_files:
        midi_files = list(temp_midi_dir.glob("*.mid"))

    if not midi_files:
        raise FileNotFoundError(f"No MIDI file created for {wav_path}")

    return midi_files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default="midi_cache_bassoon_fps75")
    parser.add_argument("--temp_midi_root", type=str, default="midi_outputs/cache_temp_midis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--velocity_threshold", type=float, default=40.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_root = Path(args.cache_root)
    temp_midi_root = Path(args.temp_midi_root)

    wav_files = sorted(dataset_root.glob("*/stems_audio/*_bassoon.wav"))

    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    print(f"Found {len(wav_files)} bassoon wav files")

    success = 0
    skipped = 0
    failed = 0

    for i, wav_path in enumerate(wav_files, start=1):
        track_name = wav_path.parent.parent.name
        out_path = cache_root / track_name / f"{wav_path.stem}_pianoroll.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{i}/{len(wav_files)}] SKIP exists: {out_path}")
            continue

        temp_midi_dir = temp_midi_root / track_name

        try:
            midi_path = run_basic_pitch(wav_path, temp_midi_dir)
            shape = midi_to_pianoroll(
                midi_path=midi_path,
                out_path=out_path,
                fps=args.fps,
                velocity_threshold=args.velocity_threshold,
            )

            success += 1
            print(f"[{i}/{len(wav_files)}] OK {track_name} shape={shape}")

        except Exception as e:
            failed += 1
            print(f"[{i}/{len(wav_files)}] FAIL {track_name}: {e}")

    print("\nDone")
    print(f"success: {success}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()