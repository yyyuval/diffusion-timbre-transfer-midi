from pathlib import Path
import argparse
import numpy as np
import pretty_midi


def midi_to_pianoroll(
    midi_path: Path,
    out_path: Path,
    fps: int = 50,
    velocity_threshold: float = 40.0,
):
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    # shape: [128, T], values are MIDI velocities
    velocity_roll = midi_data.get_piano_roll(fs=fps)

    # Keep only strong/confident note activations
    piano_roll = (velocity_roll > velocity_threshold).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, piano_roll)

    active_notes = np.where(piano_roll.sum(axis=1) > 0)[0]
    active_per_frame = piano_roll.sum(axis=0)

    print(f"Saved piano roll to: {out_path}")
    print(f"Shape: {piano_roll.shape}")
    print(f"Velocity threshold: {velocity_threshold}")
    print(f"Active MIDI notes: {active_notes}")
    print(f"Average active notes per frame: {active_per_frame.mean():.3f}")
    print(f"Frames with 0 notes: {np.sum(active_per_frame == 0)}")
    print(f"Frames with 1 note: {np.sum(active_per_frame == 1)}")
    print(f"Frames with 2 notes: {np.sum(active_per_frame == 2)}")
    print(f"Frames with 3+ notes: {np.sum(active_per_frame >= 3)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--velocity_threshold", type=float, default=40.0)
    args = parser.parse_args()

    midi_to_pianoroll(
        midi_path=Path(args.midi),
        out_path=Path(args.out),
        fps=args.fps,
        velocity_threshold=args.velocity_threshold,
    )


if __name__ == "__main__":
    main()