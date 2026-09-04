import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def make_fifth(input_path: Path, output_path: Path, sr: int = 24000, fifth_gain: float = 0.8):
    """
    Creates a simple synthetic polyphonic audio file:
    original signal + pitch-shifted copy by +7 semitones (perfect fifth).
    """

    # Load audio as mono
    y, _ = librosa.load(input_path, sr=sr, mono=True)

    # Pitch shift by perfect fifth: +7 semitones
    y_fifth = librosa.effects.pitch_shift(y, sr=sr, n_steps=7)

    # Match lengths
    min_len = min(len(y), len(y_fifth))
    y = y[:min_len]
    y_fifth = y_fifth[:min_len]

    # Mix original + fifth
    y_mix = y + fifth_gain * y_fifth

    # Normalize to avoid clipping
    max_val = np.max(np.abs(y_mix))
    if max_val > 0:
        y_mix = 0.95 * y_mix / max_val

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y_mix, sr)

    print(f"Saved synthetic fifth polyphony to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic polyphonic audio by adding a perfect fifth."
    )

    parser.add_argument("--input", type=str, required=True, help="Path to input mono wav file")
    parser.add_argument("--output", type=str, required=True, help="Path to output wav file")
    parser.add_argument("--sr", type=int, default=24000, help="Sampling rate")
    parser.add_argument(
        "--fifth_gain",
        type=float,
        default=0.8,
        help="Gain of the pitch-shifted fifth layer"
    )

    args = parser.parse_args()

    make_fifth(
        input_path=Path(args.input),
        output_path=Path(args.output),
        sr=args.sr,
        fifth_gain=args.fifth_gain
    )


if __name__ == "__main__":
    main()