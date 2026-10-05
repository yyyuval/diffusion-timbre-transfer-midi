from pathlib import Path
import pandas as pd
import torch
import torchaudio

DATASET_ROOT = Path(
    "/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/"
    "cocochorales_tiny_v1_zipped/main_dataset"
)

METRICS_CSV = Path("Outputs/WAV_files/metrics/inference_metrics.csv")
FAD_ROOT = Path("Outputs/WAV_files/metrics/fad")

REAL_DIR = FAD_ROOT / "real_violin_cello"
GEN_NO_MIDI_DIR = FAD_ROOT / "generated_no_midi"
GEN_ORIGINAL_MIDI_DIR = FAD_ROOT / "generated_original_midi"

SAMPLE_RATE = 24000


def load_audio(path: Path):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def match_length(wavs):
    min_len = min(w.shape[-1] for w in wavs)
    return [w[..., :min_len] for w in wavs]


def peak_normalize(wav, eps=1e-8):
    peak = wav.abs().max()
    if peak > eps:
        wav = wav / peak * 0.95
    return wav


def prepare_generated():
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {METRICS_CSV}")

    df = pd.read_csv(METRICS_CSV)

    for d in [GEN_NO_MIDI_DIR, GEN_ORIGINAL_MIDI_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        experiment = row["experiment"]
        sample_id = row["sample_id"]
        src = Path(row["output_audio"])

        if not src.exists():
            raise FileNotFoundError(f"Generated audio not found: {src}")

        if experiment == "no_midi":
            dst_dir = GEN_NO_MIDI_DIR
        elif experiment == "original_midi":
            dst_dir = GEN_ORIGINAL_MIDI_DIR
        else:
            print(f"Skipping unknown experiment: {experiment}")
            continue

        dst = dst_dir / f"{sample_id}_{experiment}.wav"

        wav = load_audio(src)
        torchaudio.save(str(dst), wav.cpu(), SAMPLE_RATE)

    print(f"Prepared generated no_midi:        {len(list(GEN_NO_MIDI_DIR.glob('*.wav')))} files")
    print(f"Prepared generated original_midi:  {len(list(GEN_ORIGINAL_MIDI_DIR.glob('*.wav')))} files")


def prepare_real_reference(n_files: int):
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    string_tracks = sorted(DATASET_ROOT.glob("string_track*/stems_audio"))

    made = 0
    for stems_dir in string_tracks:
        violin = stems_dir / "1_violin.wav"
        cello = stems_dir / "4_cello.wav"

        if not violin.exists() or not cello.exists():
            continue

        violin_wav = load_audio(violin)
        cello_wav = load_audio(cello)

        violin_wav, cello_wav = match_length([violin_wav, cello_wav])
        mix = peak_normalize(violin_wav + cello_wav)

        track_name = stems_dir.parent.name
        out_path = REAL_DIR / f"{track_name}_real_violin_cello.wav"
        torchaudio.save(str(out_path), mix.cpu(), SAMPLE_RATE)

        made += 1
        if made >= n_files:
            break

    print(f"Prepared real violin_cello reference: {made} files")

    if made < n_files:
        raise RuntimeError(f"Only created {made} real reference files, requested {n_files}")


def main():
    prepare_generated()

    n_no_midi = len(list(GEN_NO_MIDI_DIR.glob("*.wav")))
    n_original = len(list(GEN_ORIGINAL_MIDI_DIR.glob("*.wav")))
    n_ref = min(n_no_midi, n_original)

    if n_ref == 0:
        raise RuntimeError("No generated files found; cannot prepare FAD reference size.")

    prepare_real_reference(n_ref)

    print("\nFAD folders:")
    print(f"  real:          {REAL_DIR}")
    print(f"  no_midi:       {GEN_NO_MIDI_DIR}")
    print(f"  original_midi: {GEN_ORIGINAL_MIDI_DIR}")


if __name__ == "__main__":
    main()
