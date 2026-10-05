from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from scipy import linalg
from transformers import ClapModel, ClapProcessor


FAD_ROOT = Path("Outputs/WAV_files/metrics/fad")

REAL_DIR = FAD_ROOT / "real_violin_cello"
GEN_NO_MIDI_DIR = FAD_ROOT / "generated_no_midi"
GEN_ORIGINAL_MIDI_DIR = FAD_ROOT / "generated_original_midi"

OUT_CSV = FAD_ROOT / "clap_fad_results.csv"

MODEL_NAME = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000


def load_audio(path: Path, sr: int = SAMPLE_RATE):
    audio, _ = librosa.load(str(path), sr=sr, mono=True)

    # avoid huge amplitude differences
    peak = np.max(np.abs(audio)) + 1e-9
    audio = audio / peak * 0.95

    return audio.astype(np.float32)


def get_embeddings(audio_dir: Path, model, processor, device):
    paths = sorted(audio_dir.glob("*.wav"))
    if not paths:
        raise RuntimeError(f"No wav files found in {audio_dir}")

    embeddings = []

    for i, path in enumerate(paths, start=1):
        print(f"[{i}/{len(paths)}] embedding {path.name}")

        audio = load_audio(path)

        inputs = processor(
            audios=audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = model.get_audio_features(**inputs)

        emb = emb.detach().cpu().numpy()[0]
        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)
    return embeddings


def frechet_distance(x, y, eps=1e-6):
    """
    Frechet distance between two Gaussian distributions fitted to embeddings.
    Lower is better.
    """
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)

    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    # numerical stability
    sigma_x = sigma_x + np.eye(sigma_x.shape[0]) * eps
    sigma_y = sigma_y + np.eye(sigma_y.shape[0]) * eps

    diff = mu_x - mu_y

    covmean, _ = linalg.sqrtm(sigma_x @ sigma_y, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = diff @ diff + np.trace(sigma_x + sigma_y - 2 * covmean)
    return float(fad)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading CLAP model: {MODEL_NAME}")

    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    print("\nExtracting real violin+cello embeddings")
    real_emb = get_embeddings(REAL_DIR, model, processor, device)

    results = []

    for experiment, gen_dir in [
        ("no_midi", GEN_NO_MIDI_DIR),
        ("original_midi", GEN_ORIGINAL_MIDI_DIR),
    ]:
        print("=" * 70)
        print(f"Computing CLAP-FAD for {experiment}")
        print("=" * 70)

        gen_emb = get_embeddings(gen_dir, model, processor, device)

        score = frechet_distance(real_emb, gen_emb)

        results.append({
            "experiment": experiment,
            "reference": "real_violin_cello",
            "embedding_model": MODEL_NAME,
            "n_reference": real_emb.shape[0],
            "n_generated": gen_emb.shape[0],
            "clap_fad": score,
        })

        print(f"{experiment} CLAP-FAD: {score:.6f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved:")
    print(f"  {OUT_CSV}")
    print("\nResults:")
    print(df)


if __name__ == "__main__":
    main()
