"""FAD on raw EnCodec latents (mean-pooled over time).

Reuses frechet_distance from compute_clap_fad.py. Does NOT normalise latents
with ensemble mean/std — that would erase the cross-ensemble difference FAD
is meant to measure.

Keep CLAP-FAD (compute_clap_fad.py) alongside; never compare the two scales.

  python evaluation/compute_encodec_fad.py
  python evaluation/compute_encodec_fad.py --real-dir Outputs/.../fad/real_violin_cello \\
      --gen-dirs no_midi=Outputs/.../fad/generated_no_midi \\
                 original_midi=Outputs/.../fad/generated_original_midi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.compute_clap_fad import frechet_distance  # noqa: E402

FAD_ROOT = Path("Outputs/WAV_files/metrics/fad")
DEFAULT_REAL = FAD_ROOT / "real_violin_cello"
DEFAULT_GEN = {
    "no_midi": FAD_ROOT / "generated_no_midi",
    "original_midi": FAD_ROOT / "generated_original_midi",
}
OUT_CSV = FAD_ROOT / "encodec_fad_results.csv"
SAMPLE_RATE = 24000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", type=Path, default=DEFAULT_REAL)
    p.add_argument(
        "--gen-dirs",
        nargs="*",
        default=None,
        help="name=path pairs, e.g. no_midi=Outputs/.../generated_no_midi",
    )
    p.add_argument("--out-csv", type=Path, default=OUT_CSV)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def parse_gen_dirs(items):
    if not items:
        return dict(DEFAULT_GEN)
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--gen-dirs entries must be name=path, got %r" % item)
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def load_audio_24k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    peak = float(wav.abs().max()) + 1e-9
    wav = wav / peak * 0.95
    return wav


@torch.no_grad()
def get_encodec_embeddings(audio_dir: Path, enc, device, limit=None) -> np.ndarray:
    paths = sorted(audio_dir.glob("*.wav"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError("No wav files in %s" % audio_dir)

    embeddings = []
    for i, path in enumerate(paths, start=1):
        print("[%d/%d] encode %s" % (i, len(paths), path.name))
        wav = load_audio_24k(path).unsqueeze(0).to(device)  # [1, 1, N]
        # mean=None, std=None -> raw latent, no per-ensemble normalisation
        z = enc.encode_latent(wav, None, None)  # [1, 128, T]
        vec = z.mean(dim=-1).squeeze(0).detach().cpu().numpy()  # [128]
        embeddings.append(vec)
    return np.stack(embeddings, axis=0)


def main():
    args = parse_args()
    gen_dirs = parse_gen_dirs(args.gen_dirs)

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    from audio_diffusion_pytorch import NormalizedEncodec

    print("Device:", device)
    enc = NormalizedEncodec(device=device)

    print("\nReference embeddings:", args.real_dir)
    real_emb = get_encodec_embeddings(args.real_dir, enc, device, limit=args.limit)

    # sanity: real vs real should be ~0 (split halves if enough clips)
    results = []
    if real_emb.shape[0] >= 4:
        mid = real_emb.shape[0] // 2
        sanity = frechet_distance(real_emb[:mid], real_emb[mid:])
        print("Sanity real-vs-real (halves) FAD: %.6f" % sanity)
        results.append(
            {
                "experiment": "sanity_real_halves",
                "reference": str(args.real_dir),
                "embedding_model": "facebook/encodec_24khz_raw_meanpool",
                "n_reference": mid,
                "n_generated": real_emb.shape[0] - mid,
                "fad": sanity,
            }
        )

    for name, gen_dir in gen_dirs.items():
        print("=" * 70)
        print("EnCodec-FAD:", name, gen_dir)
        print("=" * 70)
        gen_emb = get_encodec_embeddings(gen_dir, enc, device, limit=args.limit)
        score = frechet_distance(real_emb, gen_emb)
        results.append(
            {
                "experiment": name,
                "reference": str(args.real_dir),
                "embedding_model": "facebook/encodec_24khz_raw_meanpool",
                "n_reference": int(real_emb.shape[0]),
                "n_generated": int(gen_emb.shape[0]),
                "fad": score,
            }
        )
        print("FAD = %.6f" % score)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print("Saved", args.out_csv)


if __name__ == "__main__":
    main()
