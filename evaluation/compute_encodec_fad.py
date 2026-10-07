"""FAD on raw EnCodec latents (mean-pooled over time).

Reuses frechet_distance from compute_clap_fad.py. Does NOT normalise latents
with ensemble mean/std — that would erase the cross-ensemble difference FAD
is meant to measure.

Keep CLAP-FAD (compute_clap_fad.py) alongside; never compare the two scales.

This is the paper's FAD and the one to lead with: 128 dimensions against CLAP's
512, so the covariance it fits is well determined by a few hundred clips where
CLAP needs a thousand or more.

--out-root/--cases picks up whatever compute_clap_fad_200.py already built --
the same reference set and the same generated clips, so the two FADs describe
the same audio and only the embedder differs.

  python evaluation/compute_encodec_fad.py --out-root Outputs/WAV_files/metrics_200
  python evaluation/compute_encodec_fad.py --real-dir <dir> --gen-dirs no_midi=<dir>
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
    p.add_argument("--out-root", type=Path, default=None,
                   help="discover <out-root>/fad/<case>/reference and "
                        ".../generated/<experiment>, as compute_clap_fad_200.py "
                        "lays them out. Overrides --real-dir/--gen-dirs.")
    p.add_argument("--cases", nargs="+", default=["poly", "mono"])
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
    p.add_argument("--no-peak-normalize", action="store_true",
                   help="leave clip levels alone. The default rescales every clip "
                        "to 0.95 peak, which keeps FAD from being driven by loudness "
                        "but also hides a real output-level difference between "
                        "models. Both sides get the same treatment either way.")
    p.add_argument("--no-cache", action="store_true",
                   help="re-embed even when a cached .npz matches the file list")
    return p.parse_args()


def discover(out_root, cases):
    """[(case, real_dir, {experiment: gen_dir})] from the CLAP-FAD layout."""
    found = []
    for case in cases:
        fad_dir = Path(out_root) / "fad" / case
        real_dir = fad_dir / "reference"
        if not real_dir.is_dir():
            print("no %s -- skipping case %s" % (real_dir, case))
            continue
        gen = {d.name: d for d in sorted((fad_dir / "generated").glob("*")) if d.is_dir()}
        if not gen:
            print("no generated clips under %s -- skipping case %s" % (fad_dir / "generated", case))
            continue
        found.append((case, real_dir, gen))
    if not found:
        raise SystemExit("nothing to score under %s. Run the bridge and "
                         "compute_clap_fad_200.py first." % out_root)
    return found


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


def load_audio_24k(path: Path, peak_normalize=True) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if peak_normalize:
        peak = float(wav.abs().max()) + 1e-9
        wav = wav / peak * 0.95
    return wav


@torch.no_grad()
def get_encodec_embeddings(audio_dir: Path, enc, device, limit=None,
                           peak_normalize=True, cache=True) -> np.ndarray:
    """[n, 128] raw mean-pooled latents, cached beside the audio by file list.

    The cache is keyed on the sorted file names, so adding a clip invalidates it
    and nothing silently mixes two different sets.
    """
    paths = sorted(audio_dir.glob("*.wav"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError("No wav files in %s" % audio_dir)

    names = np.array([p.name for p in paths])
    cache_path = audio_dir.parent / ("%s_encodec.npz" % audio_dir.name)
    if cache and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if list(cached["names"]) == list(names):
            print("  cached: %d embeddings from %s" % (cached["emb"].shape[0], cache_path))
            return cached["emb"]

    embeddings = []
    for i, path in enumerate(paths, start=1):
        wav = load_audio_24k(path, peak_normalize).unsqueeze(0).to(device)  # [1, 1, N]
        # mean=None, std=None -> raw latent, no per-ensemble normalisation
        z = enc.encode_latent(wav, None, None)  # [1, 128, T]
        vec = z.mean(dim=-1).squeeze(0).detach().cpu().numpy()  # [128]
        embeddings.append(vec)
        if i % 100 == 0 or i == len(paths):
            print("  encoded %d/%d" % (i, len(paths)), flush=True)
    emb = np.stack(embeddings, axis=0)
    if cache:
        np.savez(cache_path, emb=emb, names=names)
    return emb


def main():
    args = parse_args()
    if args.out_root:
        jobs = discover(args.out_root, args.cases)
    else:
        jobs = [("", args.real_dir, parse_gen_dirs(args.gen_dirs))]

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    from audio_diffusion_pytorch import NormalizedEncodec

    print("Device:", device)
    print("peak normalisation:", "off" if args.no_peak_normalize else "on (0.95)")
    enc = NormalizedEncodec(device=device)

    def embed(d):
        return get_encodec_embeddings(d, enc, device, limit=args.limit,
                                      peak_normalize=not args.no_peak_normalize,
                                      cache=not args.no_cache)

    results = []
    for case, real_dir, gen_dirs in jobs:
        print("=" * 70)
        print("case %s | reference %s" % (case or "(explicit dirs)", real_dir))
        print("=" * 70, flush=True)
        real_emb = embed(real_dir)

        # Sanity: real against real, same distribution, should land near 0. If it
        # does not, n is too small for the covariance and no other row is readable.
        if real_emb.shape[0] >= 4:
            mid = real_emb.shape[0] // 2
            sanity = frechet_distance(real_emb[:mid], real_emb[mid:])
            print("  sanity, reference half vs half (n=%d each): %.6f" % (mid, sanity))
            results.append({
                "case": case, "experiment": "sanity_real_halves",
                "reference": str(real_dir),
                "embedding_model": "facebook/encodec_24khz_raw_meanpool",
                "n_reference": mid, "n_generated": real_emb.shape[0] - mid,
                "fad": sanity,
            })

        for name, gen_dir in gen_dirs.items():
            gen_emb = embed(gen_dir)
            score = frechet_distance(real_emb, gen_emb)
            results.append({
                "case": case, "experiment": name,
                "reference": str(real_dir),
                "embedding_model": "facebook/encodec_24khz_raw_meanpool",
                "n_reference": int(real_emb.shape[0]),
                "n_generated": int(gen_emb.shape[0]),
                "fad": score,
            })
            print("  %-40s EnCodec-FAD %.6f  (n_ref %d, n_gen %d)"
                  % (name, score, real_emb.shape[0], gen_emb.shape[0]), flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print("Saved", args.out_csv)


if __name__ == "__main__":
    main()
