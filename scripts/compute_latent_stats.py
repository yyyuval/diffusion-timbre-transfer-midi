"""
Compute the Encodec latent mean/std for a dataset, and save them in the same
format as checkpoints/mean_*.pt and std_*.pt.

Why this is needed for two-instrument training: Model.encode_latent z-normalises
the latent with per-channel stats loaded from those .pt files, and the training
config then asserts diffusion_sigma_data: 1 -- i.e. "the normalised latent has
unit variance". A mixture of two instruments has a different latent distribution
from either instrument alone, so reusing mean_cello.pt for a violin+cello mixture
leaves the noise schedule miscalibrated.

The repo ships no script for this (the existing tensors came from Zenodo), so
here is one. It reuses WAVDataset, which means it sees exactly the same audio the
training run will see, mixture logic and all.

    # single instrument (reproduces the shipped tensors)
    python scripts/compute_latent_stats.py \
      --dataset_root /dsi/.../main_dataset --instruments cello \
      --out_prefix checkpoints/cello_check

    # a mixture
    python scripts/compute_latent_stats.py \
      --dataset_root /dsi/.../main_dataset --mix_instruments violin cello \
      --out_prefix checkpoints/violin_cello
"""

from pathlib import Path
import argparse
import sys

import torch
from torch.utils.data import DataLoader
from transformers import EncodecModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_data_pytorch import WAVDataset          # noqa: E402
from audio_data_pytorch import AllTransform        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--instruments", nargs="*", default=None,
                   help="single-instrument mode, e.g. --instruments cello")
    p.add_argument("--mix_instruments", nargs="*", default=None,
                   help="mixture mode, e.g. --mix_instruments violin cello")
    p.add_argument("--out_prefix", required=True,
                   help="writes <prefix>_mean.pt and <prefix>_std.pt")
    p.add_argument("--num_clips", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--sampling_rate", type=int, default=24000)
    p.add_argument("--length", type=int, default=409600)
    p.add_argument("--encodec_model", type=str, default="facebook/encodec_24khz")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    if not args.instruments and not args.mix_instruments:
        sys.exit("Pass either --instruments or --mix_instruments.")

    device = args.device if torch.cuda.is_available() else "cpu"
    print("device:", device)

    dataset = WAVDataset(
        path=args.dataset_root,
        recursive=True,
        instruments=args.instruments or "",
        mix_instruments=args.mix_instruments,
        sample_rate=args.sampling_rate,
        random_crop_size=args.length,
        # No MIDI needed to measure the audio latent, and skipping it avoids
        # requiring a cache that may not exist yet for a new mixture.
        midi_cache_root=None,
        transforms=AllTransform(random_crop_size=args.length),
    )

    if len(dataset) == 0:
        sys.exit("Dataset is empty -- check --dataset_root and the instrument names.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    encodec = EncodecModel.from_pretrained(args.encodec_model).to(device)
    encodec.requires_grad_(False)
    encodec.eval()
    encoder = encodec.get_encoder()

    # Accumulate in float64: summing ~1000 clips x 1280 frames in float32 loses
    # enough precision to matter for a variance computed as E[x^2] - E[x]^2.
    total = None
    total_sq = None
    count = 0
    seen = 0

    with torch.no_grad():
        for batch in loader:
            waveform = batch[0].to(device)
            z = encoder(waveform)                       # [B, 128, T]

            zf = z.double()
            s = zf.sum(dim=(0, 2))                      # [128]
            s2 = (zf ** 2).sum(dim=(0, 2))              # [128]

            total = s if total is None else total + s
            total_sq = s2 if total_sq is None else total_sq + s2
            count += zf.shape[0] * zf.shape[2]
            seen += zf.shape[0]

            if seen % (args.batch_size * 20) == 0:
                print("  %d/%d clips" % (seen, args.num_clips))
            if seen >= args.num_clips:
                break

    mean = total / count
    var = (total_sq / count) - mean ** 2
    std = var.clamp_min(0).sqrt()

    mean = mean.float().cpu()
    std = std.float().cpu()

    out_mean = Path(args.out_prefix + "_mean.pt")
    out_std = Path(args.out_prefix + "_std.pt")
    out_mean.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean, out_mean)
    torch.save(std, out_std)

    print()
    print("clips used      :", seen)
    print("frames used     :", count)
    print("mean  shape %s  range [%.4f, %.4f]" % (tuple(mean.shape), mean.min(), mean.max()))
    print("std   shape %s  range [%.4f, %.4f]" % (tuple(std.shape), std.min(), std.max()))
    print("saved:", out_mean)
    print("saved:", out_std)

    if float(std.min()) < 1e-6:
        print()
        print("WARNING: at least one channel has near-zero std. Dividing by it in")
        print("encode_latent will blow up. Check the dataset is not mostly silence.")


if __name__ == "__main__":
    main()
