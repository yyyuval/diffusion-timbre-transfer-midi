"""Transfer a custom flute/bassoon mixture to violin/cello without MIDI.

Uses the evaluation bridge and existing trained sigma=5/100 checkpoints.
Long inputs are processed in overlapping model-length windows. Individual
window outputs are saved alongside a crossfaded, original-length output.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def window_starts(n_samples, clip_length):
    """Cover the whole recording with evenly spaced, full-length windows."""
    if n_samples <= 0:
        raise ValueError("The input recording is empty")
    if n_samples <= clip_length:
        return [0]
    count = math.ceil(n_samples / clip_length)
    return [round(i * (n_samples - clip_length) / (count - 1))
            for i in range(count)]


def stitch_windows(windows, starts, n_samples):
    """Crossfade adjacent overlaps without dropping or repeating time."""
    import torch

    result = torch.zeros(1, n_samples)
    weights = torch.zeros(1, n_samples)
    for i, (wav, start) in enumerate(zip(windows, starts)):
        length = min(wav.shape[-1], n_samples - start)
        weight = torch.ones(length)
        if i > 0:
            overlap = max(0, starts[i - 1] + windows[i - 1].shape[-1] - start)
            overlap = min(overlap, length)
            if overlap:
                weight[:overlap] *= torch.linspace(0, 1, overlap)
        if i + 1 < len(starts):
            overlap = max(0, start + length - starts[i + 1])
            overlap = min(overlap, length)
            if overlap:
                weight[-overlap:] *= torch.linspace(1, 0, overlap)
        result[:, start:start + length] += wav[:, :length] * weight
        weights[:, start:start + length] += weight
    if (weights <= 0).any():
        raise RuntimeError("Window stitching left uncovered samples")
    return result / weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="Physical CUDA GPU index")
    parser.add_argument("--sigmas", nargs="+", type=int, choices=[5, 100],
                        default=[5, 100])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate and save input preparation without loading models")
    args = parser.parse_args()

    os.umask(0o002)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    input_path = args.input.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    os.chdir(ROOT)

    import torch
    import torchaudio
    import soundfile as sf
    from evaluation.bridge_200 import CLIP_LENGTH, MODEL_PAIRS, SAMPLING_RATE, resolve_ckpt

    samples, original_sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(samples.T.copy())
    if not torch.isfinite(wav).all():
        raise ValueError("The input contains non-finite samples")
    original_channels = wav.shape[0]
    wav = wav.mean(dim=0, keepdim=True)
    if original_sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, original_sr, SAMPLING_RATE)
    n_samples = wav.shape[-1]
    starts = window_starts(n_samples, CLIP_LENGTH)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Choose a fresh output directory; manifest already exists")
    sf.write(str(out_dir / "input_24k_mono.wav"), wav.squeeze(0).numpy(),
             SAMPLING_RATE, subtype="FLOAT")
    metadata = {
        "input": str(input_path), "original_sample_rate": original_sr,
        "original_channels": original_channels, "sample_rate": SAMPLING_RATE,
        "samples": n_samples, "duration_seconds": n_samples / SAMPLING_RATE,
        "source": "flute_bassoon", "target": "violin_cello", "midi": False,
        "window_samples": CLIP_LENGTH, "window_starts": starts,
        "stitching": "linear crossfade in overlapping windows",
        "seed": 2026, "outputs": [],
    }
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the container and GPU selection")

    from evaluation.run_bridge_batched import Bridge

    for sigma in dict.fromkeys(args.sigmas):
        pair = "no_midi" if sigma == 5 else "no_midi_s100"
        cfg = MODEL_PAIRS[pair]
        source_ckpt = resolve_ckpt(cfg["source_ckpt"], str(ROOT))
        target_ckpt = resolve_ckpt(cfg["target_ckpt"], str(ROOT))
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        bridge = Bridge(pair, source_ckpt, target_ckpt, "cuda:0", None)
        bridge.source.eval()
        bridge.target.eval()
        bridge.encodec.encodec_model.eval()
        generated = []
        for i, start in enumerate(starts):
            x = wav[:, start:start + CLIP_LENGTH]
            x = torch.nn.functional.pad(x, (0, CLIP_LENGTH - x.shape[-1]))
            print(f"Sigma {sigma}: window {i + 1}/{len(starts)}, "
                  f"start {start / SAMPLING_RATE:.3f}s", flush=True)
            with torch.inference_mode():
                z = bridge.encode(x.unsqueeze(0).to("cuda:0"))
                y = bridge.transfer(z, None, sigma).squeeze(0).cpu()
            if y.shape != (1, CLIP_LENGTH) or not torch.isfinite(y).all():
                raise RuntimeError(f"Unexpected generated window: {tuple(y.shape)}")
            generated.append(y)
            sf.write(str(out_dir / f"sigma{sigma}_window{i + 1}.wav"),
                     y.squeeze(0).numpy(), SAMPLING_RATE, subtype="FLOAT")
        full = stitch_windows(generated, starts, n_samples)
        peak = float(full.abs().max())
        output_path = out_dir / f"violin_cello_no_midi_sigma{sigma}.wav"
        # Float WAV preserves the model's amplitude without clipping or normalisation.
        sf.write(str(output_path), full.squeeze(0).numpy(),
                 SAMPLING_RATE, subtype="FLOAT")
        metadata["outputs"].append({
            "sigma": sigma, "path": str(output_path), "peak": peak,
            "source_checkpoint": source_ckpt, "target_checkpoint": target_ckpt,
        })
        manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {output_path}", flush=True)
        del bridge, z, y, generated, full
        gc.collect()
        torch.cuda.empty_cache()
    print("ALL CUSTOM AUDIO STAGES COMPLETED", flush=True)


if __name__ == "__main__":
    main()
