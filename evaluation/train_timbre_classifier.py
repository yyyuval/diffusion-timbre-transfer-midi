"""Multi-label instrument classifier on mean-pooled EnCodec embeddings.

Paper (§III-E): two FC layers 128→64→K on EnCodec embeddings. Here K is the
instrument inventory and outputs are sigmoid (multi-label) so the same model
covers 1→1 and 2→2 transfer.

  # train on real stems (+ synthetic mixtures), validate on held-out reals
  python evaluation/train_timbre_classifier.py --dataset-root ... --epochs 20

  # score generated wavs (requires a saved checkpoint)
  python evaluation/train_timbre_classifier.py --eval-only \\
      --ckpt Outputs/WAV_files/metrics/timbre_classifier.pt \\
      --eval-wav-dir Outputs/WAV_files/metrics/fad/generated_original_midi \\
      --target-instruments violin cello
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 24000
CLIP_LENGTH = 409600

# Filename substring -> label. Prefer underscore names; "double_bass" not "double bass".
DEFAULT_INSTRUMENTS = [
    "violin",
    "viola",
    "cello",
    "double_bass",
    "flute",
    "oboe",
    "clarinet",
    "bassoon",
    "trumpet",
    "horn",
    "trombone",
    "tuba",
    "saxophone",
]


class TimbreClassifier(nn.Module):
    def __init__(self, n_labels: int, in_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/"
            "cocochorales_tiny_v1_zipped/main_dataset"
        ),
    )
    p.add_argument("--instruments", nargs="*", default=DEFAULT_INSTRUMENTS)
    p.add_argument("--exclude-track-lists", nargs="*", type=Path, default=[],
                   help="Exclude these track IDs from classifier training and validation")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-tracks", type=int, default=2000)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--mix-prob", type=float, default=0.5, help="Train on 2-stem mixes this often")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("Outputs/WAV_files/metrics/timbre_classifier.pt"),
    )
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-wav-dir", type=Path, default=None)
    p.add_argument(
        "--target-instruments",
        nargs="*",
        default=None,
        help="For eval-only: which labels should fire (multi-label)",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def instrument_from_stem_name(name: str, instruments: Sequence[str]) -> str | None:
    """Match stem filename stem (e.g. 4_cello) to an instrument key."""
    base = Path(name).stem.lower().replace(" ", "_")
    # strip leading index like 1_ / 4_
    parts = base.split("_", 1)
    candidate = parts[1] if len(parts) == 2 and parts[0].isdigit() else base
    for inst in instruments:
        if candidate == inst or candidate.endswith(inst) or inst in candidate:
            return inst
    return None


def list_stem_files(dataset_root: Path, instruments: Sequence[str], max_tracks: int) -> List[Tuple[Path, str]]:
    tracks = sorted(dataset_root.glob("*_track*"))
    random.shuffle(tracks)
    out: List[Tuple[Path, str]] = []
    for track in tracks:
        stems_dir = track / "stems_audio"
        if not stems_dir.is_dir():
            continue
        for wav in sorted(stems_dir.glob("*.wav")):
            inst = instrument_from_stem_name(wav.name, instruments)
            if inst is None:
                continue
            out.append((wav, inst))
        if len({p.parent.parent for p, _ in out}) >= max_tracks:
            break
    return out


def load_crop(path: Path, device: torch.device) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[-1] >= CLIP_LENGTH:
        wav = wav[..., :CLIP_LENGTH]
    else:
        wav = torch.nn.functional.pad(wav, (0, CLIP_LENGTH - wav.shape[-1]))
    return wav.to(device)


@torch.no_grad()
def embed_wav(enc, wav_1_n: torch.Tensor) -> torch.Tensor:
    z = enc.encode_latent(wav_1_n.unsqueeze(0), None, None)  # [1, 128, T]
    return z.mean(dim=-1).squeeze(0)  # [128]


class EmbeddingCacheDataset(Dataset):
    def __init__(self, items: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor, thr: float) -> float:
    preds = (torch.sigmoid(logits) >= thr).float()
    # exact match subset accuracy
    return float((preds == targets).all(dim=-1).float().mean().item())


def per_label_f1(logits: torch.Tensor, targets: torch.Tensor, thr: float) -> Dict[str, float]:
    preds = (torch.sigmoid(logits) >= thr).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {
        "macro_f1": float(f1.mean().item()),
        "mean_precision": float(prec.mean().item()),
        "mean_recall": float(rec.mean().item()),
    }


def build_embeddings(
    stem_files: List[Tuple[Path, str]],
    instruments: Sequence[str],
    enc,
    device: torch.device,
    mix_prob: float,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    label_to_idx = {n: i for i, n in enumerate(instruments)}
    # group by track for mixtures
    by_track: Dict[Path, List[Tuple[Path, str]]] = {}
    for path, inst in stem_files:
        by_track.setdefault(path.parent.parent, []).append((path, inst))

    items: List[Tuple[torch.Tensor, torch.Tensor]] = []
    tracks = list(by_track.keys())
    for track in tracks:
        stems = by_track[track]
        # always add each solo stem
        for path, inst in stems:
            wav = load_crop(path, device)
            emb = embed_wav(enc, wav).cpu()
            y = torch.zeros(len(instruments))
            y[label_to_idx[inst]] = 1.0
            items.append((emb, y))

        # optional 2-stem mixture from same track
        if len(stems) >= 2 and random.random() < mix_prob:
            (p1, i1), (p2, i2) = random.sample(stems, 2)
            if i1 == i2:
                continue
            w1 = load_crop(p1, device)
            w2 = load_crop(p2, device)
            mix = w1 + w2
            peak = float(mix.abs().max())
            if peak > 0.95:
                mix = 0.95 * mix / peak
            emb = embed_wav(enc, mix).cpu()
            y = torch.zeros(len(instruments))
            y[label_to_idx[i1]] = 1.0
            y[label_to_idx[i2]] = 1.0
            items.append((emb, y))
    return items


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_t = torch.device(device)

    from audio_diffusion_pytorch import NormalizedEncodec

    print("Listing stems…")
    stem_files = list_stem_files(args.dataset_root, args.instruments, args.max_tracks)
    excluded = set()
    for path in args.exclude_track_lists:
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                excluded.add(line.strip().split("/")[0])
    stem_files = [(p, i) for p, i in stem_files if p.parent.parent.name not in excluded]
    print("Excluded evaluation/reference tracks:", len(excluded))
    if len(stem_files) < 50:
        raise SystemExit("Too few stems found (%d). Check --dataset-root." % len(stem_files))
    print("Found %d stem files" % len(stem_files))

    # track-level split so mixes don't leak
    tracks = sorted({p.parent.parent for p, _ in stem_files})
    random.shuffle(tracks)
    n_val = max(1, int(len(tracks) * args.val_fraction))
    val_tracks = set(tracks[:n_val])
    train_stems = [(p, i) for p, i in stem_files if p.parent.parent not in val_tracks]
    val_stems = [(p, i) for p, i in stem_files if p.parent.parent in val_tracks]

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    args.ckpt.with_suffix(".split.json").write_text(json.dumps({
        "excluded_tracks": sorted(excluded),
        "train_tracks": sorted({p.parent.parent.name for p, _ in train_stems}),
        "validation_tracks": sorted({p.parent.parent.name for p, _ in val_stems}),
        "seed": args.seed,
    }, indent=2))
    print("Encoding with EnCodec…")
    enc = NormalizedEncodec(device=device)
    train_items = build_embeddings(train_stems, args.instruments, enc, device_t, args.mix_prob)
    val_items = build_embeddings(val_stems, args.instruments, enc, device_t, mix_prob=0.5)
    print("train embeddings:", len(train_items), "val:", len(val_items))

    model = TimbreClassifier(n_labels=len(args.instruments)).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        EmbeddingCacheDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        EmbeddingCacheDataset(val_items),
        batch_size=args.batch_size,
        shuffle=False,
    )

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device_t), y.to(device_t)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
        train_loss = total / max(1, len(train_items))

        model.eval()
        logits_all, y_all = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits_all.append(model(x.to(device_t)).cpu())
                y_all.append(y)
        logits_all = torch.cat(logits_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        stats = per_label_f1(logits_all, y_all, args.threshold)
        acc = multilabel_accuracy(logits_all, y_all, args.threshold)
        print(
            "epoch %02d  loss=%.4f  val_macro_f1=%.3f  subset_acc=%.3f"
            % (epoch, train_loss, stats["macro_f1"], acc)
        )
        if stats["macro_f1"] > best_f1:
            best_f1 = stats["macro_f1"]
            args.ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "instruments": list(args.instruments),
                    "threshold": args.threshold,
                    "val_macro_f1": best_f1,
                    "val_subset_acc": acc,
                },
                args.ckpt,
            )
            meta = args.ckpt.with_suffix(".json")
            meta.write_text(
                json.dumps(
                    {
                        "instruments": list(args.instruments),
                        "val_macro_f1": best_f1,
                        "val_subset_acc": acc,
                        "threshold": args.threshold,
                    },
                    indent=2,
                )
            )
            print("  saved", args.ckpt)

    print("Best val macro-F1: %.3f" % best_f1)
    print("Validate on held-out REAL stems before trusting generated-audio scores.")


@torch.no_grad()
def eval_generated(args):
    if args.eval_wav_dir is None or not args.eval_wav_dir.is_dir():
        raise SystemExit("--eval-wav-dir required for --eval-only")
    if not args.ckpt.is_file():
        raise SystemExit("Missing checkpoint: %s" % args.ckpt)

    blob = torch.load(args.ckpt, map_location="cpu")
    instruments = blob["instruments"]
    thr = float(blob.get("threshold", args.threshold))
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_t = torch.device(device)

    from audio_diffusion_pytorch import NormalizedEncodec

    model = TimbreClassifier(n_labels=len(instruments)).to(device_t)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    enc = NormalizedEncodec(device=device)

    target = args.target_instruments or []
    target_idx = [instruments.index(t) for t in target]

    rows = []
    paths = sorted(args.eval_wav_dir.glob("*.wav"))
    for path in paths:
        wav = load_crop(path, device_t)
        emb = embed_wav(enc, wav)
        prob = torch.sigmoid(model(emb.unsqueeze(0))).squeeze(0).cpu().numpy()
        pred = [instruments[i] for i, p in enumerate(prob) if p >= thr]
        hit = all(prob[i] >= thr for i in target_idx) if target_idx else None
        rows.append(
            {
                "file": path.name,
                "pred": ",".join(pred),
                "target_ok": hit,
                "target_exact": set(pred) == set(target) if target else None,
                **{instruments[i]: float(prob[i]) for i in range(len(instruments))},
            }
        )
        print(path.name, "->", pred, "target_ok=" + str(hit))

    out = args.eval_wav_dir / "timbre_classifier_scores.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(out, index=False)
    if target_idx:
        ok = [r["target_ok"] for r in rows]
        print("Target instruments %s accuracy: %.3f" % (target, float(np.mean(ok))))
    print("Wrote", out)


def main():
    args = parse_args()
    if args.eval_only:
        eval_generated(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
