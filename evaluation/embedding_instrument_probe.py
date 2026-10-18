"""Which embedding hears instruments best: EnCodec (mean-pooled), CLAP, or music CLAP?

The two FADs disagree, and CLAP's TEXT side cannot name a cello or a bassoon
(clap_multiple_choice.py). FAD uses only the AUDIO side, so test that side
directly, with no text: the same real single-instrument recordings -- every
one of the 13 CocoChorales instruments -- go through each embedding exactly as
its FAD prepares them, and a simple linear classifier is trained to tell the
13 instruments apart, scored on tracks it never saw (5-fold, split by track,
so no piece is in both training and test).

**Decided before the numbers are seen:** the embedding with the higher
held-out accuracy carries more information about which instrument is
playing -- more timbre. A gap smaller than the spread over folds is a tie.
Reported alongside, because they matter most here: how often cello and
bassoon are recognised, and what they are mistaken for.

What it does NOT show: that the FAD *distance* in that embedding matches what
a listener hears. Only the listening test answers that.

Each embedding sees the audio as its FAD does: EnCodec the whole 17.07 s at
24 kHz, mean over time; CLAP a random 10 s at 48 kHz (its own preprocessing,
seeded here). Same clips for all three.

  venv_yuval/bin/python evaluation/embedding_instrument_probe.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INSTRUMENTS = ["violin", "viola", "cello", "double_bass", "flute", "oboe", "clarinet", "bassoon",
               "trumpet", "horn", "trombone", "tuba", "saxophone"]
MODELS = {"encodec": "facebook/encodec_24khz (raw latent, mean over time)",
          "clap": "laion/clap-htsat-unfused (the CLAP-FAD model)",
          "clap_music": "laion/larger_clap_music"}
WATCH = {"cello": ["violin", "viola", "double_bass"], "bassoon": ["trombone", "tuba", "horn", "clarinet"]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-instrument", type=int, default=100)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="Outputs/embedding_instrument_probe")
    return p.parse_args()


# ------------------------------------------------------------------ data
def collect_stems(dataset_root, n, seed):
    """Up to n stems per instrument, at most one per instrument per track, from shuffled tracks."""
    from evaluation.train_timbre_classifier import instrument_from_stem_name
    tracks = sorted(p for p in Path(dataset_root).glob("*_track*") if (p / "stems_audio").is_dir())
    random.Random(seed).shuffle(tracks)
    picked = defaultdict(list)
    for track in tracks:
        if all(len(picked[i]) >= n for i in INSTRUMENTS):
            break
        seen = set()
        for wav in sorted((track / "stems_audio").glob("*.wav")):
            inst = instrument_from_stem_name(wav.name, INSTRUMENTS)
            if inst and inst not in seen and len(picked[inst]) < n:
                picked[inst].append((wav, track.name))
                seen.add(inst)
    return picked


def load_24k(path):
    """Mono, 24 kHz, first 17.07 s (zero-padded), as the bridge and both FADs crop."""
    import torch
    import torchaudio
    from evaluation.bridge_200 import CLIP_LENGTH, SAMPLING_RATE
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0, keepdim=True)
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    wav = wav[..., :CLIP_LENGTH]
    return torch.nn.functional.pad(wav, (0, CLIP_LENGTH - wav.shape[-1]))[0].numpy().astype(np.float32)


def peak95(x):
    return x / (np.max(np.abs(x)) + 1e-9) * 0.95


# ------------------------------------------------------------------ embedders
def make_embedders(names, device):
    import torch
    out = {}
    if "encodec" in names:
        from audio_diffusion_pytorch import NormalizedEncodec
        enc = NormalizedEncodec(device=device)

        def encodec(x24):
            w = torch.from_numpy(peak95(x24)).view(1, 1, -1).to(device)
            return enc.encode_latent(w, None, None).mean(dim=-1).squeeze(0).cpu().numpy()
        out["encodec"] = encodec
    for key, model_name in (("clap", "laion/clap-htsat-unfused"), ("clap_music", "laion/larger_clap_music")):
        if key not in names:
            continue
        try:
            from transformers import ClapModel, ClapProcessor
            proc = ClapProcessor.from_pretrained(model_name)
            model = ClapModel.from_pretrained(model_name).to(device).eval()
        except Exception as exc:  # noqa: BLE001 -- e.g. no download; say so and go on
            print("  %s could not be loaded (%s: %s) -- skipped" % (model_name, type(exc).__name__, exc))
            continue

        def clap(x24, proc=proc, model=model):
            import librosa
            x48 = peak95(librosa.resample(x24, orig_sr=24000, target_sr=48000))
            inputs = {k: v.to(device) for k, v in proc(audios=x48, sampling_rate=48000, return_tensors="pt").items()}
            with torch.no_grad():
                return model.get_audio_features(**inputs).cpu().numpy()[0]
        out[key] = clap
    return out


# ------------------------------------------------------------------ evaluation
def linear_probe(Xtr, ytr, Xte, n_classes, l2=1e-3):
    """Multinomial logistic regression on standardised features, same settings for every embedding."""
    import torch
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float64)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float64)
    y = torch.tensor(ytr, dtype=torch.long)
    W = torch.zeros(A.shape[1], n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (B @ W + b).argmax(1).numpy()


def nearest_centroid(Xtr, ytr, Xte, n_classes):
    """No training beyond class means: cosine to each standardised class centroid."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    C = np.stack([A[ytr == c].mean(0) for c in range(n_classes)])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (B @ C.T).argmax(1)


def evaluate(X, y, tracks, n_folds, seed):
    """Per fold accuracies of both classifiers, and the pooled confusion matrix of the linear probe."""
    uniq = sorted(set(tracks))
    random.Random(seed).shuffle(uniq)
    fold_of = {t: i % n_folds for i, t in enumerate(uniq)}
    folds = np.array([fold_of[t] for t in tracks])
    k = len(INSTRUMENTS)
    acc_lin, acc_nc, conf = [], [], np.zeros((k, k), int)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(set(y[tr])) < k:
            continue
        p = linear_probe(X[tr], y[tr], X[te], k)
        acc_lin.append(float(np.mean(p == y[te])))
        acc_nc.append(float(np.mean(nearest_centroid(X[tr], y[tr], X[te], k) == y[te])))
        for t, q in zip(y[te], p):
            conf[t, q] += 1
    return np.array(acc_lin), np.array(acc_nc), conf


def main():
    args = parse_args()
    from evaluation.bridge_200 import DATASET_ROOT
    np.random.seed(args.seed)          # CLAP's random 10 s crop draws from numpy's global RNG
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    picked = collect_stems(DATASET_ROOT, args.per_instrument, args.seed)
    print("stems per instrument: " + ", ".join("%s %d" % (i, len(picked[i])) for i in INSTRUMENTS))
    missing = [i for i in INSTRUMENTS if len(picked[i]) < 10]
    if missing:
        raise SystemExit("too few stems for %s -- check the dataset root %s" % (missing, DATASET_ROOT))

    embedders = make_embedders(args.models, args.device)
    feats = {m: [] for m in embedders}
    y, tracks = [], []
    items = [(wav, trk, INSTRUMENTS.index(inst)) for inst in INSTRUMENTS for wav, trk in picked[inst]]
    skipped = 0
    for n, (wav, trk, label) in enumerate(items, 1):
        x = load_24k(wav)
        if np.sqrt(np.mean(x ** 2)) < 1e-4:      # silent in the first 17 s
            skipped += 1
            continue
        for m, fn in embedders.items():
            feats[m].append(fn(x))
        y.append(label)
        tracks.append(trk)
        if n % 200 == 0:
            print("  embedded %d/%d" % (n, len(items)), flush=True)
    y = np.array(y)
    print("  %d clips embedded, %d silent ones skipped, %d tracks" % (len(y), skipped, len(set(tracks))))
    np.savez(out / "embeddings.npz", y=y, tracks=np.array(tracks), **{m: np.array(v) for m, v in feats.items()})

    print("\n" + "=" * 112)
    print("  Telling 13 instruments apart, on tracks the classifier never saw (%d folds). Chance %.1f%%."
          % (args.folds, 100 / len(INSTRUMENTS)))
    print("=" * 112)
    print("  %-11s %5s  %-22s %-22s %8s %8s   %s" % ("embedding", "dim", "linear probe", "nearest centroid",
                                                   "cello", "bassoon", "cello / bassoon mistaken for"))
    rows = []
    confs = {}
    for m in embedders:
        X = np.array(feats[m], dtype=np.float64)
        lin, nc, conf = evaluate(X, y, tracks, args.folds, args.seed)
        confs[m] = conf
        rec = conf.diagonal() / np.maximum(conf.sum(1), 1)
        mist = []
        for inst, others in WATCH.items():
            i = INSTRUMENTS.index(inst)
            row = conf[i] / max(conf[i].sum(), 1)
            mist.append("%s->%s" % (inst[:4], ",".join("%s %.0f%%" % (o[:6], 100 * row[INSTRUMENTS.index(o)])
                                                          for o in others if row[INSTRUMENTS.index(o)] >= 0.05) or "-"))
        print("  %-11s %5d  %5.1f%% +- %4.1f%%        %5.1f%% +- %4.1f%%        %6.0f%% %7.0f%%   %s"
              % (m, X.shape[1], 100 * lin.mean(), 100 * lin.std(), 100 * nc.mean(), 100 * nc.std(),
                 100 * rec[INSTRUMENTS.index("cello")], 100 * rec[INSTRUMENTS.index("bassoon")], " | ".join(mist)))
        rows.append(dict(embedding=m, model=MODELS[m], dim=X.shape[1], linear_acc=lin.mean(), linear_sd=lin.std(),
                         centroid_acc=nc.mean(), centroid_sd=nc.std(),
                         **{"recall_" + i: rec[j] for j, i in enumerate(INSTRUMENTS)}))

    print("\n  Recognised per instrument (linear probe, all folds pooled):")
    print("  %-12s" % "instrument" + "".join("%12s" % m for m in embedders))
    for j, inst in enumerate(INSTRUMENTS):
        print("  %-12s" % inst + "".join("%11.0f%%" % (100 * confs[m][j, j] / max(confs[m][j].sum(), 1))
                                         for m in embedders))

    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m, conf in confs.items():
        with (out / ("confusion_%s.csv" % m)).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true \\ predicted"] + INSTRUMENTS)
            for j, inst in enumerate(INSTRUMENTS):
                w.writerow([inst] + list(conf[j]))
    print("\n-> %s/{summary.csv,confusion_<embedding>.csv,embeddings.npz}" % out)


if __name__ == "__main__":
    main()
