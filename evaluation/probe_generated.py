"""Do EnCodec and CLAP agree on *which instrument* the models' outputs sound like?

The instrument probes of embedding_instrument_probe.py tell the 13 instruments
apart on real recordings almost perfectly in both embeddings (99%). Here the
same kind of probe -- one per embedding, trained on all those real stems --
is applied to the clips the models generated, using the embeddings the FADs
already cached. No audio is re-embedded and nothing is regenerated.

Monophonic (cello -> bassoon): per model, the share of clips each probe calls
bassoon, the share it still calls cello, and the mean probability of bassoon.
Real bassoon recordings (the FAD reference) are the anchor: they should read
~100% bassoon. If the two probes agree here while the two FADs disagree, the
FADs differ in something other than instrument identity.

Polyphonic, exploratory: a probe trained on single instruments is applied to
two-instrument mixtures, so only its probability mass is read -- violin +
cello against flute + bassoon -- with real violin + cello as the anchor.

Probe training excludes every track that is a model input or a reference
clip here, so no output is judged by a probe that saw its own piece.

  venv_yuval/bin/python evaluation/probe_generated.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.clap_text_probe import read_fads, spearman  # noqa: E402
from evaluation.embedding_instrument_probe import INSTRUMENTS  # noqa: E402

GROUPS = [("Outputs/WAV_files/clap_validation_1000", "mono"), ("Outputs/WAV_files/clap_validation_1000", "poly"),
          ("Outputs/WAV_files/merged_midi_test", "poly")]
EMBEDDINGS = ["encodec", "clap"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-data", default="Outputs/embedding_instrument_probe/embeddings.npz")
    return p.parse_args()


def train_probe(X, y, l2=1e-3):
    """Same linear probe as embedding_instrument_probe.py, trained on everything; returns predict_proba."""
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-8
    A = torch.tensor((X - mu) / sd, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.long)
    k = len(INSTRUMENTS)
    W = torch.zeros(A.shape[1], k, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, t) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, b = W.detach(), b.detach()

    def proba(Z):
        with torch.no_grad():
            return torch.softmax(torch.tensor((Z - mu) / sd, dtype=torch.float64) @ W + b, dim=1).numpy()
    return proba


def main():
    args = parse_args()
    data = np.load(args.probe_data)
    y, tracks = data["y"], np.array([str(t) for t in data["tracks"]])

    # tracks that appear here as inputs or references must not train the probe
    used = set()
    for root, case in GROUPS:
        d = Path(root) / "fad" / case
        used |= {p.stem for p in (d / "reference").glob("*.wav")}
        used |= {p.stem for p in (d / "generated").glob("*/*.wav")}
    keep = np.array([t not in used for t in tracks])
    print("probe training clips: %d of %d (%d dropped: their track is a model input or a reference here)"
          % (keep.sum(), len(y), (~keep).sum()))
    if len(set(y[keep])) < len(INSTRUMENTS):
        raise SystemExit("an instrument lost all its training clips -- rerun embedding_instrument_probe.py with more")
    probes = {e: train_probe(data[e][keep].astype(np.float64), y[keep]) for e in EMBEDDINGS if e in data.files}
    I = {name: i for i, name in enumerate(INSTRUMENTS)}

    for root, case in GROUPS:
        d = Path(root) / "fad" / case
        if not (d / "reference_clap.npz").is_file():
            continue
        enc_fad, clap_fad = read_fads(root)
        exps = sorted(p.name[:-len("_clap.npz")] for p in (d / "generated").glob("*_clap.npz"))
        sets = [("REAL reference", d / "reference_%s.npz")] + [(e, d / "generated" / ("%s_%%s.npz" % e)) for e in exps]
        print("\n" + "=" * 112)
        if case == "mono":
            print("  %s / mono -- what do the probes call the outputs? (target bassoon, source cello)" % Path(root).name)
            print("=" * 112)
            print("  %-32s %8s %7s | %-27s | %-27s" % ("condition", "CLAP-FAD", "EnC-FAD",
                  "EnCodec probe: bsn / cello", "CLAP probe: bsn / cello"))
        else:
            print("  %s / poly -- EXPLORATORY: probability mass, violin+cello vs flute+bassoon" % Path(root).name)
            print("=" * 112)
            print("  %-32s %8s %7s | %-27s | %-27s" % ("condition", "CLAP-FAD", "EnC-FAD",
                  "EnCodec probe: vln+vc / fl+bsn", "CLAP probe: vln+vc / fl+bsn"))
        score = {e: [] for e in probes}
        fads = {"clap": [], "encodec": []}
        for name, pattern in sets:
            cells = []
            for e, proba in probes.items():
                path = Path(str(pattern) % e)
                if not path.is_file():
                    cells.append("%27s" % "--")
                    continue
                P = proba(np.load(path)["emb"].astype(np.float64))
                if case == "mono":
                    pred = P.argmax(1)
                    a, b = np.mean(pred == I["bassoon"]), np.mean(pred == I["cello"])
                    top_other = np.bincount(pred, minlength=len(INSTRUMENTS))
                    top_other[[I["bassoon"], I["cello"]]] = 0
                    other = INSTRUMENTS[int(top_other.argmax())] if top_other.max() else "-"
                    cells.append("%5.0f%% / %3.0f%%  (else %-7s)" % (100 * a, 100 * b, other[:7]))
                    s = a
                else:
                    a = np.mean(P[:, I["violin"]] + P[:, I["cello"]])
                    b = np.mean(P[:, I["flute"]] + P[:, I["bassoon"]])
                    cells.append("%11.2f / %-13.2f" % (a, b))
                    s = a - b
                if name != "REAL reference":
                    score[e].append(s)
            c = clap_fad.get((case, name), float("nan"))
            en = enc_fad.get((case, name), float("nan"))
            if name != "REAL reference":
                fads["clap"].append(c)
                fads["encodec"].append(en)
            print("  %-32s %8s %7s | %s | %s" % (name, "" if np.isnan(c) else "%.4f" % c,
                                                 "" if np.isnan(en) else "%.2f" % en, cells[0], cells[1]))
        if all(len(v) > 2 for v in score.values()) and len(score) == 2:
            print("\n  Spearman across the %d models:" % len(score["clap"]))
            print("    EnCodec probe vs CLAP probe (do they agree on the instrument?)   %+.2f"
                  % spearman(score["encodec"], score["clap"]))
            for e in probes:
                print("    %-8s probe vs its own FAD (more target-like = lower FAD?)      %+.2f"
                      % (e, -spearman(score[e], fads[e])))


if __name__ == "__main__":
    main()
