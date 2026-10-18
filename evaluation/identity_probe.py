"""Does our pipeline's own noise make CLAP hear wind instruments that are not there?

On the transfer outputs the instrument probes split: EnCodec's calls every
output the target (100% bassoon), CLAP's hears the source family in a large
share (flute, flute + bassoon). Either CLAP is right and EnCodec approves
anything made in its own latent space, or the outputs' noise reads as
"flute" to CLAP.

Identity transfer separates the two: the target ensemble goes in AND out
(violin + cello -> violin + cello, bassoon -> bassoon), through the same
codec, the same diffusion models and the same handoffs. The instruments do
not change, so anything non-target the probes report is the pipeline's
artifacts, not a leftover source. The inputs are the real clips of the FAD
reference, so every clip is compared with itself before the pipeline.

**Decided before the numbers are seen.** For CLAP's probe, the share of what
it hears in the transfers that the identity outputs reproduce:
    r = (identity - real) / (transfer - real)
  - poly: probability mass on flute + bassoon; mono: share NOT called bassoon,
  - compared at the same handoff (100 with 100, 5 with 5).
  r >= 0.5  the pipeline's noise alone makes at least half of it: CLAP's
            "source" calls on the transfers are mostly noise.
  r <= 0.25 the noise makes little of it: what CLAP hears in the transfers is
            mostly the source instruments, and EnCodec's 100% is suspect.
  between   both contribute.
The test only counts if the identity outputs are about as noisy as the
transfers (CLAP's noisy-vs-clean text axis); if they are clearly cleaner the
script says the noise was only partly reproduced.

  venv_yuval/bin/python evaluation/identity_probe.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.embedding_instrument_probe import INSTRUMENTS  # noqa: E402

TIMBRE = "Outputs/WAV_files/clap_validation_1000"
CASES = {
    "poly": dict(identity="identity_vc_midi_s100", transfer="original_midi_s100",
                 target=("violin", "cello"), source=("flute", "bassoon")),
    "mono": dict(identity="identity_bsn_midi_s100", transfer="mono_c2b_original_midi_s100",
                 target=("bassoon",), source=("cello",)),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="Outputs/WAV_files/identity_test")
    p.add_argument("--probe-data", default="Outputs/embedding_instrument_probe/embeddings.npz")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def output_rows(out_root):
    """{experiment: [(sample_id, output wav)]} from the per-condition CSVs, merged first."""
    from evaluation.bridge_200 import merge_csvs
    merged = merge_csvs(out_root)
    rows = defaultdict(list)
    with open(merged, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["experiment"]].append((r["sample_id"], r["output_audio"]))
    return rows


def embed_outputs(rows, device, cache):
    """CLAP and EnCodec embeddings of every output, prepared exactly as their FADs prepare clips."""
    import torch
    from audio_diffusion_pytorch import NormalizedEncodec
    from evaluation.clap_text_probe import embed_audio, load_model
    from evaluation.compute_clap_fad import load_audio
    from evaluation.compute_encodec_fad import load_audio_24k

    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        have = {k for k in z.files}
        if all(("%s__clap" % e) in have for e in rows):
            print("  embeddings cached: %s" % cache)
            return {k: z[k] for k in z.files}
    proc, model = load_model(device)
    enc = NormalizedEncodec(device=device)
    out = {}
    for exp, items in sorted(rows.items()):
        paths = [p for _, p in items]
        out[exp + "__names"] = np.array([s for s, _ in items])
        out[exp + "__clap"] = embed_audio(proc, model, device, [load_audio(Path(p)) for p in paths])
        with torch.no_grad():
            out[exp + "__encodec"] = np.stack([
                enc.encode_latent(load_audio_24k(Path(p)).unsqueeze(0).to(device), None, None)
                .mean(dim=-1).squeeze(0).cpu().numpy() for p in paths])
        print("  embedded %s: %d clips" % (exp, len(paths)), flush=True)
    np.savez(cache, **out)
    return out


def main():
    args = parse_args()
    np.random.seed(args.seed)      # CLAP's random 10 s crop
    from evaluation.clap_text_probe import load_model, text_directions, unit
    from evaluation.probe_generated import train_probe

    out_root = Path(args.out_root)
    rows = output_rows(out_root)
    if not rows:
        raise SystemExit("no identity outputs under %s -- run the generation first" % out_root)
    emb = embed_outputs(rows, args.device, out_root / "identity_embeddings.npz")

    proc, model = load_model(args.device)
    T = text_directions(proc, model, args.device)
    noisy_axis = T["low_quality"] - T["high_quality"]

    data = np.load(args.probe_data)
    y, tracks = data["y"], np.array([str(t) for t in data["tracks"]])
    used = {str(s) for k in emb if k.endswith("__names") for s in emb[k]}
    for case in CASES:
        d = Path(TIMBRE) / "fad" / case
        used |= {p.stem for p in (d / "reference").glob("*.wav")}
        used |= {p.stem for p in (d / "generated").glob("*/*.wav")}
    keep = np.array([t not in used for t in tracks])
    print("probe training clips: %d of %d" % (keep.sum(), len(y)))
    probes = {e: train_probe(data[e][keep].astype(np.float64), y[keep]) for e in ("encodec", "clap")}
    I = {n: i for i, n in enumerate(INSTRUMENTS)}

    for case, c in CASES.items():
        d = Path(TIMBRE) / "fad" / case
        ids = [k[:-len("__names")] for k in emb if k.endswith("__names") and k.startswith(c["identity"])]
        if not ids:
            print("\n%s: no identity outputs -- skipped" % case)
            continue
        same = {str(s) + ".wav" for i in ids for s in emb[i + "__names"]}

        def load_set(kind, exp=None):
            if kind == "real":
                out = {}
                for e in ("clap", "encodec"):
                    z = np.load(d / ("reference_%s.npz" % e))
                    m = np.array([str(n) in same for n in z["names"]])
                    out[e] = z["emb"][m]
                return out
            if kind == "identity":
                return {e: emb["%s__%s" % (exp, e)] for e in ("clap", "encodec")}
            return {e: np.load(d / "generated" / ("%s_%s.npz" % (exp, e)))["emb"] for e in ("clap", "encodec")}

        sets = [("real, the same clips", load_set("real"))]
        for h, suffix in ((100, ""), (5, "_h5")):
            if c["identity"] + suffix in ids:
                sets.append(("identity, handoff %d" % h, load_set("identity", c["identity"] + suffix)))
            tp = d / "generated" / ("%s%s_clap.npz" % (c["transfer"], suffix))
            if tp.is_file():
                sets.append(("transfer, handoff %d" % h, load_set("transfer", c["transfer"] + suffix)))

        stats = {}
        print("\n" + "=" * 112)
        print("  %s: %s -> %s through the pipeline, against the real clips and the real transfer"
              % (case, " + ".join(c["target"]), " + ".join(c["target"])))
        print("=" * 112)
        if case == "poly":
            print("  %-26s %5s | %-27s | %-27s | %10s" % ("set", "n", "EnCodec probe: vln+vc / fl+bsn",
                                                       "CLAP probe: vln+vc / fl+bsn", "noisiness"))
        else:
            print("  %-26s %5s | %-16s | %-34s | %10s" % ("set", "n", "EnCodec: bassoon",
                                                       "CLAP: bassoon / cello / flute", "noisiness"))
        for name, s in sets:
            Pe, Pc = probes["encodec"](s["encodec"].astype(np.float64)), probes["clap"](s["clap"].astype(np.float64))
            noisy = float(np.mean(unit(s["clap"]) @ noisy_axis))
            if case == "poly":
                mass = lambda P, insts: float(np.mean(P[:, [I[i] for i in insts]].sum(1)))  # noqa: E731
                src = mass(Pc, c["source"])
                print("  %-26s %5d | %11.2f / %-13.2f | %11.2f / %-13.2f | %+10.4f"
                      % (name, len(Pc), mass(Pe, c["target"]), mass(Pe, c["source"]),
                         mass(Pc, c["target"]), src, noisy))
            else:
                share = lambda P, i: float(np.mean(P.argmax(1) == I[i]))  # noqa: E731
                src = 1 - share(Pc, "bassoon")
                print("  %-26s %5d | %15.0f%% | %8.0f%% / %5.0f%% / %5.0f%%        | %+10.4f"
                      % (name, len(Pc), 100 * share(Pe, "bassoon"), 100 * share(Pc, "bassoon"),
                         100 * share(Pc, "cello"), 100 * share(Pc, "flute"), noisy))
            stats[name] = (src, noisy)

        what = "flute + bassoon mass" if case == "poly" else "share not called bassoon"
        print("\n  Decision (CLAP probe, %s; r = share of the transfer's excess the identity reproduces):" % what)
        real_src, real_noisy = stats["real, the same clips"]
        for h in (100, 5):
            ik, tk = "identity, handoff %d" % h, "transfer, handoff %d" % h
            if ik not in stats or tk not in stats:
                continue
            (i_src, i_noisy), (t_src, t_noisy) = stats[ik], stats[tk]
            if t_src - real_src <= 1e-9:
                print("    handoff %3d: the transfer shows no excess over the real clips -- nothing to explain" % h)
                continue
            r = (i_src - real_src) / (t_src - real_src)
            verdict = ("the pipeline's noise alone makes at least half of it -> CLAP's source calls are mostly noise"
                       if r >= 0.5 else
                       "the noise makes little of it -> CLAP hears the source instruments; EnCodec's 100% is suspect"
                       if r <= 0.25 else "both contribute")
            print("    handoff %3d: real %.2f, identity %.2f, transfer %.2f -> r = %.2f: %s"
                  % (h, real_src, i_src, t_src, r, verdict))
            if t_noisy - real_noisy > 0 and (i_noisy - real_noisy) < 0.5 * (t_noisy - real_noisy):
                print("                 caution: the identity outputs are clearly cleaner than the transfer "
                      "(noisiness %+.4f vs %+.4f, real %+.4f) -- the noise was only partly reproduced"
                      % (i_noisy, t_noisy, real_noisy))


if __name__ == "__main__":
    main()
