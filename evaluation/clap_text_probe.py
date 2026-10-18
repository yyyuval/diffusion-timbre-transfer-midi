"""What does CLAP-FAD respond to: the instruments, or how clean the recording is?

CLAP maps audio and text into one space, so each clip's cached audio embedding
can be compared with descriptions: "violin and cello" against "flute and
bassoon" (the instrument axis), and "an old noisy recording" against "a clean
studio recording" (the quality axis). No generated audio is re-embedded.

**Hypothesis (Shlomo, 2026-10-18).** CLAP-FAD ranks the models mainly by how
clean or noisy they sound, not by how much they sound like the target
instruments -- which would explain why it disagrees with EnCodec-FAD.

**Decided before the numbers are seen:**
  0. The probes must work first, on real audio with a known answer:
     - instrument: real target vs real source recordings, AUC >= 0.8;
     - quality: real recordings vs the same recordings made to sound old
       (low-passed at 4 kHz, hiss at 25 dB SNR), the degraded one scored
       noisier in >= 80% of pairs.
     A probe that fails says nothing; the question is then left open.
  1. Supported if, in the pairs the two FADs order differently (robust ones
     where diagnose_fad.py's output is found), CLAP's pick is the cleaner clip
     set clearly more often than it is the more target-like one.
  2. Refuted if CLAP's picks are the more target-like ones.
This tells what CLAP *responds to*, not what is true of the audio; the
listening test answers that.

  venv_yuval/bin/python evaluation/clap_text_probe.py            # CPU, ~5-10 min
  venv_yuval/bin/python evaluation/clap_text_probe.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOTS = ["Outputs/WAV_files/clap_validation_1000", "Outputs/WAV_files/merged_midi_test"]
# several phrasings per concept, averaged, so no single wording decides
CONCEPTS = {
    "violin_cello": ["the sound of a violin and a cello playing together",
                     "a string duet of violin and cello", "violin and cello"],
    "flute_bassoon": ["the sound of a flute and a bassoon playing together",
                      "a woodwind duet of flute and bassoon", "flute and bassoon"],
    "bassoon": ["the sound of a solo bassoon", "a bassoon playing", "bassoon"],
    "cello": ["the sound of a solo cello", "a cello playing", "cello"],
    "low_quality": ["an old noisy recording", "a low quality recording with hiss and crackle",
                    "a muffled lo-fi recording", "a distorted recording with digital artifacts"],
    "high_quality": ["a clean high quality studio recording", "a crisp and clear recording",
                     "high fidelity audio"],
}
TARGET_SOURCE = {"poly": ("violin_cello", "flute_bassoon"), "mono": ("bassoon", "cello")}
SR48 = 48000
WINDOW = 10 * SR48           # CLAP's window; clips cut to exactly this are not cropped at random


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    p.add_argument("--n", type=int, default=60, help="real clips per anchor (source; clean/degraded)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cpu")
    p.add_argument("--diagnosis", default="Outputs/fad_diagnosis/disagreeing_pairs.csv",
                   help="diagnose_fad.py output, to mark which disagreements are robust")
    p.add_argument("--out-dir", default="Outputs/clap_text_probe")
    return p.parse_args()


# ------------------------------------------------------------------ CLAP
def load_model(device):
    from transformers import ClapModel, ClapProcessor
    from evaluation.compute_clap_fad import MODEL_NAME
    return ClapProcessor.from_pretrained(MODEL_NAME), ClapModel.from_pretrained(MODEL_NAME).to(device).eval()


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def text_directions(proc, model, device):
    """{concept: mean of its phrasings' unit text embeddings}; cos(a, t) averaged = a . mean(t)."""
    import torch
    out = {}
    for concept, phrases in CONCEPTS.items():
        inputs = {k: v.to(device) for k, v in proc(text=phrases, return_tensors="pt", padding=True).items()}
        with torch.no_grad():
            out[concept] = unit(model.get_text_features(**inputs).cpu().numpy()).mean(0)
    return out


def embed_audio(proc, model, device, clips):
    """Same call as compute_clap_fad.get_embeddings, one clip at a time."""
    import torch
    embs = []
    for x in clips:
        inputs = {k: v.to(device) for k, v in proc(audios=x.astype(np.float32), sampling_rate=SR48,
                                                    return_tensors="pt").items()}
        with torch.no_grad():
            embs.append(model.get_audio_features(**inputs).cpu().numpy()[0])
    return np.stack(embs)


# ------------------------------------------------------------------ audio
def peak95(x):
    return x / (np.max(np.abs(x)) + 1e-9) * 0.95


def real_source_audio(case, tracks):
    """The model inputs, preprocessed exactly as the bridge saw them, at 48 kHz."""
    import librosa
    from evaluation.bridge_200 import CASES, DATASET_ROOT, SAMPLING_RATE, load_input_waveform, stem_paths
    out = []
    for t in tracks:
        w = load_input_waveform(stem_paths(DATASET_ROOT, t, CASES[case]["stems"])).squeeze(0).mean(0).numpy()
        out.append(peak95(librosa.resample(w.astype(np.float32), orig_sr=SAMPLING_RATE, target_sr=SR48)))
    return out


def degrade(x, rng):
    """'Old recording': low-pass at 4 kHz and white hiss at 25 dB SNR."""
    from scipy.signal import butter, sosfiltfilt
    y = sosfiltfilt(butter(6, 4000, btype="low", fs=SR48, output="sos"), x)
    noise = rng.normal(size=y.shape)
    noise *= np.sqrt(np.mean(y ** 2) / 10 ** (25 / 10)) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return peak95(y + noise)


def clean_and_degraded(ref_paths, rng):
    from evaluation.compute_clap_fad import load_audio
    clean, bad = [], []
    for p in ref_paths:
        x = load_audio(p)[:WINDOW]
        x = np.pad(x, (0, WINDOW - len(x)))
        clean.append(peak95(x))
        bad.append(degrade(x, rng))
    return clean, bad


# ------------------------------------------------------------------ scoring
def scores(E, T, case):
    """instrument = cos(target) - cos(source); noisiness = cos(old/noisy) - cos(clean)."""
    E = unit(E)
    tgt, src = TARGET_SOURCE[case]
    return E @ (T[tgt] - T[src]), E @ (T["low_quality"] - T["high_quality"])


def auc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean(pos > neg) + 0.5 * np.mean(pos == neg))


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation) if len(x) > 2 else float("nan")


def read_fads(root):
    def load(name, col):
        path = Path(root) / "fad" / name
        if not path.is_file():
            return {}
        with io.open(path, newline="", encoding="utf-8") as f:
            return {(r["case"], r["experiment"]): float(r[col]) for r in csv.DictReader(f) if r.get(col)}
    return load("encodec_fad_results.csv", "fad"), load("clap_fad_200.csv", "clap_fad")


def read_robust(path):
    if not Path(path).is_file():
        return None
    with io.open(path, newline="", encoding="utf-8") as f:
        return {(r["group"], frozenset((r["encodec_better"], r["encodec_worse"])))
                for r in csv.DictReader(f)
                if float(r["p_encodec"]) >= 0.95 and float(r["p_clap_same_as_encodec"]) <= 0.05}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pick = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc, model = load_model(args.device)
    T = text_directions(proc, model, args.device)

    groups = []
    for root in args.roots:
        for case_dir in sorted((Path(root) / "fad").glob("*")):
            ref = case_dir / "reference_clap.npz"
            if case_dir.name in TARGET_SOURCE and ref.is_file():
                groups.append((Path(root), case_dir.name, case_dir))
    if not groups:
        raise SystemExit("no cached CLAP embeddings under %s" % args.roots)

    # ---- anchors and probe checks, once per case
    anchors = {}
    print("=" * 100)
    print("  0. Do the probes work? (real audio, known answer)")
    print("=" * 100)
    for case in sorted({g[1] for g in groups}):
        case_dir = next(g[2] for g in groups if g[1] == case)
        ref = np.load(case_dir / "reference_clap.npz")["emb"]
        sids = sorted({p.stem for p in (case_dir / "generated").glob("*/*.wav")})
        src_cache = out / ("real_source_%s_clap.npz" % case)
        if src_cache.is_file():
            src = np.load(src_cache)["emb"]
        else:
            src = embed_audio(proc, model, args.device, real_source_audio(case, pick.sample(sids, min(args.n, len(sids)))))
            np.savez(src_cache, emb=src)
        refs = sorted((case_dir / "reference").glob("*.wav"))
        clean, bad = clean_and_degraded(pick.sample(refs, min(args.n, len(refs))), rng)
        E_clean, E_bad = embed_audio(proc, model, args.device, clean), embed_audio(proc, model, args.device, bad)
        i_ref, q_ref = scores(ref, T, case)
        i_src, q_src = scores(src, T, case)
        i_cl, q_cl = scores(E_clean, T, case)
        i_bad, q_bad = scores(E_bad, T, case)
        a_inst = auc(i_ref, i_src)
        share_q = float(np.mean(q_bad > q_cl))
        tgt, srcn = TARGET_SOURCE[case]
        print("  %s  instrument probe (%s vs %s): AUC %.2f  -> %s" % (case, tgt, srcn, a_inst,
              "works" if a_inst >= 0.8 else "DOES NOT WORK"))
        print("  %s  quality probe: degraded scored noisier in %.0f%% of %d pairs (shift %+.4f) -> %s"
              % (case, 100 * share_q, len(q_cl), np.mean(q_bad - q_cl), "works" if share_q >= 0.8 else "DOES NOT WORK"))
        print("  %s  does degrading also move the instrument score? %+.4f (target-minus-source gap %.4f)"
              % (case, np.mean(i_bad - i_cl), np.mean(i_ref) - np.mean(i_src)))
        anchors[case] = dict(i_src=float(np.mean(i_src)), q_shift=float(np.mean(q_bad - q_cl)),
                             ok_inst=a_inst >= 0.8, ok_q=share_q >= 0.8)

    robust = read_robust(args.diagnosis)
    cond_rows, pair_rows = [], []
    for root, case, case_dir in groups:
        label = "%s / %s" % (root.name, case)
        enc, clap = read_fads(root)
        ref = np.load(case_dir / "reference_clap.npz")["emb"]
        i_ref, q_ref = scores(ref, T, case)
        a = anchors[case]
        span = float(np.mean(i_ref)) - a["i_src"]
        res = {}
        for p in sorted((case_dir / "generated").glob("*_clap.npz")):
            exp = p.name[:-len("_clap.npz")]
            if (case, exp) not in clap or (case, exp) not in enc:
                continue
            i, q = scores(np.load(p)["emb"], T, case)
            res[exp] = dict(
                clap=clap[(case, exp)], encodec=enc[(case, exp)],
                position=(float(np.mean(i)) - a["i_src"]) / span if span else float("nan"),
                position_se=float(np.std(i) / np.sqrt(len(i))) / abs(span) if span else float("nan"),
                noisiness=(float(np.mean(q)) - float(np.mean(q_ref))) / a["q_shift"] if a["q_shift"] else float("nan"),
                noisiness_se=float(np.std(q) / np.sqrt(len(q))) / abs(a["q_shift"]) if a["q_shift"] else float("nan"))
        exps = sorted(res, key=lambda e: res[e]["clap"])
        print("\n" + "=" * 100)
        print("  %s  (sorted by CLAP-FAD)" % label)
        print("  instrument position: 0 = sounds like the real source, 1 = like the real target (to CLAP)")
        print("  noisiness: 0 = as clean as real target recordings, 1 = as noisy as our 'old recording'")
        print("=" * 100)
        print("  %-34s %9s %9s %16s %16s" % ("condition", "CLAP-FAD", "EnC-FAD", "instr. position", "noisiness"))
        for e in exps:
            r = res[e]
            print("  %-34s %9.4f %9.3f %9.3f +-%.3f %9.3f +-%.3f"
                  % (e, r["clap"], r["encodec"], r["position"], r["position_se"], r["noisiness"], r["noisiness_se"]))
            cond_rows.append(dict(group=label, condition=e, **r))
        c, en = [res[e]["clap"] for e in exps], [res[e]["encodec"] for e in exps]
        far = [-res[e]["position"] for e in exps]
        noisy = [res[e]["noisiness"] for e in exps]
        print("\n  Spearman across the %d conditions (FAD vs distance from target / vs noisiness):" % len(exps))
        print("    CLAP-FAD    vs not-target %+.2f   vs noisiness %+.2f" % (spearman(c, far), spearman(c, noisy)))
        print("    EnCodec-FAD vs not-target %+.2f   vs noisiness %+.2f" % (spearman(en, far), spearman(en, noisy)))

        for x, y in itertools.combinations(exps, 2):
            if np.sign(res[x]["clap"] - res[y]["clap"]) == np.sign(res[x]["encodec"] - res[y]["encodec"]):
                continue
            cp = x if res[x]["clap"] < res[y]["clap"] else y
            ep = y if cp == x else x

            def clear(k):
                d = res[cp][k] - res[ep][k]
                return abs(d) > 2 * np.hypot(res[cp][k + "_se"], res[ep][k + "_se"])
            pair_rows.append(dict(
                group=label, clap_pick=cp, encodec_pick=ep,
                robust="" if robust is None else int((label, frozenset((cp, ep))) in robust),
                clap_pick_cleaner=int(res[cp]["noisiness"] < res[ep]["noisiness"]), noisiness_clear=int(clear("noisiness")),
                clap_pick_more_target=int(res[cp]["position"] > res[ep]["position"]), position_clear=int(clear("position")),
                d_noisiness=res[cp]["noisiness"] - res[ep]["noisiness"], d_position=res[cp]["position"] - res[ep]["position"]))

    # ---- the question
    print("\n" + "=" * 100)
    print("  Pairs the two FADs order differently: what does CLAP's pick have that EnCodec's lacks?")
    print("=" * 100)
    sets = [("all disagreeing pairs", pair_rows)]
    if robust is not None:
        sets.append(("robust ones only (diagnose_fad.py: P >= 0.95 / <= 0.05)", [r for r in pair_rows if r["robust"] == 1]))
    for name, rows in sets:
        n = len(rows)
        if not n:
            print("  %s: none" % name)
            continue
        cl = sum(r["clap_pick_cleaner"] for r in rows)
        cl_c = sum(r["clap_pick_cleaner"] and r["noisiness_clear"] for r in rows)
        mt = sum(r["clap_pick_more_target"] for r in rows)
        mt_c = sum(r["clap_pick_more_target"] and r["position_clear"] for r in rows)
        print("  %s: %d" % (name, n))
        print("    CLAP's pick is the CLEANER one        %3d (%3.0f%%), clearly so in %d" % (cl, 100 * cl / n, cl_c))
        print("    CLAP's pick is the MORE TARGET-LIKE   %3d (%3.0f%%), clearly so in %d" % (mt, 100 * mt / n, mt_c))
    if not all(a["ok_inst"] and a["ok_q"] for a in anchors.values()):
        print("\n  !! A probe failed its check (section 0). Do not read the rows it feeds.")

    for name, rows in (("conditions.csv", cond_rows), ("pairs.csv", pair_rows)):
        if rows:
            with (out / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print("\n-> %s/{conditions.csv,pairs.csv}" % out)


if __name__ == "__main__":
    main()
