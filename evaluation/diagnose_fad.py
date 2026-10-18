"""Why do CLAP-FAD and EnCodec-FAD rank some models in opposite order?

Reads only the embeddings the FAD scripts cached -- no GPU, no audio decoded,
nothing regenerated -- and puts one hypothesis to the test.

**Hypothesis.** The CLAP reference holds ~290-310 clips in 512 dimensions, so
its sample covariance has rank <= n-1 and is exactly zero in 200+ directions.
In those directions a Frechet distance charges every unit of the generated
set's variance (and mean offset) in full, with nothing to cancel it: a model
with more varied output loses, one with more uniform output wins, whatever it
sounds like. That bias differs from model to model, so it can flip a ranking.
EnCodec (128-D against the same ~300 clips) is not singular.

**What would support it** (written down before the numbers were seen):
  1. For CLAP, a large share of each FAD sits in the null space of the
     reference, and that part differs between models by as much as the
     reported gaps. For EnCodec the null-space part is zero.
  2. CLAP's ranking of a disagreeing pair moves with the reference size;
     EnCodec's ranking of the same pair does not, once n > 128.
  3. With a covariance estimate that is not singular (Ledoit-Wolf shrinkage,
     or a k-dimensional projection with k well below n), CLAP agrees with
     EnCodec on the pairs where it disagreed.

**What would refute it:** CLAP keeps its order under every estimator and every
reference size. Then the disagreement is real -- the two embedders hear
different things -- and both numbers are reported as they are.

Every estimator is printed, whichever way it points; none is chosen after the
fact. Values from the projections and from shrinkage are on other scales than
the reported FAD: read their ORDER only.

Also checked, because a wrong file would make the rest meaningless:
  - each reported FAD is reproduced from the cache with the original code;
  - the cache matches the clips now on disk;
  - how many input clips the models of one group share;
  - whether CLAP's preprocessing crops each clip at random (--crop-check).

  venv_yuval/bin/python evaluation/diagnose_fad.py
  venv_yuval/bin/python evaluation/diagnose_fad.py --crop-check 12
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOTS = ["Outputs/WAV_files/clap_validation_1000", "Outputs/WAV_files/merged_midi_test"]
# embedder -> (results CSV under <root>/fad/, its value column)
EMBEDDERS = {"encodec": ("encodec_fad_results.csv", "fad"),
             "clap": ("clap_fad_200.csv", "clap_fad")}
PCA_KS = (16, 32, 64, 128)
REF_SIZES = (64, 128, 192, 256)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    p.add_argument("--reference-name", default="reference")
    p.add_argument("--subsets", type=int, default=30,
                   help="random reference subsets per size for the size curve")
    p.add_argument("--resamples", type=int, default=50,
                   help="joint resamples for P(A better than B): 80%% of the reference "
                        "without replacement, every generated set bootstrapped")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--crop-check", type=int, default=0,
                   help="embed this many reference clips twice with CLAP on the CPU and "
                        "report how much of the embedding spread is crop randomness")
    p.add_argument("--out-dir", default="Outputs/fad_diagnosis")
    return p.parse_args()


# ------------------------------------------------------------------ statistics
def frechet(mu_r, A, mu_g, S_g):
    """Exact Frechet distance, no regularisation. The reference covariance is
    passed as a factor A with Sigma_r = A.T @ A, so a singular reference costs
    nothing: the nonzero eigenvalues of Sigma_r^1/2 Sigma_g Sigma_r^1/2 are those
    of A Sigma_g A.T."""
    M = A @ S_g @ A.T
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    d = mu_r - mu_g
    return float(d @ d + np.einsum("ij,ij->", A, A) + np.trace(S_g) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


def full_rank(A, rel_tol=1e-10):
    """The same A.T @ A from a factor with only as many rows as its rank. Without
    this, A Sigma_g A.T has exact zero eigenvalues whose rounding noise, square
    rooted, adds up to a visible error."""
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = int((s > rel_tol * s[0]).sum()) if s.size and s[0] > 0 else 0
    return s[:k, None] * Vt[:k]


def sample_factor(X):
    """mean, and A with A.T @ A == np.cov(X, rowvar=False)."""
    mu = X.mean(0)
    return mu, full_rank((X - mu) / np.sqrt(X.shape[0] - 1))


def matrix_factor(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return full_rank((V * np.sqrt(np.clip(w, 0, None))).T)


def ledoit_wolf(X):
    """Ledoit & Wolf (2004) shrinkage towards a scaled identity; returns (Sigma, intensity)."""
    n, d = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    m = np.trace(S) / d
    d2 = ((S - m * np.eye(d)) ** 2).sum()
    b2 = min((np.einsum("ij,ij->i", Xc, Xc) ** 2).sum() / n ** 2 - (S ** 2).sum() / n, d2)
    a = b2 / d2 if d2 > 0 else 1.0
    return a * m * np.eye(d) + (1 - a) * S, a


def null_basis(Xr, rel_tol=1e-9):
    """Orthonormal basis of the directions where the reference sample has no variance."""
    _, s, Vt = np.linalg.svd(Xr - Xr.mean(0), full_matrices=True)
    return Vt[int((s > rel_tol * s[0]).sum()):].T


def null_space_part(N, mu_r, mu_g, S_g):
    """The generated variance and squared mean offset in the null directions N.
    Exactly: FAD = FAD(projected onto the reference span) + null_var + null_mean."""
    if N.shape[1] == 0:
        return 0.0, 0.0
    off = (mu_g - mu_r) @ N
    return float(np.trace(N.T @ S_g @ N)), float(off @ off)


# ------------------------------------------------------------------ data
def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return z["emb"].astype(np.float64), [str(n) for n in z["names"]]


def read_reported(path, value_col):
    if not path.is_file():
        return {}
    with io.open(path, newline="", encoding="utf-8") as f:
        return {(r["case"], r["experiment"]): float(r[value_col]) for r in csv.DictReader(f)
                if r.get(value_col) not in (None, "")}


def condition_key(exp):
    m = re.search(r"_h(\d+)$", exp)
    base = exp[:m.start()] if m else exp
    handoff = int(m.group(1)) if m else (100 if base.endswith("_s100") else 5)
    return base, handoff


def discover(root, ref_name):
    """[(case, {embedder: (X_ref, ref_names, {exp: (X, names)})})] for one root."""
    groups = []
    for case_dir in sorted((Path(root) / "fad").glob("*")):
        if not (case_dir / "generated").is_dir():
            continue
        data = {}
        for emb in EMBEDDERS:
            ref = case_dir / ("%s_%s.npz" % (ref_name, emb))
            if not ref.is_file():
                continue
            suffix = "_%s.npz" % emb
            gens = {p.name[:-len(suffix)]: load_npz(p) for p in sorted((case_dir / "generated").glob("*" + suffix))}
            data[emb] = (*load_npz(ref), gens)
        if data:
            groups.append((case_dir, data))
    return groups


def on_disk(dirpath):
    return sorted(p.name for p in Path(dirpath).glob("*.wav"))


# ------------------------------------------------------------------ report
def fmt(v, width=8):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--".rjust(width)
    return ("%.4f" % v if abs(v) < 10 else "%.3f" % v).rjust(width)


def table(title, header, rows):
    print("\n" + title)
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = "  ".join(str(h).rjust(w) if i else str(h).ljust(w) for i, (h, w) in enumerate(zip(header, widths)))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths))))


def analyse_group(case_dir, data, args, rng, out_rows, pair_rows):
    case = case_dir.name
    root = case_dir.parent.parent
    label = "%s / %s" % (root.name, case)
    embedders = [e for e in EMBEDDERS if e in data]
    exps = sorted(set.intersection(*(set(data[e][2]) for e in embedders)), key=condition_key)
    print("\n" + "=" * 100)
    print("  %s   reference: %s" % (label, case_dir / args.reference_name))
    print("=" * 100)

    # ---- 0. the files are the ones behind the report
    problems = []
    ref_names = data[embedders[0]][1]
    for e in embedders:
        if data[e][1] != ref_names:
            problems.append("reference clips differ between %s and %s caches" % (embedders[0], e))
        for exp in exps:
            if data[e][2][exp][1] != data[embedders[0]][2][exp][1]:
                problems.append("%s: generated clips differ between embedder caches" % exp)
    if on_disk(case_dir / args.reference_name) != ref_names:
        problems.append("reference cache does not match the clips now in %s" % (case_dir / args.reference_name))
    for exp in exps:
        if on_disk(case_dir / "generated" / exp) != data[embedders[0]][2][exp][1]:
            problems.append("%s: cache does not match the clips now on disk" % exp)
    from evaluation.compute_clap_fad import frechet_distance  # the code the report used
    reported = {e: read_reported(root / "fad" / EMBEDDERS[e][0], EMBEDDERS[e][1]) for e in embedders}
    n_ref = len(ref_names)
    sizes = [m for m in REF_SIZES if m < n_ref]
    print("  reference: %d clips | %d conditions | %s" % (n_ref, len(exps), ", ".join(embedders)))

    names = {exp: set(data[embedders[0]][2][exp][1]) for exp in exps}
    common = set.intersection(*names.values()) if names else set()
    print("  generated per condition: %s clips; input clips shared by ALL conditions: %d"
          % ("/".join(sorted({str(len(v)) for v in names.values()})), len(common)))

    # ---- shared resampling indices, so both embedders see the same clips
    subsets = {m: [rng.choice(n_ref, m, replace=False) for _ in range(args.subsets)] for m in sizes}
    m80 = int(round(0.8 * n_ref))
    joint = [(rng.choice(n_ref, m80, replace=False),
              {exp: rng.integers(0, len(names[exp]), len(names[exp])) for exp in exps})
             for _ in range(args.resamples)]

    res = {}
    for e in embedders:
        Xr, _, gens = data[e]
        d = Xr.shape[1]
        mu_r, A_r = sample_factor(Xr)
        mom = {exp: (gens[exp][0].mean(0), np.cov(gens[exp][0], rowvar=False)) for exp in exps}
        S_r_lw, a_ref = ledoit_wolf(Xr)
        A_r_lw = matrix_factor(S_r_lw)
        # PCA basis that favours no condition: covariance of an equal-weight mixture
        mus = [mu_r] + [mom[x][0] for x in exps]
        covs = [np.cov(Xr, rowvar=False)] + [mom[x][1] for x in exps]
        mbar = np.mean(mus, axis=0)
        P = np.mean([c + np.outer(m - mbar, m - mbar) for m, c in zip(mus, covs)], axis=0)
        W = np.linalg.eigh(P)[1][:, ::-1]
        ks = [k for k in PCA_KS if k <= d]
        N = null_basis(Xr)
        print("  %s: %d-D, reference rank %d, Ledoit-Wolf shrinkage on the reference %.2f"
              % (e, d, d - N.shape[1], a_ref))
        # everything that depends on the reference alone, once
        pca_ref = {k: sample_factor(Xr @ W[:, :k]) for k in ks}
        sub_ref = {m: [sample_factor(Xr[idx]) for idx in subsets[m]] for m in sizes}

        r = {}
        for exp in exps:
            Xg = gens[exp][0]
            mu_g, S_g = mom[exp]
            row = {"reported": reported[e].get((case, exp)),
                   "recomputed": frechet_distance(Xr, Xg),
                   "exact": frechet(mu_r, A_r, mu_g, S_g)}
            nv, nm = null_space_part(N, mu_r, mu_g, S_g)
            row.update(null_var=nv, null_mean=nm, null_share=(nv + nm) / row["exact"])
            S_g_lw, _ = ledoit_wolf(Xg)
            row["lw"] = frechet(mu_r, A_r_lw, mu_g, S_g_lw)
            for k in ks:
                Wk = W[:, :k]
                row["pca%d" % k] = frechet(*pca_ref[k], mu_g @ Wk, Wk.T @ S_g @ Wk)
            for m in sizes:
                row["n%d" % m] = float(np.mean([frechet(mr, Ar, mu_g, S_g) for mr, Ar in sub_ref[m]]))
            r[exp] = row
            if row["reported"] is not None and abs(row["recomputed"] - row["reported"]) > 1e-6 * max(1, row["reported"]):
                problems.append("%s %s: recomputed %.6f, report says %.6f" % (e, exp, row["recomputed"], row["reported"]))
        # joint resamples
        boot = {exp: [] for exp in exps}
        for ridx, gidx in joint:
            mr, Ar = sample_factor(Xr[ridx])
            for exp in exps:
                Xg = gens[exp][0][gidx[exp]]
                boot[exp].append(frechet(mr, Ar, Xg.mean(0), np.cov(Xg, rowvar=False)))
        for exp in exps:
            r[exp]["boot"] = np.array(boot[exp])
            r[exp]["boot_sd"] = float(np.std(boot[exp]))
        res[e] = (r, ks)

        cols = ["reported", "exact", "null_share", "lw"] + ["pca%d" % k for k in ks] + ["n%d" % m for m in sizes]
        table("  %s-FAD by estimator -- compare down a column (ORDER), never across.\n"
              "  +-sd: spread over the joint resamples. no eps: the same FAD without the 1e-6 the\n"
              "  original code adds. null %%: share of it where the reference has no variance.\n"
              "  n=: mean over random reference subsets of that size, generated set whole." % e.upper(),
              ["condition", "reported", "+-sd", "no eps", "null %", "L-W"] + ["PCA%d" % k for k in ks]
              + ["n=%d" % m for m in sizes],
              [[exp, fmt(r[exp]["reported"]), fmt(r[exp]["boot_sd"]), fmt(r[exp]["exact"]),
                "%5.1f%%" % (100 * r[exp]["null_share"]), fmt(r[exp]["lw"])]
               + [fmt(r[exp]["pca%d" % k]) for k in ks] + [fmt(r[exp]["n%d" % m]) for m in sizes]
               for exp in exps])
        for exp in exps:
            out_rows.append(dict(group=label, embedder=e, condition=exp,
                                 **{c: r[exp][c] for c in cols + ["recomputed", "null_var", "null_mean", "boot_sd"]}))

    # ---- pairs the two metrics order differently
    if {"clap", "encodec"} <= set(res):
        enc, clap = res["encodec"][0], res["clap"][0]
        ks = res["clap"][1]
        est = ["exact", "lw"] + ["pca%d" % k for k in ks] + ["n%d" % m for m in sizes]
        rows = []
        n_pairs = 0
        for a, b in itertools.combinations(exps, 2):
            n_pairs += 1
            e_sign = np.sign(enc[a]["exact"] - enc[b]["exact"])
            if np.sign(clap[a]["exact"] - clap[b]["exact"]) == e_sign:
                continue
            better = a if e_sign < 0 else b
            worse = b if better == a else a
            agree = ["yes" if np.sign(clap[better][k] - clap[worse][k]) < 0 else "no" for k in est]
            p_enc = float(np.mean(enc[better]["boot"] < enc[worse]["boot"]))
            p_clap = float(np.mean(clap[better]["boot"] < clap[worse]["boot"]))
            dn = ((clap[better]["null_var"] + clap[better]["null_mean"])
                  - (clap[worse]["null_var"] + clap[worse]["null_mean"]))
            rows.append([better + " < " + worse,
                         "%.3f" % (enc[worse]["exact"] - enc[better]["exact"]),
                         "%.4f" % (clap[better]["exact"] - clap[worse]["exact"]),
                         "%+.4f" % dn, "%.2f" % p_enc, "%.2f" % p_clap] + agree)
            pair_rows.append(dict(group=label, encodec_better=better, encodec_worse=worse,
                                  encodec_gap=enc[worse]["exact"] - enc[better]["exact"],
                                  clap_gap=clap[better]["exact"] - clap[worse]["exact"],
                                  clap_null_gap=dn, p_encodec=p_enc, p_clap_same_as_encodec=p_clap,
                                  **{"clap_agrees_" + k: v for k, v in zip(est, agree)}))
        print("\n  %d of %d pairs are ordered differently by the two metrics." % (len(rows), n_pairs))
        if rows:
            table("  Disagreeing pairs. 'A < B' = EnCodec says A is closer to the real target.\n"
                  "  CLAP gap > 0 means CLAP says the opposite. 'null gap' = the part of CLAP's\n"
                  "  A-minus-B that sits where the reference has no variance; if it is near the\n"
                  "  whole CLAP gap, the singular covariance alone made the flip.\n"
                  "  P(...) = share of joint resamples in which that metric puts A first.\n"
                  "  The last columns: does CLAP put A first under that estimator?",
                  ["EnCodec order", "Enc gap", "CLAP gap", "null gap", "P enc", "P clap",
                   "no eps", "L-W"] + ["PCA%d" % k for k in ks] + ["n=%d" % m for m in sizes], rows)

    if problems:
        print("\n  !! CHECKS FAILED -- do not read this group until these are explained:")
        for p in problems:
            print("     - " + p)
    else:
        print("\n  checks: every reported value reproduced from the cache; caches match the clips on disk.")


def crop_check(roots, ref_name, n):
    """Embed n reference clips twice. If CLAP crops at random, the two embeddings
    of one clip differ; compare that spread with the spread between clips."""
    import torch
    from transformers import ClapModel, ClapProcessor
    from evaluation.compute_clap_fad import MODEL_NAME, SAMPLE_RATE, load_audio

    proc = ClapProcessor.from_pretrained(MODEL_NAME)
    fe = proc.feature_extractor
    print("\n" + "=" * 100)
    print("  CLAP preprocessing: truncation=%r padding=%r, window %.1f s"
          % (getattr(fe, "truncation", None), getattr(fe, "padding", None),
             getattr(fe, "nb_max_samples", float("nan")) / SAMPLE_RATE))
    ref_dir = next((Path(r) / "fad" / "poly" / ref_name for r in roots
                    if (Path(r) / "fad" / "poly" / ref_name).is_dir()), None)
    if ref_dir is None:
        print("  no poly reference directory found -- crop check skipped")
        return
    paths = sorted(ref_dir.glob("*.wav"))[:n]
    model = ClapModel.from_pretrained(MODEL_NAME).eval()
    twice = []
    for p in paths:
        audio = load_audio(p)
        pair = []
        for _ in range(2):
            inputs = proc(audios=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                pair.append(model.get_audio_features(**inputs).numpy()[0])
        twice.append(pair)
    E = np.array(twice)                                  # [n, 2, 512]
    within = float(np.mean(((E[:, 0] - E[:, 1]) ** 2).sum(1)) / 2)   # crop variance per clip
    total = float(np.trace(np.cov(E[:, 0], rowvar=False)))
    print("  %d clips of %.1f s, each embedded twice: identical=%s" % (
        len(paths), len(load_audio(paths[0])) / SAMPLE_RATE, bool(np.allclose(E[:, 0], E[:, 1]))))
    print("  crop variance within a clip %.5f vs total variance between clips %.5f -> %.0f%% of the spread"
          % (within, total, 100 * within / total if total else float("nan")))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_rows, pair_rows = [], []
    for root in args.roots:
        groups = discover(root, args.reference_name)
        if not groups:
            print("nothing cached under %s/fad -- skipped" % root)
        for case_dir, data in groups:
            analyse_group(case_dir, data, args, rng, out_rows, pair_rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("estimators.csv", out_rows), ("disagreeing_pairs.csv", pair_rows)):
        if rows:
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
    print("\n-> %s/{estimators.csv,disagreeing_pairs.csv}" % out)

    if args.crop_check:   # last: it loads the CLAP model, and the tables are saved already
        try:
            crop_check(args.roots, args.reference_name, args.crop_check)
        except Exception as exc:  # noqa: BLE001 -- report and keep the results above
            print("\n  crop check failed: %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
