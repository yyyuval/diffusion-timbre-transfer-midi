"""The roll-swap test: does the EnCodec-FAD follow the roll or the weights?

Four conditions, all sigma_max = 100 at handoff 100, all on set B's inputs:

  merged_real_midi_s100           real-trained models,      real roll       (existing)
  merged_generated_midi_s100      generated-trained models, generated roll  (existing)
  swap_realtrained_genroll_s100   real-trained models,      generated roll  (new)
  swap_gentrained_realroll_s100   generated-trained models, real roll       (new)

Three steps, run by evaluation/run_roll_swap.sh:

  --prepare  link each new condition's clips into fad/poly/generated/<exp>/ (the
             layout compute_encodec_fad.py reads) and write the bridge rows of the
             four conditions on the 200 content clips, for diagnose_false_notes.py
  --report   REPORT.md and results.csv from the content CSVs, the cached EnCodec
             embeddings, the false-note diagnosis and compare_rolls.py's output

The FAD spread is the one the project book reports: 50 joint resamples, 80% of
the reference without replacement, each generated set bootstrapped -- with the
same draws for every condition, so P(A < B) is a paired comparison.

  python evaluation/report_roll_swap.py --prepare
  python evaluation/report_roll_swap.py --report
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL = ["merged_real_midi_s100", "merged_generated_midi_s100"]
SWAP = ["swap_realtrained_genroll_s100", "swap_gentrained_realroll_s100"]
CONDITIONS = ORIGINAL + SWAP
LABEL = {
    "merged_real_midi_s100": ("real", "real"),
    "merged_generated_midi_s100": ("generated", "generated"),
    "swap_realtrained_genroll_s100": ("real", "generated"),
    "swap_gentrained_realroll_s100": ("generated", "real"),
}
CEILING = 0.779   # basic-pitch on real violin + cello against its own MIDI
N_FAD = 1000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--report", action="store_true")
    p.add_argument("--original-only", action="store_true",
                   help="with --prepare: only the bridge rows of the two existing "
                        "conditions, so their transcriptions can be cached while the "
                        "new ones are still generating")
    p.add_argument("--root", default="Outputs/WAV_files/merged_midi_test")
    p.add_argument("--content-list", default="evaluation/merged_test_lists/content_poly.txt")
    p.add_argument("--resamples", type=int, default=50)
    p.add_argument("--boot", type=int, default=2000, help="bootstrap draws for the paired F1 CI")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def read_csv(path):
    return list(csv.DictReader(io.open(path, encoding="utf-8")))


def read_list(path):
    lines = io.open(path, encoding="utf-8").read().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


# ------------------------------------------------------------------ prepare

def prepare(args):
    root = Path(args.root)
    rows = read_csv(root / "bridge_metrics_all.csv")
    out = root / "roll_swap"
    out.mkdir(parents=True, exist_ok=True)

    for exp in ([] if args.original_only else SWAP):
        mine = [r for r in rows if r["experiment"] == exp]
        if len(mine) < N_FAD:
            raise SystemExit("%s has %d clips in bridge_metrics_all.csv, expected %d -- "
                             "generation is incomplete" % (exp, len(mine), N_FAD))
        gen_dir = root / "fad" / "poly" / "generated" / exp
        gen_dir.mkdir(parents=True, exist_ok=True)
        made = 0
        for r in mine:
            src = Path(r["output_audio"])
            if not src.is_absolute():
                src = Path.cwd() / src
            dst = gen_dir / ("%s.wav" % r["sample_id"])
            if dst.is_symlink() or dst.exists():
                continue
            if not src.is_file():
                raise SystemExit("missing clip %s" % src)
            os.symlink(src, dst)
            made += 1
        print("%s: %d clips linked into %s (%d new)" % (exp, len(mine), gen_dir, made))

    # The ORIGINAL dirs must hold the clips the book's FAD was computed on.
    for exp in ORIGINAL:
        d = root / "fad" / "poly" / "generated" / exp
        n = len(list(d.glob("*.wav")))
        if n != N_FAD:
            raise SystemExit("%s holds %d clips, expected %d" % (d, n, N_FAD))

    content = set(read_list(args.content_list))
    wanted = ORIGINAL if args.original_only else CONDITIONS
    keep = [r for r in rows if r["experiment"] in wanted and r["sample_id"] in content]
    counts = {exp: sum(r["experiment"] == exp for r in keep) for exp in wanted}
    if set(counts.values()) != {len(content)}:
        raise SystemExit("content clips per condition: %s, expected %d each" % (counts, len(content)))
    outputs = [("bridge_rows_original.csv", ORIGINAL)]
    if not args.original_only:
        outputs.append(("bridge_rows_all.csv", CONDITIONS))
    for name, exps in outputs:
        with io.open(out / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(r for r in keep if r["experiment"] in exps)
    print("content rows for the false-note diagnosis: %s -> %s" % (counts, out))


# ------------------------------------------------------------------- report

def fad_table(args, rng):
    from evaluation.compute_clap_fad import frechet_distance

    fad_dir = Path(args.root) / "fad" / "poly"
    ref = np.load(fad_dir / "reference_encodec.npz", allow_pickle=False)
    Xr = ref["emb"]
    gens = {}
    for exp in CONDITIONS:
        z = np.load(fad_dir / "generated" / ("%s_encodec.npz" % exp), allow_pickle=False)
        gens[exp] = (z["emb"], list(z["names"]))
    names = gens[CONDITIONS[0]][1]
    paired = all(g[1] == names for g in gens.values())
    if not paired:
        print("WARNING: the four conditions do not hold the same input clips -- "
              "P(A < B) is then unpaired")

    point = {exp: frechet_distance(Xr, gens[exp][0]) for exp in CONDITIONS}
    # The existing two must reproduce the reported FAD, or the cache is not the
    # one the book used and nothing below can be compared with it.
    reported = {r["experiment"]: float(r["fad"])
                for r in read_csv(Path(args.root) / "fad" / "encodec_fad_results.csv")}
    for exp in ORIGINAL:
        if exp in reported and abs(point[exp] - reported[exp]) > 1e-6 * max(1.0, reported[exp]):
            raise SystemExit("%s: recomputed FAD %.6f, reported %.6f -- the cached embeddings "
                             "are not the ones behind the report" % (exp, point[exp], reported[exp]))
    m80 = int(round(0.8 * len(Xr)))
    boot = {exp: [] for exp in CONDITIONS}
    for _ in range(args.resamples):
        ridx = rng.choice(len(Xr), m80, replace=False)
        shared = rng.integers(0, len(names), len(names))
        for exp in CONDITIONS:
            X = gens[exp][0]
            gidx = shared if paired else rng.integers(0, len(X), len(X))
            boot[exp].append(frechet_distance(Xr[ridx], X[gidx]))
    boot = {k: np.array(v) for k, v in boot.items()}
    return point, boot, len(Xr), {k: len(v[0]) for k, v in gens.items()}


def content_table(args, rng):
    root = Path(args.root)
    per = {}
    for exp in CONDITIONS:
        rows = {r["sample_id"]: r for r in read_csv(root / "content" / ("%s.csv" % exp))}
        per[exp] = rows
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    stats = {}
    for exp in CONDITIONS:
        rs = [per[exp][s] for s in common]
        f = lambda k: np.array([float(r[k]) for r in rs])
        stats[exp] = dict(
            f1=f("transcription_f1"), p=f("transcription_precision").mean(),
            r=f("transcription_recall").mean(), frame_f1=f("multipitch_f1").mean(),
            n_est=f("n_est_notes").sum(), n_ref=f("n_ref_notes").sum())

    def paired(a, b):
        d = stats[a]["f1"] - stats[b]["f1"]
        idx = rng.integers(0, len(d), (args.boot, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        return d.mean(), lo, hi

    return stats, len(common), paired


def report(args):
    rng = np.random.default_rng(args.seed)
    root = Path(args.root)
    out = root / "roll_swap"
    point, boot, n_ref, n_gen = fad_table(args, rng)
    stats, n_content, paired = content_table(args, rng)

    def P(a, b):
        return float(np.mean(boot[a] < boot[b]))

    lines = ["# Roll swap: does EnCodec-FAD follow the roll or the weights?", "",
             "Two-instrument bridges, sigma_max = 100, handoff 100, set B. Note F1 on the "
             "%d paired content clips; EnCodec-FAD on %s generated clips per condition "
             "against %d real violin + cello clips, +- one SD over %d joint resamples."
             % (n_content, "/".join(sorted({str(v) for v in n_gen.values()})), n_ref, args.resamples), "",
             "| models trained on | roll at test time | note F1 | % of ceiling | P | R | "
             "frame F1 | est./ref. notes | EnCodec-FAD |",
             "|---|---|---|---|---|---|---|---|---|"]
    results = []
    for exp in CONDITIONS:
        s, (trained, roll) = stats[exp], LABEL[exp]
        sd = float(np.std(boot[exp]))
        lines.append("| %s | %s | %.3f | %.0f%% | %.3f | %.3f | %.3f | %.2f | %.2f +- %.2f |"
                     % (trained, roll, s["f1"].mean(), 100 * s["f1"].mean() / CEILING, s["p"], s["r"],
                        s["frame_f1"], s["n_est"] / s["n_ref"], point[exp], sd))
        results.append(dict(experiment=exp, trained_on=trained, roll=roll,
                            note_f1=s["f1"].mean(), pct_ceiling=s["f1"].mean() / CEILING,
                            precision=s["p"], recall=s["r"], frame_f1=s["frame_f1"],
                            est_over_ref_notes=s["n_est"] / s["n_ref"],
                            encodec_fad=point[exp], encodec_fad_sd=sd))

    real, gen, sw_rg, sw_gr = CONDITIONS
    lines += ["", "**Paired comparisons** (same input clips; F1: mean difference, 95% bootstrap CI; "
              "FAD: share of joint resamples in which the first is lower, i.e. closer to real violin + cello).", ""]
    for a, b, what in [(sw_rg, real, "real-trained models: generated roll vs real roll"),
                       (sw_gr, gen, "generated-trained models: real roll vs generated roll"),
                       (sw_gr, real, "real roll: generated-trained vs real-trained models"),
                       (sw_rg, gen, "generated roll: real-trained vs generated-trained models")]:
        d, lo, hi = paired(a, b)
        lines.append("- %s: note F1 %+.3f [%+.3f, %+.3f]; FAD %.2f vs %.2f, P(first lower) = %.2f"
                     % (what, d, lo, hi, point[a], point[b], P(a, b)))

    lines += ["", "**How to read it.** The first two rows reproduce the book. If the FAD "
              "follows the roll, the swap rows trade places with them: real-trained models "
              "with a generated roll land near %.2f, generated-trained models with a real "
              "roll near %.2f. If it follows the weights, each swap row stays near its own "
              "models' row. Anything in between is both." % (point[gen], point[real])]

    notes = out / "false_notes.csv"
    if notes.is_file():
        lines += ["", "## What kind of extra notes", "",
                  "Share of all estimated notes (false positives) and of all reference notes "
                  "(misses), from `diagnose_false_notes.py`. `ceiling_poly` is the real "
                  "flute + bassoon input, transcribed the same way: read each row against it.", "",
                  "| condition | clips | P | octave | reattack | late | spurious | merged | missed |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in read_csv(notes):
            lines.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
                         % (r["experiment"], r["clips"], float(r["precision"]), float(r["fp_octave"]),
                            float(r["fp_reattack"]), float(r["fp_late"]), float(r["fp_spurious"]),
                            float(r["fn_merged"]), float(r["fn_missed"])))

    rolls = out / "roll_comparison.csv"
    if rolls.is_file():
        lines += ["", "## How wrong is the generated roll itself", "",
                  "Generated merged roll against the real merged roll, cell by cell, first 1280 "
                  "frames. Flute + bassoon: the bridge's own test inputs. Violin + cello: what "
                  "the target model trained on. Extra-cell shares sum to 1.", "",
                  "| ensemble | tracks | P | R | F1 | extra: octave | harmonic | neighbour | other "
                  "| notes at once, real / gen | onsets per s, real / gen |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in read_csv(rolls):
            g = lambda k: float(r[k])
            lines.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f / %.2f | %.2f / %.2f |"
                         % (r["domain"], r["tracks"], g("precision"), g("recall"), g("f1"),
                            g("fp_share_octave"), g("fp_share_harmonic"), g("fp_share_neighbour"),
                            g("fp_share_other"), g("polyphony_real"), g("polyphony_gen"),
                            g("onsets_per_s_real"), g("onsets_per_s_gen")))

    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with io.open(out / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print("\n".join(lines))
    print("\n-> %s" % (out / "REPORT.md"))


def main():
    args = parse_args()
    if args.prepare:
        prepare(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
