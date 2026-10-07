"""Summary of the 200-clip handoff sweep: one table, paired tests, one plot per metric.

Reported set (the rest of the columns are by-products):
  poly  flute_bassoon -> violin_cello   mir_eval transcription P / R / F1 + CLAP-FAD
  mono  cello -> bassoon                DPD (JD alongside)                + CLAP-FAD

For every sigma_handoff, MIDI vs no-MIDI is tested on the paired clips with
evaluation/paired_stats.py (paired_deltas for the pairing, report for the
printed verdict). FAD has one number per condition, so it is not paired.

  python evaluation/summarize_200.py
  python evaluation/summarize_200.py --out-root /home/chenyuv/metrics_200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bridge_200 import DEFAULT_OUT_ROOT  # noqa: E402

PRIMARY = {
    "poly": ["transcription_f1", "transcription_precision", "transcription_recall"],
    "mono": ["DPD", "JD"],
}
BYPRODUCTS = ["DPD", "JD", "transcription_f1", "transcription_precision",
              "transcription_recall", "multipitch_f1"]
HIGHER_IS_BETTER = {"transcription_f1", "transcription_precision", "transcription_recall",
                    "multipitch_f1", "multipitch_precision", "multipitch_recall"}
CASE_TITLE = {"poly": "flute+bassoon -> violin+cello", "mono": "cello -> bassoon"}
ID_COLS = ["case", "experiment", "pair", "use_midi", "trained_sigma", "sigma_handoff"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--bridge-csv", default=None)
    p.add_argument("--content-glob", default=None, help="default: <out-root>/content/*.csv")
    p.add_argument("--fad-csv", default=None)
    p.add_argument("--ceiling-poly", default=None)
    p.add_argument("--ceiling-mono", default=None)
    p.add_argument("--summary-dir", default=None)
    return p.parse_args()


def as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1"])


def load_per_clip(bridge_csv, content_paths):
    df = pd.read_csv(bridge_csv)
    df["use_midi"] = as_bool(df["use_midi"])
    for col in ("DPD", "JD", "trained_sigma", "sigma_handoff"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if content_paths:
        content = pd.concat([pd.read_csv(p) for p in content_paths], ignore_index=True)
        content = content.drop_duplicates(["experiment", "sample_id"], keep="last")
        keep = ["experiment", "sample_id"] + [c for c in content.columns
                                              if c.startswith(("transcription_", "multipitch_"))]
        df = df.merge(content[keep], on=["experiment", "sample_id"], how="left")
    return df


def ci95(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float("nan")
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def condition_table(df, fad):
    metrics = [m for m in ["transcription_f1", "transcription_precision", "transcription_recall",
                           "DPD", "JD", "multipitch_f1"] if m in df.columns]
    rows = []
    for key, g in df.groupby(ID_COLS, sort=True):
        row = dict(zip(ID_COLS, key))
        row["n"] = len(g)
        row["reported"] = " + ".join(PRIMARY.get(row["case"], []) + ["clap_fad"])
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[m + "_n"] = int(vals.notna().sum())
            row[m + "_mean"] = vals.mean()
            row[m + "_sd"] = vals.std(ddof=1)
        rows.append(row)
    table = pd.DataFrame(rows)
    if fad is not None and not table.empty:
        f = fad[fad["experiment"] != "sanity_reference_halves"][
            ["experiment", "clap_fad", "n_reference", "n_generated"]]
        f = f.rename(columns={"n_reference": "clap_fad_n_reference",
                              "n_generated": "clap_fad_n_generated"})
        table = table.merge(f, on="experiment", how="left")
    return table.sort_values(["case", "trained_sigma", "use_midi", "sigma_handoff"],
                             ascending=[True, False, False, True])


def paired_table(df):
    from scipy import stats
    from evaluation.paired_stats import paired_deltas, report

    rows = []
    for (case, trained, handoff), g in df.groupby(["case", "trained_sigma", "sigma_handoff"]):
        midi = g[g["use_midi"]]["experiment"].unique()
        nomidi = g[~g["use_midi"]]["experiment"].unique()
        if len(midi) != 1 or len(nomidi) != 1:
            continue
        a, b = nomidi[0], midi[0]
        metrics = [m for m in PRIMARY.get(case, []) + ["multipitch_f1"] if m in g.columns
                   and g[m].notna().any()]
        for metric in metrics:
            sub = g[["experiment", "sample_id", metric]].dropna()
            va, vb, dropped = paired_deltas(sub, metric, a, b)
            higher = metric in HIGHER_IS_BETTER
            # Printed verdict from paired_stats.report, which assumes lower is
            # better; negate higher-is-better metrics so "b is better" stays true.
            sign = -1.0 if higher else 1.0
            label = "%s%s  [trained %g, handoff %g]" % (
                metric, " (negated: higher is better)" if higher else "", trained, handoff)
            report(label, sign * va, sign * vb, dropped, a, b)

            d = vb - va
            n = len(d)
            improvement = d if higher else -d
            wins = int((improvement > 0).sum())
            ties = int((improvement == 0).sum())
            decided = n - ties
            p_sign = stats.binomtest(wins, decided, 0.5).pvalue if decided else float("nan")
            try:
                p_wil = stats.wilcoxon(d).pvalue if n >= 3 else float("nan")
            except ValueError:
                p_wil = float("nan")
            sem = stats.sem(d) if n >= 2 else float("nan")
            lo, hi = (stats.t.interval(0.95, n - 1, loc=d.mean(), scale=sem)
                      if n >= 3 and sem > 0 else (float("nan"), float("nan")))
            sd = d.std(ddof=1) if n >= 2 else float("nan")
            rows.append({
                "case": case, "trained_sigma": trained, "sigma_handoff": handoff,
                "metric": metric, "higher_is_better": higher,
                "no_midi_experiment": a, "midi_experiment": b, "n_pairs": n, "dropped": dropped,
                "no_midi_mean": va.mean() if n else float("nan"),
                "midi_mean": vb.mean() if n else float("nan"),
                "delta_midi_minus_no_midi": d.mean() if n else float("nan"),
                "delta_ci95_lo": lo, "delta_ci95_hi": hi,
                "midi_better_on": wins, "ties": ties,
                "p_sign": p_sign, "p_wilcoxon": p_wil,
                "dz": (d.mean() / sd) if n >= 2 and sd > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def ceiling_means(path):
    if path is None or not Path(path).is_file():
        return {}
    c = pd.read_csv(path)
    return {col: float(c[col].mean()) for col in c.columns if col.startswith("transcription_")}


def plot_metric(df, table, case, metric, out_png, ceiling=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table[table["case"] == case]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for use_midi, color, label in ((True, "tab:blue", "MIDI"), (False, "tab:orange", "no MIDI")):
        sweep = t[(t["use_midi"] == use_midi) & (t["trained_sigma"] == t["trained_sigma"].max())]
        sweep = sweep.sort_values("sigma_handoff")
        if metric == "clap_fad":
            if "clap_fad" not in sweep or sweep["clap_fad"].isna().all():
                continue
            ax.plot(sweep["sigma_handoff"], sweep["clap_fad"], "o-", color=color, label=label)
        else:
            if metric + "_mean" not in sweep:
                continue
            err = [ci95(df[(df["experiment"] == e)][metric]) for e in sweep["experiment"]]
            ax.errorbar(sweep["sigma_handoff"], sweep[metric + "_mean"], yerr=err, fmt="o-",
                        color=color, capsize=3, label=label)
        base = t[(t["use_midi"] == use_midi) & (t["trained_sigma"] < t["trained_sigma"].max())]
        col = "clap_fad" if metric == "clap_fad" else metric + "_mean"
        if not base.empty and col in base and base[col].notna().any():
            ax.plot(base["sigma_handoff"], base[col], "s", color=color, mfc="none", ms=8,
                    label="%s, trained sigma %g" % (label, base["trained_sigma"].iloc[0]))
    if ceiling is not None:
        ax.axhline(ceiling, color="gray", ls="--",
                   label="transcription ceiling (real input) %.3f" % ceiling)
    ax.set_xlabel("sigma_handoff")
    ax.set_ylabel(metric + ("" if metric == "clap_fad" else "  (mean, 95% CI)"))
    trained = t["trained_sigma"].max() if not t.empty else float("nan")
    ax.set_title("%s -- %s (models trained at sigma_max=%g)" % (CASE_TITLE.get(case, case), metric, trained),
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    bridge_csv = Path(args.bridge_csv) if args.bridge_csv else out_root / "bridge_metrics_all.csv"
    content_paths = sorted(Path(".").glob(args.content_glob)) if args.content_glob \
        else sorted((out_root / "content").glob("*.csv"))
    fad_csv = Path(args.fad_csv) if args.fad_csv else out_root / "fad" / "clap_fad_200.csv"
    ceilings = {
        "poly": ceiling_means(args.ceiling_poly or out_root / "ceiling" / "ceiling_poly.csv"),
        "mono": ceiling_means(args.ceiling_mono or out_root / "ceiling" / "ceiling_mono.csv"),
    }
    summary_dir = Path(args.summary_dir) if args.summary_dir else out_root / "summary"
    plot_dir = summary_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_per_clip(bridge_csv, content_paths)
    fad = pd.read_csv(fad_csv) if fad_csv.is_file() else None
    if fad is None:
        print("(no CLAP-FAD CSV at %s -- table and plots without FAD)" % fad_csv)
    if not content_paths:
        print("(no content-metric CSVs -- poly mir_eval columns will be missing)")

    table = condition_table(df, fad)
    table_path = summary_dir / "summary_200.csv"
    table.to_csv(table_path, index=False)

    paired = paired_table(df)
    paired_path = summary_dir / "paired_midi_vs_nomidi_200.csv"
    paired.to_csv(paired_path, index=False)

    for case in sorted(df["case"].unique()):
        for metric in PRIMARY.get(case, []) + ["clap_fad"]:
            if metric != "clap_fad" and metric not in df.columns:
                continue
            out_png = plot_dir / ("%s_%s_vs_handoff.png" % (case, metric))
            plot_metric(df, table, case, metric, out_png, ceilings.get(case, {}).get(metric))
            print("plot -> %s" % out_png)

    pd.set_option("display.width", 200)
    for case in sorted(table["case"].unique()):
        cols = ["experiment", "n"] + [m + s for m in PRIMARY.get(case, []) for s in ("_mean", "_sd")
                                      if m + s in table.columns]
        if "clap_fad" in table.columns:
            cols += ["clap_fad", "clap_fad_n_reference", "clap_fad_n_generated"]
        print("\n=== %s: %s ===" % (case, CASE_TITLE.get(case, case)))
        print(table[table["case"] == case][cols].to_string(index=False, float_format="%.4f"))
        if ceilings.get(case):
            print("transcription ceiling (real input vs GT MIDI): %s"
                  % ", ".join("%s %.3f" % kv for kv in sorted(ceilings[case].items())))
    print("\nCLAP-FAD at n=200 in 512-D is noisy; compare conditions, not absolute values.")
    print("table  -> %s\npaired -> %s\nplots  -> %s" % (table_path, paired_path, plot_dir))


if __name__ == "__main__":
    main()
