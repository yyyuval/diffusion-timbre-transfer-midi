"""Is the difference between two conditions real, or is it the sample?

paired_compare.py reports the mean delta and the share of tracks where one
condition wins. Neither answers the only question that matters: 12 wins out of
20 is exactly what a coin gives. This adds the test.

Every track is run under both conditions, so the pairing is real and the
per-track delta cancels the track-to-track variance -- which is far larger than
the effect we are looking for. That is what makes n=20 workable here at all.

  python evaluation/paired_stats.py
  python evaluation/paired_stats.py --a nomidi_s100 --b midi_s100
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_deltas(df, metric, a, b):
    """One row per sample, keeping only samples that ran under BOTH conditions."""
    wide = df.pivot_table(
        index="sample_id", columns="experiment", values=metric, aggfunc="first"
    )
    missing = [c for c in (a, b) if c not in wide.columns]
    if missing:
        raise SystemExit(
            "experiment(s) %s not in the CSV. Present: %s"
            % (missing, sorted(df["experiment"].unique()))
        )
    both = wide[[a, b]].dropna()
    dropped = len(wide) - len(both)
    # a - b, so a POSITIVE delta means b scored lower. DPD and JD are both
    # distances, so lower is better and positive means b won.
    return both[a].to_numpy(), both[b].to_numpy(), dropped


def report(metric, va, vb, dropped, a, b):
    d = va - vb
    n = len(d)

    print("\n" + "=" * 68)
    print("  %s   (%s vs %s)   n=%d%s"
          % (metric, a, b, n, "  [%d sample(s) dropped: only one condition]" % dropped if dropped else ""))
    print("=" * 68)

    if n < 3:
        print("  too few paired samples to say anything.")
        return

    print("  %-18s %.4f   (sd %.4f)" % (a, va.mean(), va.std(ddof=1)))
    print("  %-18s %.4f   (sd %.4f)" % (b, vb.mean(), vb.std(ddof=1)))

    mean_d = d.mean()
    sem = stats.sem(d)
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean_d, scale=sem) if sem > 0 else (mean_d, mean_d)
    print("  mean delta (%s - %s)  %+.4f   95%% CI [%+.4f, %+.4f]" % (a, b, mean_d, lo, hi))

    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    print("  %s better on %d/%d tracks%s" % (b, wins, n, "  (%d exact ties)" % ties if ties else ""))

    # Sign test: ignores magnitude, so one freak track cannot carry the result.
    decided = n - ties
    p_sign = stats.binomtest(wins, decided, 0.5).pvalue if decided else 1.0

    # Wilcoxon: uses magnitude as well, more power when the effect is consistent.
    try:
        p_wil = stats.wilcoxon(d).pvalue
    except ValueError:
        p_wil = float("nan")   # every delta identical

    dz = mean_d / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    print("  sign test p = %.4f    Wilcoxon p = %.4f    effect size dz = %+.2f"
          % (p_sign, p_wil, dz))

    # The CI is the honest read: a null result with a wide CI means "we could
    # not measure it", which is a different statement from "there is no effect".
    p = min(p_sign, p_wil) if not np.isnan(p_wil) else p_sign
    if p < 0.05 and mean_d > 0:
        verdict = "%s is better. Real at p<0.05." % b
    elif p < 0.05:
        verdict = "%s is better. Real at p<0.05." % a
    else:
        half = (hi - lo) / 2
        verdict = ("no detectable difference. With n=%d this run could only have "
                   "resolved an effect of about %.4f or larger." % (n, half))
    print("  -> %s" % verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default="Outputs/WAV_files/metrics/inference_metrics.csv")
    ap.add_argument("--a", default="no_midi", help="baseline experiment name")
    ap.add_argument("--b", default="original_midi", help="treatment experiment name")
    ap.add_argument("--metrics", nargs="+", default=["DPD", "JD"])
    args = ap.parse_args()

    path = Path(args.metrics_csv)
    if not path.exists():
        raise SystemExit("no metrics CSV at %s -- run evaluation/run_compare_20.sh first" % path)

    df = pd.read_csv(path)
    for metric in args.metrics:
        if metric not in df.columns:
            print("\n(skipping %s: not a column in the CSV)" % metric)
            continue
        va, vb, dropped = paired_deltas(df, metric, args.a, args.b)
        report(metric, va, vb, dropped, args.a, args.b)
    print()


if __name__ == "__main__":
    main()
