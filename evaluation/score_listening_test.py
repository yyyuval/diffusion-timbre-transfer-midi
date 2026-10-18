"""Score the blind listening test built by make_listening_test.py.

For every comparison: in how many trials the listener chose the model EnCodec-FAD
prefers, and in how many the one CLAP-FAD prefers (in a disagreeing comparison
these are complementary), with an exact two-sided binomial test against 50%.
The control comparison -- both metrics agree by a wide margin -- should be
heard clearly; if it is not, the other results of that listener say little.

  venv_yuval/bin/python evaluation/score_listening_test.py Outputs/listening_test/answers/*.csv
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from math import comb
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("answers", nargs="+", help="answers_<name>.csv files downloaded from the test page")
    p.add_argument("--key", default="Outputs/listening_test/key.csv")
    return p.parse_args()


def binom_two_sided(k, n):
    """Exact two-sided p for k of n under p = 0.5."""
    if n == 0:
        return float("nan")
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def read(path):
    with io.open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    key = {r["trial"]: r for r in read(args.key)}
    build = {r["build_id"] for r in key.values()}
    comparisons = sorted({r["comparison"] for r in key.values()},
                         key=lambda c: (any(r["control"] == "1" for r in key.values() if r["comparison"] == c), c))

    by_listener = defaultdict(dict)
    for path in args.answers:
        for r in read(path):
            if r["build_id"] not in build:
                raise SystemExit("%s was answered on another build (%s), not on this key (%s)"
                                 % (path, r["build_id"], ", ".join(build)))
            if r["trial"] not in key:
                raise SystemExit("%s: unknown trial %s" % (path, r["trial"]))
            by_listener[r["listener"]][r["trial"]] = r

    groups = list(by_listener.items())
    if len(groups) > 1:
        groups.append(("ALL LISTENERS", {"%s|%s" % (who, t): a for who, ans in by_listener.items()
                                         for t, a in ans.items()}))
    for who, answers in groups:
        print("\n" + "=" * 96)
        total = sum(1 for r in key.values()) * (1 if who != "ALL LISTENERS" else len(by_listener))
        print("  %s -- %d of %d trials answered" % (who, len(answers), total))
        print("=" * 96)
        print("%-42s %4s  %-16s %-16s %7s   %s" % ("comparison", "n", "chose EnCodec's", "chose CLAP's",
                                                   "p", "sure only (EnCodec's / n)"))
        for comp in comparisons:
            n = k_enc = k_clap = n_sure = k_sure = 0
            for t, a in answers.items():
                k = key[t.split("|")[-1]]
                if k["comparison"] != comp:
                    continue
                n += 1
                k_enc += a["choice"] == k["encodec_choice"]
                k_clap += a["choice"] == k["clap_choice"]
                if a["confidence"] == "sure":
                    n_sure += 1
                    k_sure += a["choice"] == k["encodec_choice"]
            control = next(r["control"] for r in key.values() if r["comparison"] == comp) == "1"
            if not n:
                print("%-42s    0" % comp)
                continue
            print("%-42s %4d  %-16s %-16s %7.3f   %d / %d%s"
                  % (comp, n, "%d (%.0f%%)" % (k_enc, 100 * k_enc / n), "%d (%.0f%%)" % (k_clap, 100 * k_clap / n),
                     binom_two_sided(k_enc, n), k_sure, n_sure,
                     "   <- CONTROL: both metrics pick the same model" if control else ""))
        print("\n  p < 0.05 means the choices lean to one side beyond chance. In a disagreeing\n"
              "  comparison 'chose EnCodec's' + 'chose CLAP's' = n. For the control, a listener\n"
              "  who does not pick the metrics' model in most trials did not hear the difference,\n"
              "  and their other rows should be read with that in mind.")


if __name__ == "__main__":
    main()
