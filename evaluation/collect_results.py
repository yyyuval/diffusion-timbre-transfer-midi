"""Every number from every stage, in one place.

The results are spread over six directories in four formats: per-clip content
CSVs, a merged bridge CSV for DPD and JD, two FAD tables, a classifier score
file per generated directory, and the ceilings. Reading them one at a time
invites transcription errors into whatever gets written up, so this collects
them into one tidy CSV and prints the tables that go in a report.

Two things it computes that no individual stage does.

Content scores are divided by the ceiling for the TARGET instruments, because
that is the most a perfect transfer could score -- basic-pitch manages only
0.780 on real bassoon, so 0.636 is 82% of what is reachable, not 64% of
perfect.

For the classifier it reports the mean probability of the SOURCE instruments
beside the target ones. A transfer that kept the source timbre scores well on
content and badly on FAD, and looks from the target probability alone like a
model that simply underperformed. The source probability separates the two.

  python evaluation/collect_results.py
  python evaluation/collect_results.py --pitch-root ... --timbre-root ...
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# experiment -> (case, pretty label, trained sigma, uses midi)
EXPERIMENTS = {
    "no_midi": ("poly", "no MIDI", 5, False),
    "original_midi": ("poly", "MIDI", 5, True),
    "no_midi_s100": ("poly", "no MIDI", 100, False),
    "original_midi_s100": ("poly", "MIDI", 100, True),
    "mono_c2b_no_midi": ("mono", "no MIDI", 5, False),
    "mono_c2b_original_midi": ("mono", "MIDI", 5, True),
    "mono_c2b_no_midi_s100": ("mono", "no MIDI", 100, False),
    "mono_c2b_original_midi_s100": ("mono", "MIDI", 100, True),
}
CASE_INSTRUMENTS = {
    "poly": dict(source=["flute", "bassoon"], target=["violin", "cello"]),
    "mono": dict(source=["cello"], target=["bassoon"]),
}


def read(path):
    path = Path(path)
    if not path.is_file():
        return []
    with io.open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, col):
    vals = []
    for r in rows:
        v = r.get(col, "")
        if v not in (None, "", "nan"):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else None


def fmt(v, spec="%.3f"):
    return "--" if v is None else spec % v


def ceilings(pitch_root):
    """{case: {"source": f1, "target": f1, ...}} from the ceiling directory."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(str(Path(pitch_root) / "ceiling" / "*.csv"))):
        name = Path(path).stem
        case = name.replace("_target", "")
        side = "target" if name.endswith("_target") else "source"
        rows = read(path)
        if not rows:
            continue
        out[case][side] = dict(
            n=len(rows),
            f1=mean(rows, "transcription_f1"),
            precision=mean(rows, "transcription_precision"),
            recall=mean(rows, "transcription_recall"),
            multipitch_f1=mean(rows, "multipitch_f1"),
            ensemble=rows[0].get("ensemble", ""),
        )
    return out


def content(pitch_root):
    """{experiment: {metric: mean}} over the per-clip content CSVs."""
    out = {}
    for path in sorted(glob.glob(str(Path(pitch_root) / "content" / "*.csv"))):
        if path.endswith(".clean.csv"):
            continue
        for exp, rows in group(read(path), "experiment").items():
            out.setdefault(exp, dict(
                n=len(rows),
                f1=mean(rows, "transcription_f1"),
                precision=mean(rows, "transcription_precision"),
                recall=mean(rows, "transcription_recall"),
                multipitch_f1=mean(rows, "multipitch_f1"),
            ))
    return out


def group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r.get(key, "")].append(r)
    return out


def dpd_jd(pitch_root):
    path = Path(pitch_root) / "bridge_metrics_all.clean.csv"
    if not path.is_file():
        path = Path(pitch_root) / "bridge_metrics_all.csv"
    out = {}
    for exp, rows in group(read(path), "experiment").items():
        out[exp] = dict(n=len({r["sample_id"] for r in rows}),
                        dpd=mean(rows, "DPD"), jd=mean(rows, "JD"),
                        source=path.name)
    return out


def fad(timbre_root):
    """{experiment: {"encodec": v, "clap": v, "clap_padded": v}} plus sanity rows."""
    base = Path(timbre_root) / "fad"
    out = defaultdict(dict)
    n_ref = {}
    for fname, key, col in (("encodec_fad_results.csv", "encodec", "fad"),
                            ("clap_fad_200.csv", "clap", "clap_fad"),
                            ("clap_fad_padded.csv", "clap_padded", "clap_fad")):
        for r in read(base / fname):
            exp = r.get("experiment", "")
            try:
                out[exp][key] = float(r[col])
            except (KeyError, ValueError):
                continue
            # the sanity row splits the reference in half, so its n_reference
            # is not the one the real conditions were scored against
            if not exp.startswith("sanity"):
                n_ref.setdefault(key, {})[r.get("case", "")] = r.get("n_reference", "")
    return out, n_ref


def classifier(timbre_root):
    """{experiment: {target_ok, target_exact, p_target, p_source}}."""
    out = {}
    for path in sorted(glob.glob(str(Path(timbre_root) / "fad" / "*" / "generated" / "*" /
                                     "timbre_classifier_scores.csv"))):
        parts = Path(path).parts
        case, exp = parts[-4], parts[-2]
        rows = read(path)
        if not rows:
            continue
        inst = CASE_INSTRUMENTS.get(case)
        if inst is None:
            continue
        def frac(col):
            vals = [r.get(col, "") for r in rows]
            vals = [v for v in vals if v not in ("", None)]
            return sum(str(v).strip().lower() == "true" for v in vals) / len(vals) if vals else None
        def prob(names):
            per = [mean(rows, n) for n in names]
            per = [p for p in per if p is not None]
            return sum(per) / len(per) if per else None
        out[exp] = dict(n=len(rows), case=case,
                        target_ok=frac("target_ok"), target_exact=frac("target_exact"),
                        p_target=prob(inst["target"]), p_source=prob(inst["source"]))
    return out


def table(title, head, rows, note=None):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(head)]
    line = "  ".join("%-*s" % (w, h) for w, h in zip(widths, head))
    print("\n" + title)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join("%-*s" % (w, c) for w, c in zip(widths, r)))
    if note:
        print(note)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pitch-root", default="Outputs/WAV_files/pitch_validation_200")
    p.add_argument("--timbre-root", default="Outputs/WAV_files/clap_validation_1000")
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    ceil = ceilings(args.pitch_root)
    cont = content(args.pitch_root)
    dj = dpd_jd(args.pitch_root)
    fd, n_ref = fad(args.timbre_root)
    clf = classifier(args.timbre_root)

    cj = Path(args.timbre_root) / "classifier" / "timbre_classifier.json"
    clf_meta = json.loads(cj.read_text()) if cj.is_file() else {}

    # ---------------------------------------------------------------- ceilings
    rows = []
    for case in sorted(ceil):
        for side in ("source", "target"):
            c = ceil[case].get(side)
            if c:
                rows.append([case, side, c["ensemble"], c["n"], fmt(c["f1"]),
                             fmt(c["precision"]), fmt(c["recall"]), fmt(c["multipitch_f1"])])
    table("TRANSCRIPTION CEILING  (basic-pitch on real audio vs its ground-truth MIDI)",
          ["case", "side", "instruments", "n", "note F1", "P", "R", "frame F1"], rows,
          "  The target row is the one to divide by: it is the most a perfect transfer could score.")

    # ----------------------------------------------------------------- content
    rows = []
    for exp, (case, label, sigma, _) in EXPERIMENTS.items():
        c = cont.get(exp, {})
        d = dj.get(exp, {})
        top = (ceil.get(case, {}).get("target") or {}).get("f1")
        pct = (c.get("f1") / top * 100) if c.get("f1") and top else None
        rows.append([case, str(sigma), label, str(c.get("n", "--")), fmt(c.get("f1")),
                     fmt(pct, "%.0f%%"), fmt(c.get("precision")), fmt(c.get("recall")),
                     fmt(c.get("multipitch_f1")), fmt(d.get("dpd")), fmt(d.get("jd"))])
    table("CONTENT  (200 shared clips, 50 ms onset tolerance)",
          ["case", "sigma", "cond", "n", "note F1", "of ceil", "P", "R", "frame F1", "DPD", "JD"],
          rows,
          "  DPD and JD are distances: lower is better. Note F1 and frame F1: higher is better.")

    # --------------------------------------------------------------------- FAD
    rows = []
    for exp in list(EXPERIMENTS) + [k for k in fd if k.startswith("sanity")]:
        f = fd.get(exp)
        if not f:
            continue
        case, label, sigma, _ = EXPERIMENTS.get(exp, ("--", "real vs real", "--", None))
        rows.append([case, str(sigma), label, fmt(f.get("encodec"), "%.3f"),
                     fmt(f.get("clap"), "%.4f"), fmt(f.get("clap_padded"), "%.4f")])
    table("FAD  (distance to real target audio -- lower is better)",
          ["case", "sigma", "cond", "EnCodec 128-D", "CLAP 512-D", "CLAP padded"], rows,
          "  n_reference per metric: %s\n"
          "  EnCodec is sound here (n > 128). CLAP is not (n < 512): its covariance is\n"
          "  singular, so read it as a ranking at most." % json.dumps(n_ref))

    # -------------------------------------------------------------- classifier
    if clf:
        rows = []
        for exp, (case, label, sigma, _) in EXPERIMENTS.items():
            c = clf.get(exp)
            if not c:
                continue
            rows.append([case, str(sigma), label, str(c["n"]), fmt(c["target_ok"]),
                         fmt(c["target_exact"]), fmt(c["p_target"]), fmt(c["p_source"])])
        table("INSTRUMENT CLASSIFIER  (multi-label, on generated audio)",
              ["case", "sigma", "cond", "n", "target_ok", "exact", "p(target)", "p(source)"], rows,
              "  target_ok ignores extra instruments; exact does not.\n"
              "  A HIGH p(source) is the failure the paper's Table I found in GFB: melody\n"
              "  preserved, timbre untouched. Read the two probability columns together.")
        if clf_meta:
            print("  classifier on held-out REAL stems: macro-F1 %.3f, subset accuracy %.3f, %d instruments"
                  % (clf_meta.get("val_macro_f1", float("nan")),
                     clf_meta.get("val_subset_acc", float("nan")),
                     len(clf_meta.get("instruments", []))))

    # ------------------------------------------------------------- false notes
    fn_rows = read(Path(args.pitch_root) / "false_note_diagnosis.csv")
    if fn_rows:
        rows = [[r["experiment"], r["clips"], fmt(float(r["precision"])),
                 fmt(float(r["fp_octave"])), fmt(float(r["fp_reattack"])),
                 fmt(float(r["fp_late"])), fmt(float(r["fp_spurious"]))]
                for r in fn_rows]
        table("FALSE NOTES  (share of all estimated notes)",
              ["condition", "clips", "P", "octave", "reattack", "late", "spurious"], rows,
              "  Compare each generated row with its ceiling_ row, not with zero.")

    # ------------------------------------------------------------------ tidy
    out = Path(args.out_csv or Path(args.timbre_root) / "results_all.csv")
    fields = ["case", "sigma", "condition", "experiment", "n_clips", "note_f1",
              "pct_of_ceiling", "precision", "recall", "frame_f1", "dpd", "jd",
              "encodec_fad", "clap_fad", "clap_fad_padded", "target_ok", "target_exact",
              "p_target", "p_source", "ceiling_target_f1"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for exp, (case, label, sigma, _) in EXPERIMENTS.items():
            c, d, fv, cl = cont.get(exp, {}), dj.get(exp, {}), fd.get(exp, {}), clf.get(exp, {})
            top = (ceil.get(case, {}).get("target") or {}).get("f1")
            w.writerow(dict(
                case=case, sigma=sigma, condition=label, experiment=exp,
                n_clips=c.get("n"), note_f1=c.get("f1"),
                pct_of_ceiling=(c["f1"] / top) if c.get("f1") and top else None,
                precision=c.get("precision"), recall=c.get("recall"),
                frame_f1=c.get("multipitch_f1"), dpd=d.get("dpd"), jd=d.get("jd"),
                encodec_fad=fv.get("encodec"), clap_fad=fv.get("clap"),
                clap_fad_padded=fv.get("clap_padded"),
                target_ok=cl.get("target_ok"), target_exact=cl.get("target_exact"),
                p_target=cl.get("p_target"), p_source=cl.get("p_source"),
                ceiling_target_f1=top))
    print("\n-> %s" % out)
    missing = [k for k, v in (("content", cont), ("DPD/JD", dj), ("FAD", fd),
                              ("classifier", clf), ("ceilings", ceil)) if not v]
    if missing:
        print("   nothing found for: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
