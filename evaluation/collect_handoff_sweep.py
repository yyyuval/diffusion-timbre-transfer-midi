"""Every sigma_max=100 model at every handoff, in one table.

Reads the three result folders run_handoff_sweep.sh wrote into and prints one
CSV row per (test set, model, handoff) -- short enough to paste -- and saves it
to Outputs/handoff_sweep_results.csv. handoff 100 is the experiment without a
suffix; 5, 20 and 50 are <pair>_h5, _h20, _h50.

Content means are taken over the clips every model of that set and case was
scored on, so every row of a block is paired. DPD and JD are left empty for
polyphonic rows: they keep one note per moment, which says nothing with two
instruments.

    python evaluation/collect_handoff_sweep.py
"""

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.report_merged_midi import mean, num, read, truthy  # noqa: E402

HANDOFFS = [5, 20, 50, 100]
A_CONTENT = Path("Outputs/WAV_files/pitch_validation_200")
A_TIMBRE = Path("Outputs/WAV_files/clap_validation_1000")
B_ROOT = Path("Outputs/WAV_files/merged_midi_test")

# (set, case, model label, pair, content root, timbre root)
MODELS = [
    ("A", "poly", "No MIDI", "no_midi_s100", A_CONTENT, A_TIMBRE),
    ("A", "poly", "Stacked MIDI, real", "original_midi_s100", A_CONTENT, A_TIMBRE),
    ("B", "poly", "Stacked MIDI, real", "original_midi_s100", B_ROOT, B_ROOT),
    ("B", "poly", "Merged MIDI, real", "merged_real_midi_s100", B_ROOT, B_ROOT),
    ("B", "poly", "Merged MIDI, generated", "merged_generated_midi_s100", B_ROOT, B_ROOT),
    ("A", "mono", "No MIDI", "mono_c2b_no_midi_s100", A_CONTENT, A_TIMBRE),
    ("A", "mono", "MIDI, real", "mono_c2b_original_midi_s100", A_CONTENT, A_TIMBRE),
]
TARGET = {"poly": ["violin", "cello"], "mono": ["bassoon"]}
SOURCE = {"poly": ["flute", "bassoon"], "mono": ["cello"]}
CEILING_FILES = {"poly": A_CONTENT / "ceiling" / "poly_target.csv",
                 "mono": A_CONTENT / "ceiling" / "mono_target.csv"}


def exp_name(pair, h):
    return pair if h == 100 else "%s_h%d" % (pair, h)


def content_rows(root):
    """{experiment: {sample_id: row}} from every content CSV in a folder."""
    out = {}
    for path in sorted((root / "content").glob("*.csv")):
        if path.name.endswith(".clean.csv"):
            continue
        for r in read(path):
            out.setdefault(r["experiment"], {})[r["sample_id"]] = r
    return out


def bridge_rows(root):
    out = {}
    for r in read(root / "bridge_metrics_all.csv"):
        out.setdefault(r["experiment"], {})[r["sample_id"]] = r
    return out


def main():
    ceilings = {c: mean(num(r.get("transcription_f1")) for r in read(p)) for c, p in CEILING_FILES.items()}
    content = {root: content_rows(root) for root in (A_CONTENT, B_ROOT)}
    bridges = {root: bridge_rows(root) for root in (A_CONTENT, B_ROOT)}
    fad, clap = {}, {}
    for root in (A_TIMBRE, B_ROOT):
        for r in read(root / "fad" / "encodec_fad_results.csv"):
            fad[(root, r.get("case"), r.get("experiment"))] = num(r.get("fad"))
        for r in read(root / "fad" / "clap_fad_200.csv"):
            clap[(root, r.get("case"), r.get("experiment"))] = num(r.get("clap_fad"))

    # the paired clip set of each (set, case): clips scored under every row
    paired = {}
    for s, case, _, pair, croot, _ in MODELS:
        for h in HANDOFFS:
            ids = set(content[croot].get(exp_name(pair, h), {}))
            key = (s, case)
            paired[key] = ids if key not in paired else paired[key] & ids

    fields = ["set", "case", "model", "handoff", "n_content", "note_f1", "pct_ceiling", "precision",
              "recall", "frame_f1", "dpd", "jd", "encodec_fad", "clap_fad", "target_found", "p_source"]
    out_rows, missing = [], []
    for s, case, label, pair, croot, troot in MODELS:
        for h in HANDOFFS:
            e = exp_name(pair, h)
            ids = paired[(s, case)]
            c = [content[croot].get(e, {}).get(i) for i in ids]
            c = [x for x in c if x]
            b = [bridges[croot].get(e, {}).get(i) for i in ids] if case == "mono" else []
            b = [x for x in b if x]
            scores = read(troot / "fad" / case / "generated" / e / "timbre_classifier_scores.csv")
            f1 = mean(num(x.get("transcription_f1")) for x in c)
            row = dict(
                set=s, case=case, model=label, handoff=h, n_content=len(c),
                note_f1=f1, pct_ceiling=(100 * f1 / ceilings[case]) if f1 and ceilings[case] else None,
                precision=mean(num(x.get("transcription_precision")) for x in c),
                recall=mean(num(x.get("transcription_recall")) for x in c),
                frame_f1=mean(num(x.get("multipitch_f1")) for x in c),
                dpd=mean(num(x.get("DPD")) for x in b) if b else None,
                jd=mean(num(x.get("JD")) for x in b) if b else None,
                encodec_fad=fad.get((troot, case, e)), clap_fad=clap.get((troot, case, e)),
                target_found=mean(float(truthy(x.get("target_ok"))) for x in scores) if scores else None,
                p_source=mean(mean(num(x.get(i)) for i in SOURCE[case]) for x in scores) if scores else None,
            )
            if not c or row["encodec_fad"] is None or not scores:
                missing.append("%s/%s" % (s, e))
            out_rows.append(row)

    def cell(v, k):
        if v is None:
            return ""
        if k == "pct_ceiling":
            return "%.0f" % v
        if isinstance(v, float):
            return "%.3f" % v
        return str(v)

    path = Path("Outputs/handoff_sweep_results.csv")
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: cell(r[k], k) for k in fields})

    print("ceilings: poly %.3f, mono %.3f" % (ceilings["poly"], ceilings["mono"]))
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(fields)
    for r in out_rows:
        w.writerow([cell(r[k], k) for k in fields])
    if missing:
        print("\nMISSING data for: %s" % ", ".join(missing))
    print("\n-> %s" % path)


if __name__ == "__main__":
    main()
