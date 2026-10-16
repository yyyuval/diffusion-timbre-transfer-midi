"""The merged-MIDI test, written up: REPORT.md, a tidy CSV, and a listening folder.

Reads what evaluation/run_merged_test.sh produced under --out-root and writes:

  REPORT.md        every table, readable as it stands -- training, test data,
                   content, paired differences, timbre, and the listening clip
  results.csv      one row per condition, every number in REPORT.md
  listen/<clip>/   ONE test clip through every model, copied (not linked) so
                   the folder can be taken off the server as it is. Each wav
                   has a .txt beside it with that clip's own scores -- not the
                   averages -- and README.md puts them side by side.

The listening clip is chosen, not picked: the one whose merged-real-MIDI note
F1 sits closest to that condition's median, so it is a typical clip rather
than the best one. --clip overrides it.

    python evaluation/report_merged_midi.py --out-root Outputs/WAV_files/merged_midi_test
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# experiment -> (label, layout, MIDI source, sigma_max, run of the source model)
CONDITIONS = {
    "original_midi":              ("stacked, real, sigma 5",     "stacked (256 rows)", "real", 5,
                                   "flute_bassoon_gt_ms_85ep"),
    "original_midi_s100":         ("stacked, real, sigma 100",   "stacked (256 rows)", "real", 100,
                                   "flute_bassoon_gt_ms_85ep_sigma100"),
    "merged_real_midi":           ("merged, real, sigma 5",      "merged (128 rows)",  "real", 5,
                                   "flute_bassoon_gtmerged_ms_85ep"),
    "merged_real_midi_s100":      ("merged, real, sigma 100",    "merged (128 rows)",  "real", 100,
                                   "flute_bassoon_gtmerged_ms_85ep_sigma100"),
    "merged_generated_midi":      ("merged, generated, sigma 5", "merged (128 rows)",  "basic-pitch", 5,
                                   "flute_bassoon_bpmerged_ms_85ep"),
    "merged_generated_midi_s100": ("merged, generated, sigma 100", "merged (128 rows)", "basic-pitch", 100,
                                   "flute_bassoon_bpmerged_ms_85ep_sigma100"),
}
NEW = [e for e in CONDITIONS if e.startswith("merged_")]
STACKED = ["original_midi", "original_midi_s100"]

# The questions the test is for, each a paired comparison on identical clips.
COMPARISONS = [
    ("merged_real_midi", "original_midi", "merged vs stacked, real MIDI, sigma 5"),
    ("merged_real_midi_s100", "original_midi_s100", "merged vs stacked, real MIDI, sigma 100"),
    ("merged_generated_midi", "merged_real_midi", "generated vs real, merged, sigma 5"),
    ("merged_generated_midi_s100", "merged_real_midi_s100", "generated vs real, merged, sigma 100"),
]

TARGET, SOURCE = ["violin", "cello"], ["flute", "bassoon"]
# File-name order in the listening folder: the four new models, then references.
LISTEN_ORDER = NEW + STACKED
FILE_NAMES = {
    "merged_real_midi": "merged_real_midi_sigma5",
    "merged_real_midi_s100": "merged_real_midi_sigma100",
    "merged_generated_midi": "merged_generated_midi_sigma5",
    "merged_generated_midi_s100": "merged_generated_midi_sigma100",
    "original_midi": "stacked_real_midi_sigma5",
    "original_midi_s100": "stacked_real_midi_sigma100",
}


# --------------------------------------------------------------------- reading
def read(path):
    path = Path(path)
    if not path.is_file():
        return []
    with io.open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def fmt(v, spec="%.3f"):
    return "–" if v is None else spec % v


def truthy(v):
    return str(v).strip().lower() == "true"


def load(out_root: Path):
    """{experiment: {sample_id: {metric: value}}}, merged across every source."""
    per = {e: {} for e in CONDITIONS}

    for r in read(out_root / "bridge_metrics_all.csv"):
        e = r.get("experiment")
        if e in per:
            d = per[e].setdefault(r["sample_id"], {})
            d.update(dpd=num(r.get("DPD")), jd=num(r.get("JD")),
                     output_audio=r.get("output_audio"), input_audio=r.get("input_audio"),
                     source_ckpt=r.get("source_ckpt"), target_ckpt=r.get("target_ckpt"))

    for path in glob.glob(str(out_root / "content" / "*.csv")):
        for r in read(path):
            e = r.get("experiment")
            if e in per:
                per[e].setdefault(r["sample_id"], {}).update(
                    note_f1=num(r.get("transcription_f1")),
                    precision=num(r.get("transcription_precision")),
                    recall=num(r.get("transcription_recall")),
                    frame_f1=num(r.get("multipitch_f1")),
                    n_ref_notes=num(r.get("n_ref_notes")),
                    n_est_notes=num(r.get("n_est_notes")))

    for e in CONDITIONS:
        scores = out_root / "fad" / "poly" / "generated" / e / "timbre_classifier_scores.csv"
        for r in read(scores):
            sid = Path(r.get("file", "")).stem
            if not sid:
                continue
            probs = {k: num(v) for k, v in r.items()
                     if k not in ("file", "pred", "target_ok", "target_exact")}
            per[e].setdefault(sid, {}).update(
                pred=r.get("pred", ""), target_ok=truthy(r.get("target_ok")),
                target_exact=truthy(r.get("target_exact")), probs=probs)
    return per


def fad_tables(out_root: Path):
    enc = {r["experiment"]: r for r in read(out_root / "fad" / "encodec_fad_results.csv")
           if r.get("case") == "poly"}
    clap = {r["experiment"]: r for r in read(out_root / "fad" / "clap_fad_200.csv")
            if r.get("case") == "poly"}
    return enc, clap


def content_ids(per):
    """Clips scored for content under EVERY condition -- the paired set."""
    sets = [{s for s, d in per[e].items() if d.get("note_f1") is not None} for e in CONDITIONS]
    sets = [s for s in sets if s]
    return sorted(set.intersection(*sets)) if sets else []


def ceiling(path):
    rows = read(path)
    return mean(num(r.get("transcription_f1")) for r in rows), len(rows)


def best_checkpoint(ckpt_root: Path, run: str):
    hits = glob.glob(str(ckpt_root / run / "ckpts" / "epoch=*.ckpt"))
    best = None
    for h in hits:
        m = re.search(r"epoch=(\d+)-valid_loss=([0-9.]+?)\.?ckpt$", os.path.basename(h))
        if m:
            cand = (float(m.group(2)), int(m.group(1)))
            best = cand if best is None or cand < best else best
    return best  # (valid_loss, epoch) or None


# ------------------------------------------------------------------ statistics
def paired(per, ids, a, b, key="note_f1", n_boot=5000, seed=2026):
    va = np.array([per[a][s][key] for s in ids], dtype=float)
    vb = np.array([per[b][s][key] for s in ids], dtype=float)
    d = va - vb
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(d), mean_a=va.mean(), mean_b=vb.mean(), delta=d.mean(), lo=lo, hi=hi,
                a_better=int((d > 0).sum()), ties=int((d == 0).sum()))


# --------------------------------------------------------------------- writing
def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def condition_summary(per, ids, enc, clap, ceil):
    rows = {}
    for e in CONDITIONS:
        c = {s: per[e][s] for s in ids if s in per[e]}
        clf = [d for d in per[e].values() if "target_ok" in d]
        f1 = mean(d.get("note_f1") for d in c.values())
        rows[e] = dict(
            n_content=len(c), note_f1=f1,
            pct_ceiling=(f1 / ceil) if f1 is not None and ceil else None,
            precision=mean(d.get("precision") for d in c.values()),
            recall=mean(d.get("recall") for d in c.values()),
            frame_f1=mean(d.get("frame_f1") for d in c.values()),
            dpd=mean(d.get("dpd") for d in c.values()),
            jd=mean(d.get("jd") for d in c.values()),
            encodec_fad=num(enc.get(e, {}).get("fad")),
            clap_fad=num(clap.get(e, {}).get("clap_fad")),
            n_generated=num(enc.get(e, {}).get("n_generated")),
            n_classified=len(clf),
            target_ok=mean(float(d["target_ok"]) for d in clf),
            target_exact=mean(float(d["target_exact"]) for d in clf),
            p_target=mean(mean(d["probs"].get(i) for i in TARGET) for d in clf),
            p_source=mean(mean(d["probs"].get(i) for i in SOURCE) for d in clf),
        )
    return rows


def write_report(args, per, ids, summary, enc, clap, ceil, n_ceil, stats, listen_info):
    L = []
    L.append("# Merged-MIDI test — results\n")
    L.append("Generated %s from `%s` by `evaluation/report_merged_midi.py`.\n"
             % (date.today().isoformat(), args.out_root))

    L.append("## What was tested\n")
    L.append("Polyphonic transfer, flute + bassoon → violin + cello. The four new models get **one** "
             "MIDI roll for both instruments; the two stacked models, trained earlier, get one roll "
             "per instrument and are here as the comparison.\n")
    L.append(md_table(["condition", "MIDI layout", "MIDI notes", "sigma_max"],
                      [[f"`{e}`", CONDITIONS[e][1], CONDITIONS[e][2], CONDITIONS[e][3]]
                       for e in CONDITIONS]) + "\n")

    L.append("## Training\n")
    L.append("Best checkpoint of each network. `valid_loss` is only comparable within one sigma_max, "
             "and each architecture validated on its own random 20%, so differences under ~0.01 are "
             "inside sampling noise.\n")
    rows = []
    ck = Path(args.checkpoints)
    for e in CONDITIONS:
        src = CONDITIONS[e][4]
        tgt = src.replace("flute_bassoon", "violin_cello")
        for side, run in (("source", src), ("target", tgt)):
            b = best_checkpoint(ck, run)
            rows.append([f"`{run}`", side, CONDITIONS[e][3],
                         "–" if b is None else b[1], "–" if b is None else "%.3f" % b[0]])
    L.append(md_table(["run", "side", "sigma_max", "best epoch", "valid_loss"], rows) + "\n")

    L.append("## Test data\n")
    L.append(f"- **Content metrics: {len(ids)} clips**, the same clips in every condition, each held "
             "out by the stacked AND the merged models (both source and target). Paired, so the "
             "differences below are clip by clip.\n"
             "- **FAD and the classifier: up to 1,000 clips per condition**, each drawn from clips "
             "its own models never trained on. FAD compares distributions, so these need not match "
             "across conditions.\n"
             "- **FAD reference:** real violin + cello that both target architectures held out.\n"
             "- The merged models landed on a different train/validation split from the stacked "
             "ones (their parameter count differs), so earlier test lists could not be reused: "
             "about 80% of those clips were in the merged models' training set.\n")

    L.append("## Content preservation\n")
    L.append(f"Higher is better, except DPD and JD. Transcription ceiling (basic-pitch on real violin "
             f"+ cello, {n_ceil} clips): note F1 **{fmt(ceil)}** — the most a perfect transfer "
             "could score.\n")
    L.append(md_table(
        ["condition", "n", "note F1", "% of ceiling", "precision", "recall", "frame F1", "DPD", "JD"],
        [[f"`{e}`", s["n_content"], fmt(s["note_f1"]), fmt(s["pct_ceiling"] and 100 * s["pct_ceiling"], "%.0f%%"),
          fmt(s["precision"]), fmt(s["recall"]), fmt(s["frame_f1"]), fmt(s["dpd"]), fmt(s["jd"])]
         for e, s in summary.items()]) + "\n")
    L.append("JD saturates near zero on 17-second clips and is reported for comparison with the "
             "paper only.\n")

    L.append("## Paired differences (note F1)\n")
    L.append("Mean of A − B over the same clips, with a 95% bootstrap interval (5,000 resamples). "
             "An interval that excludes zero is a real difference.\n")
    rows = []
    for (a, b, label), s in zip(COMPARISONS, stats):
        if s is None:
            rows.append([label, "–", "–", "–", "–", "–"])
            continue
        verdict = ("`%s` better" % a if s["lo"] > 0 else "`%s` better" % b if s["hi"] < 0
                   else "no clear difference")
        rows.append([label, s["n"], "%+.3f" % s["delta"], "[%+.3f, %+.3f]" % (s["lo"], s["hi"]),
                     "%d/%d" % (s["a_better"], s["n"]), verdict])
    L.append(md_table(["A vs B", "n", "mean A − B", "95% CI", "A better on", "verdict"], rows) + "\n")

    L.append("## Timbre\n")
    L.append("EnCodec-FAD: distance to real violin + cello, lower is better — the paper's metric, "
             "and sound at these sample sizes (128-D). CLAP-FAD is 512-D against a reference of a "
             "few hundred clips, so its covariance is singular: read it as a ranking at most.\n")
    sane_enc = [r for k, r in enc.items() if k.startswith("sanity")]
    sane_clap = [r for k, r in clap.items() if k.startswith("sanity")]
    L.append(md_table(
        ["condition", "n generated", "EnCodec-FAD", "CLAP-FAD", "target found", "exact set",
         "p(violin, cello)", "p(flute, bassoon)"],
        [[f"`{e}`", fmt(s["n_generated"], "%.0f"), fmt(s["encodec_fad"]), fmt(s["clap_fad"], "%.4f"),
          fmt(s["target_ok"]), fmt(s["target_exact"]), fmt(s["p_target"]), fmt(s["p_source"])]
         for e, s in summary.items()]
        + [["*real vs real*", "", fmt(num(sane_enc[0]["fad"])) if sane_enc else "–",
            fmt(num(sane_clap[0]["clap_fad"]), "%.4f") if sane_clap else "–", "", "", "", ""]]) + "\n")
    L.append("The classifier was trained on real stems; a high p(flute, bassoon) would mean the "
             "source timbre survived the transfer.\n")

    if listen_info:
        clip, why, folder = listen_info
        L.append("## Listening example\n")
        L.append(f"One test clip, **`{clip}`**, through every model: `{folder}`. {why} Each wav has "
                 "a `.txt` beside it with that clip's own scores, and `README.md` there puts them "
                 "side by side.\n")

    path = Path(args.out_root) / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def write_csv(out_root: Path, summary):
    path = out_root / "results.csv"
    fields = ["experiment", "layout", "midi_notes", "sigma_max"] + list(next(iter(summary.values())))
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e, s in summary.items():
            w.writerow(dict(experiment=e, layout=CONDITIONS[e][1], midi_notes=CONDITIONS[e][2],
                            sigma_max=CONDITIONS[e][3], **s))
    return path


# ------------------------------------------------------------------ listening
def choose_clip(per, ids, override=None):
    if override:
        if override not in ids:
            raise SystemExit("--clip %s is not one of the %d paired test clips" % (override, len(ids)))
        return override, "Chosen by hand with --clip."
    anchor = "merged_real_midi"
    f1 = np.array([per[anchor][s]["note_f1"] for s in ids])
    med = float(np.median(f1))
    clip = ids[int(np.argmin(np.abs(f1 - med)))]
    return clip, ("Chosen as typical, not as best: its `%s` note F1 (%.3f) is the closest to that "
                  "condition's median over the %d test clips (%.3f)." % (anchor, per[anchor][clip]["note_f1"],
                                                                          len(ids), med))


def clip_lines(e, d, summary):
    s = summary[e]
    lines = [
        "Model: %s  --  %s" % (e, CONDITIONS[e][0]),
        "  MIDI: %s, %s notes; trained at sigma_max %s" % (CONDITIONS[e][1], CONDITIONS[e][2], CONDITIONS[e][3]),
        "  source checkpoint: %s" % d.get("source_ckpt", "?"),
        "  target checkpoint: %s" % d.get("target_ckpt", "?"),
        "",
        "Content -- THIS clip (condition mean over the test clips in brackets)",
        "  note F1      %s   (%s)   higher is better" % (fmt(d.get("note_f1")), fmt(s["note_f1"])),
        "  precision    %s   (%s)" % (fmt(d.get("precision")), fmt(s["precision"])),
        "  recall       %s   (%s)" % (fmt(d.get("recall")), fmt(s["recall"])),
        "  frame F1     %s   (%s)" % (fmt(d.get("frame_f1")), fmt(s["frame_f1"])),
        "  notes        %s transcribed against %s in the score" % (fmt(d.get("n_est_notes"), "%.0f"),
                                                                   fmt(d.get("n_ref_notes"), "%.0f")),
        "  DPD          %s   (%s)   lower is better" % (fmt(d.get("dpd")), fmt(s["dpd"])),
        "  JD           %s   (%s)   lower is better; saturated, do not interpret" % (fmt(d.get("jd")), fmt(s["jd"])),
        "",
        "Timbre -- THIS clip",
    ]
    if "target_ok" in d:
        p = d["probs"]
        lines += [
            "  classifier heard   %s" % (d.get("pred") or "(nothing above threshold)"),
            "  violin and cello   %s   exact set: %s" % ("found" if d["target_ok"] else "NOT found",
                                                         "yes" if d["target_exact"] else "no"),
            "  p(violin) %s  p(cello) %s   |   p(flute) %s  p(bassoon) %s"
            % tuple(fmt(p.get(i)) for i in TARGET + SOURCE),
        ]
    else:
        lines.append("  no classifier score for this clip")
    lines += ["", "FAD compares two SETS of clips and has no per-clip value. This model's FAD: "
              "EnCodec %s, CLAP %s (see REPORT.md)." % (fmt(s["encodec_fad"]), fmt(s["clap_fad"], "%.4f"))]
    return lines


def build_listening(out_root: Path, per, ids, summary, override=None):
    clip, why = choose_clip(per, ids, override)
    folder = out_root / "listen" / clip
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "reference_stacked_midi").mkdir(parents=True)

    any_row = per[NEW[0]][clip]
    src_in = Path(any_row["input_audio"])
    shutil.copyfile(src_in, folder / "00_input_flute_bassoon.wav")

    table = []
    for i, e in enumerate(LISTEN_ORDER, start=1):
        d = per[e].get(clip)
        if not d or not d.get("output_audio") or not Path(d["output_audio"]).is_file():
            print("  listening: no %s output for %s -- left out" % (e, clip))
            continue
        name = "%02d_%s" % (i, FILE_NAMES[e])
        sub = folder if e in NEW else folder / "reference_stacked_midi"
        shutil.copyfile(Path(d["output_audio"]), sub / (name + ".wav"))
        header = ["Clip: %s   (flute + bassoon -> violin + cello, first 17.07 s)" % clip, ""]
        (sub / (name + ".txt")).write_text("\n".join(header + clip_lines(e, d, summary)) + "\n",
                                           encoding="utf-8")
        table.append([("" if e in NEW else "reference_stacked_midi/") + name + ".wav",
                      CONDITIONS[e][0], fmt(d.get("note_f1")), fmt(d.get("precision")),
                      fmt(d.get("recall")), fmt(d.get("dpd")),
                      d.get("pred", "–") if "target_ok" in d else "–",
                      fmt(mean(d["probs"].get(x) for x in TARGET)) if "probs" in d else "–"])

    readme = [
        "# Listening example: `%s`\n" % clip,
        "Flute + bassoon in (`00_input_flute_bassoon.wav`), violin + cello out. The same 17 s through "
        "the four merged-MIDI models, plus the two stacked-MIDI models for reference. %s\n" % why,
        "These are the scores of **this clip**, not averages. Each wav's `.txt` has the full set.\n",
        md_table(["file", "model", "note F1", "precision", "recall", "DPD", "classifier heard",
                  "p(violin, cello)"], table),
        "",
        "Higher is better except DPD. FAD has no per-clip value; see `../../REPORT.md`.",
    ]
    (folder / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return clip, why, folder


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="Outputs/WAV_files/merged_midi_test")
    p.add_argument("--ceiling-csv", default="Outputs/WAV_files/pitch_validation_200/ceiling/poly_target.csv",
                   help="basic-pitch on real violin + cello; independent of every model")
    p.add_argument("--checkpoints", default="our_checkpoints")
    p.add_argument("--clip", default=None, help="the listening clip; default: a typical one")
    p.add_argument("--no-listen", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out_root)
    per = load(out_root)
    ids = content_ids(per)
    if not ids:
        raise SystemExit("no clip has content scores under every condition in %s -- has the "
                         "content stage run?" % out_root)
    enc, clap = fad_tables(out_root)
    ceil, n_ceil = ceiling(args.ceiling_csv)
    summary = condition_summary(per, ids, enc, clap, ceil)
    stats = []
    for a, b, _ in COMPARISONS:
        ok = all(per[a].get(s, {}).get("note_f1") is not None and
                 per[b].get(s, {}).get("note_f1") is not None for s in ids)
        stats.append(paired(per, ids, a, b) if ok else None)

    listen_info = None
    if not args.no_listen:
        listen_info = build_listening(out_root, per, ids, summary, args.clip)

    report = write_report(args, per, ids, summary, enc, clap, ceil, n_ceil, stats, listen_info)
    table = write_csv(out_root, summary)

    print("%d paired test clips, ceiling %s" % (len(ids), fmt(ceil)))
    for e, s in summary.items():
        print("  %-28s note F1 %s (%s of ceiling)  EnCodec-FAD %s  target found %s"
              % (e, fmt(s["note_f1"]), fmt(s["pct_ceiling"] and 100 * s["pct_ceiling"], "%.0f%%"),
                 fmt(s["encodec_fad"]), fmt(s["target_ok"])))
    print("\n-> %s\n-> %s" % (report, table))
    if listen_info:
        print("-> %s" % listen_info[2])


if __name__ == "__main__":
    main()
