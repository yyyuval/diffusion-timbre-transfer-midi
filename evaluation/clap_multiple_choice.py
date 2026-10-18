"""A multiple-choice test for CLAP: which instruments does it hear in each clip?

Every clip's cached CLAP audio embedding is compared with 91 answers -- each
of the 13 CocoChorales instruments alone, and all 78 pairs -- and the answers
are ranked by similarity. Only the 507 answer phrasings go through CLAP's
text encoder (once); the audio was embedded when CLAP-FAD was computed.

Each answer is phrased several ways (pairs in both orders), the phrasings'
unit embeddings are averaged and re-normalised (standard prompt ensembling),
so no single wording -- and no answer's number of phrasings -- decides.

**Decided before the numbers are seen.** The test is usable only if, on real
recordings with a known answer, (a) the right answer is in the top 5 for at
least half the clips, and (b) real SOURCE recordings get the target answer as
first choice in under 10% of clips. Otherwise its rows on the models say
nothing, and the script says so. Chance: 1.1% first choice, 5.5% top 5.

This is CLAP's opinion of each clip, not the truth about it; the listening
test decides that.

  venv_yuval/bin/python evaluation/clap_multiple_choice.py
  -> Outputs/clap_multiple_choice/{summary.csv,per_clip.csv}
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.clap_text_probe import load_model, read_fads, unit  # noqa: E402

NAMES = ["violin", "viola", "cello", "double bass", "flute", "oboe", "clarinet", "bassoon",
         "trumpet", "French horn", "trombone", "tuba", "saxophone"]
SINGLE = ["the sound of a solo {a}", "a solo {a} playing", "{a}"]
PAIR = ["the sound of {a} and {b} playing together", "a duet for {a} and {b}", "{a} and {b}"]
ANSWERS = {"poly": {"target": ("violin", "cello"), "source": ("flute", "bassoon")},
           "mono": {"target": ("bassoon",), "source": ("cello",)}}
DEFAULT_ROOTS = ["Outputs/WAV_files/clap_validation_1000", "Outputs/WAV_files/merged_midi_test"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    p.add_argument("--real-source-dir", default="Outputs/clap_text_probe",
                   help="where clap_text_probe.py saved the real source embeddings")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="Outputs/clap_multiple_choice")
    return p.parse_args()


def options():
    """The 91 answers as tuples of instrument names, and each one's phrasings."""
    opts = [(a,) for a in NAMES] + list(itertools.combinations(NAMES, 2))
    phrasings = []
    for o in opts:
        if len(o) == 1:
            phrasings.append([t.format(a=o[0]) for t in SINGLE])
        else:
            phrasings.append([t.format(a=x, b=y) for t in PAIR for x, y in (o, o[::-1])])
    return opts, phrasings


def answer_matrix(proc, model, device, phrasings, batch=64):
    """[n_answers, 512]: mean of each answer's unit phrasing embeddings, re-normalised."""
    import torch
    flat = [p for ph in phrasings for p in ph]
    embs = []
    for i in range(0, len(flat), batch):
        inputs = {k: v.to(device) for k, v in proc(text=flat[i:i + batch], return_tensors="pt",
                                                    padding=True).items()}
        with torch.no_grad():
            embs.append(unit(model.get_text_features(**inputs).cpu().numpy()))
    embs = np.concatenate(embs)
    out, j = [], 0
    for ph in phrasings:
        out.append(embs[j:j + len(ph)].mean(0))
        j += len(ph)
    return unit(np.array(out))


def label(o):
    return " + ".join(o)


def rank(E, M, k=5):
    """Top-k answer indices per clip and their cosine scores."""
    S = unit(E) @ M.T
    top = np.argsort(-S, axis=1)[:, :k]
    return top, np.take_along_axis(S, top, axis=1)


def judge(top, opts, case, expect):
    """Shares over clips: first choice == expected answer, expected in top 5, first choice
    containing each target / source instrument; and the most common first choices."""
    want = set(ANSWERS[case][expect])
    first = [set(opts[t[0]]) for t in top]
    tgt, src = ANSWERS[case]["target"], ANSWERS[case]["source"]
    r = dict(n=len(top),
             exact=float(np.mean([f == want for f in first])),
             top5=float(np.mean([any(set(opts[i]) == want for i in t) for t in top])),
             first_target=float(np.mean([f == set(tgt) for f in first])))
    for inst in dict.fromkeys(tgt + src):
        r["has_" + inst] = float(np.mean([inst in f for f in first]))
    r["common"] = Counter(label(opts[t[0]]) for t in top).most_common(3)
    return r


def common_str(c, n):
    return ", ".join("%s %.0f%%" % (lab, 100 * k / n) for lab, k in c)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc, model = load_model(args.device)
    opts, phrasings = options()
    M = answer_matrix(proc, model, args.device, phrasings)
    print("%d answers (%d single instruments + %d pairs), %d phrasings through the text encoder"
          % (len(opts), len(NAMES), len(opts) - len(NAMES), sum(map(len, phrasings))))
    print("chance: %.1f%% first choice, %.1f%% in the top 5" % (100 / len(opts), 500 / len(opts)))

    per_clip, summary = [], []

    def record(group, setname, E, names):
        top, sc = rank(E, M)
        for i in range(len(top)):
            row = dict(group=group, set=setname, clip=names[i] if names is not None else str(i))
            for j in range(top.shape[1]):
                row["answer_%d" % (j + 1)] = label(opts[top[i, j]])
                row["score_%d" % (j + 1)] = round(float(sc[i, j]), 4)
            per_clip.append(row)
        return top

    groups = []
    for root in args.roots:
        for case_dir in sorted((Path(root) / "fad").glob("*")):
            if case_dir.name in ANSWERS and (case_dir / "reference_clap.npz").is_file():
                groups.append((Path(root), case_dir.name, case_dir))

    # ---- 0. does the test work on real recordings?
    print("\n" + "=" * 118)
    print("  0. Real recordings, known answer -- does the test work?")
    print("=" * 118)
    usable = {}
    for root, case, case_dir in groups:
        z = np.load(case_dir / "reference_clap.npz")
        top = record("%s / %s" % (root.name, case), "REAL target", z["emb"], [str(s) for s in z["names"]])
        r = judge(top, opts, case, "target")
        print("  %-38s real %-14s n=%4d  right answer first %5.1f%%  in top 5 %5.1f%%   most common: %s"
              % ("%s / %s" % (root.name, case), label(ANSWERS[case]["target"]), r["n"], 100 * r["exact"],
                 100 * r["top5"], common_str(r["common"], r["n"])))
        usable.setdefault(case, []).append(r["top5"] >= 0.5)
        summary.append(dict(group="%s / %s" % (root.name, case), set="REAL target", **{k: v for k, v in r.items() if k != "common"},
                            most_common=common_str(r["common"], r["n"])))
    for case in sorted({g[1] for g in groups}):
        src = Path(args.real_source_dir) / ("real_source_%s_clap.npz" % case)
        if not src.is_file():
            print("  %s: no real source embeddings at %s -- check (b) skipped" % (case, src))
            continue
        top = record(case, "REAL source", np.load(src)["emb"], None)
        r = judge(top, opts, case, "source")
        print("  %-38s real %-14s n=%4d  right answer first %5.1f%%  in top 5 %5.1f%%   TARGET answer first %5.1f%%"
              % (case, label(ANSWERS[case]["source"]), r["n"], 100 * r["exact"], 100 * r["top5"], 100 * r["first_target"]))
        print("  %-38s most common: %s" % ("", common_str(r["common"], r["n"])))
        usable[case].append(r["first_target"] < 0.10)
        summary.append(dict(group=case, set="REAL source", **{k: v for k, v in r.items() if k != "common"},
                            most_common=common_str(r["common"], r["n"])))
    for case, checks in sorted(usable.items()):
        print("  -> %s: %s" % (case, "the test WORKS" if all(checks) else
                                  "the test DOES NOT WORK here -- do not read the model rows below"))

    # ---- the models
    for root, case, case_dir in groups:
        group = "%s / %s" % (root.name, case)
        enc, clap = read_fads(root)
        tgt, src = ANSWERS[case]["target"], ANSWERS[case]["source"]
        rows = []
        for p in sorted((case_dir / "generated").glob("*_clap.npz")):
            exp = p.name[:-len("_clap.npz")]
            z = np.load(p)
            r = judge(record(group, exp, z["emb"], [str(s) for s in z["names"]]), opts, case, "target")
            rows.append((clap.get((case, exp), float("nan")), enc.get((case, exp), float("nan")), exp, r))
            summary.append(dict(group=group, set=exp, clap_fad=rows[-1][0], encodec_fad=rows[-1][1],
                                **{k: v for k, v in r.items() if k != "common"},
                                most_common=common_str(r["common"], r["n"])))
        print("\n" + "=" * 118)
        print("  %s -- share of clips (sorted by CLAP-FAD). Target %s, source %s"
              % (group, label(tgt), label(src)))
        print("=" * 118)
        inst_cols = list(dict.fromkeys(tgt + src))
        print("  %-32s %8s %7s %7s %6s  %s   %s" % ("condition", "CLAP-FAD", "EnC-FAD", "exact", "top 5",
              "  ".join("has %-7s" % i[:7] for i in inst_cols), "most common first choices"))
        for c, e, exp, r in sorted(rows):
            print("  %-32s %8.4f %7.2f %6.0f%% %5.0f%%  %s   %s"
                  % (exp, c, e, 100 * r["exact"], 100 * r["top5"],
                     "  ".join("%9.0f%%" % (100 * r["has_" + i]) for i in inst_cols), common_str(r["common"], r["n"])))
        print("  ('has X' = first choice names X, alone or in a pair. Source instruments in the first\n"
              "   choice mean CLAP still hears them.)")

    for name, rows in (("summary.csv", summary), ("per_clip.csv", per_clip)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    print("\n-> %s/summary.csv, %s/per_clip.csv (top 5 answers for every clip)" % (out, out))


if __name__ == "__main__":
    main()
