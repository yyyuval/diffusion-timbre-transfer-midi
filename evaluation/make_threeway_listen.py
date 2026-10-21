"""Three-way listening test: no MIDI against MIDI at sigma_max 5 and 100.

Fifteen questions in three blocks of five. Each question plays a real violin +
cello recording (another piece), the flute + bassoon input, and the same input
converted three ways, as A / B / C in random order. The listener picks the one
that sounds most like real violin + cello; then a button reveals which was
which, with each model's EnCodec-FAD and the clip's own distance to real
violin + cello.

  block 1  separate real roll per instrument: no_midi | original_midi | original_midi_s100
  block 2  one merged real roll:              no_midi | merged_real_midi | merged_real_midi_s100
  block 3  trained on real MIDI, given a roll generated from the audio:
                                              no_midi | swap_realtrained_genroll | swap_realtrained_genroll_s100

sigma_max = 5 models run at handoff 5, sigma_max = 100 models at handoff 100,
as in the result tables.

The three clips of a question must come from the same input, and that input
must be unseen by every model in the question -- so the tracks are chosen
here (seeded, never by ear) from the recovered training splits, and the
clips are generated for them by evaluation/run_threeway_listen.sh:

  --select   15 tracks: block 1 from those both the no-MIDI and the stacked
             source models held out (evaluation/splits.csv); blocks 2 and 3
             from those the no-MIDI source model held out AND that are in the
             merged models' held-out list. Writes Outputs/threeway_listen/tracks_*.txt
  --build    after generation: the page, the audio, the key -> threeway_listen.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path("Outputs/threeway_listen")
GEN = OUT / "gen"
A_ROOT = Path("Outputs/WAV_files/clap_validation_1000")
B_ROOT = Path("Outputs/WAV_files/merged_midi_test")
STEMS = "1_flute+4_bassoon"
PER_BLOCK = 5

# experiment: (label, kind, test set of its FAD, FAD csv, root holding its 1,000 cached embeddings)
COND = {
    "no_midi": ("בלי MIDI", "nomidi", "A", A_ROOT / "fad/encodec_fad_results.csv", A_ROOT),
    "original_midi": ("MIDI אמיתי, רול נפרד לכל כלי", "s5", "A",
                      A_ROOT / "fad/encodec_fad_results.csv", A_ROOT),
    "original_midi_s100": ("MIDI אמיתי, רול נפרד לכל כלי", "s100", "A",
                           A_ROOT / "fad/encodec_fad_results.csv", A_ROOT),
    "merged_real_midi": ("MIDI אמיתי, רול מאוחד", "s5", "B",
                         B_ROOT / "fad/encodec_fad_results.csv", B_ROOT),
    "merged_real_midi_s100": ("MIDI אמיתי, רול מאוחד", "s100", "B",
                              B_ROOT / "fad/encodec_fad_results.csv", B_ROOT),
    "swap_realtrained_genroll": ("אומן על MIDI אמיתי, קיבל רול מגונרט", "s5", "B",
                                 B_ROOT / "roll_swap/encodec_fad_s5.csv", B_ROOT),
    "swap_realtrained_genroll_s100": ("אומן על MIDI אמיתי, קיבל רול מגונרט", "s100", "B",
                                      B_ROOT / "roll_swap/encodec_fad.csv", B_ROOT),
}
BLOCKS = [
    ("MIDI אמיתי, רול נפרד לכל כלי", ["no_midi", "original_midi", "original_midi_s100"]),
    ("MIDI אמיתי, רול מאוחד", ["no_midi", "merged_real_midi", "merged_real_midi_s100"]),
    ("אומן על MIDI אמיתי, קיבל רול מגונרט מהאודיו",
     ["no_midi", "swap_realtrained_genroll", "swap_realtrained_genroll_s100"]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--link-s5", action="store_true",
                      help="link swap_realtrained_genroll's 1,000 set-B clips into fad/poly/generated/, "
                           "for compute_encodec_fad.py (only with FULL_S5=1 in the run script)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--splits", default="evaluation/splits.csv")
    p.add_argument("--merged-list", default="evaluation/merged_test_lists/track_list_poly_merged.txt")
    p.add_argument("--device", default=None)
    return p.parse_args()


def read_list(path):
    lines = io.open(path, encoding="utf-8").read().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def held_out(splits, column):
    """Woodwind tracks with flute + bassoon stems in the VALIDATION block of one run."""
    out = set()
    for r in csv.DictReader(io.open(splits, encoding="utf-8")):
        v = (r.get(column) or "").strip()
        if r["block"] == "VALIDATION" and v.endswith("/" + STEMS):
            out.add(v.split("/")[0])
    return out


def select(args):
    nomidi = held_out(args.splits, "flute_bassoon_nomidi")
    stacked = held_out(args.splits, "flute_bassoon_midi")
    merged = set(read_list(args.merged_list))
    cand1 = sorted(nomidi & stacked)
    cand23 = sorted(nomidi & merged)
    print("held out by the no-MIDI source model: %d | by the stacked one: %d | merged list: %d"
          % (len(nomidi), len(stacked), len(merged)))
    print("candidates: block 1 %d, blocks 2-3 %d" % (len(cand1), len(cand23)))
    rng = random.Random(args.seed)
    b1 = rng.sample(cand1, PER_BLOCK)
    rest = [t for t in cand23 if t not in b1]
    if len(rest) < 2 * PER_BLOCK:
        raise SystemExit("only %d candidates for blocks 2-3" % len(rest))
    b23 = rng.sample(rest, 2 * PER_BLOCK)
    blocks = [b1, b23[:PER_BLOCK], b23[PER_BLOCK:]]
    for cache in ("midi_cache_wood_fps75_gt_merged", "midi_cache_wood_fps75_bp_merged"):
        for t in blocks[1] + blocks[2]:
            if not (Path(cache) / t / "merged_flute_bassoon_pianoroll.npy").is_file():
                raise SystemExit("no merged roll for %s in %s" % (t, cache))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for i, b in enumerate(blocks, 1):
        (OUT / ("tracks_block%d.txt" % i)).write_text("\n".join(b) + "\n", encoding="utf-8")
        print("block %d: %s" % (i, " ".join(b)))
    (OUT / "tracks_all.txt").write_text("\n".join(sum(blocks, [])) + "\n", encoding="utf-8")


def read_fad(path, exp):
    if not Path(path).is_file():
        return None
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        if r["experiment"] == exp and r.get("case", "") in ("poly", ""):
            return float(r["fad"])
    return None


def build(args):
    import torch
    from audio_diffusion_pytorch import NormalizedEncodec
    from evaluation.compute_encodec_fad import load_audio_24k
    from evaluation.make_listening_test import PAGE as BLIND_PAGE, level_match, load_mono, write_wav16

    rows = {(r["experiment"], r["sample_id"]): r
            for r in csv.DictReader(io.open(GEN / "bridge_metrics_all.csv", encoding="utf-8"))}
    blocks = [read_list(OUT / ("tracks_block%d.txt" % i)) for i in (1, 2, 3)]
    for (_, exps), tracks in zip(BLOCKS, blocks):
        missing = [(e, t) for e in exps for t in tracks if (e, t) not in rows]
        if missing:
            raise SystemExit("not generated yet: %s" % missing[:5])

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = NormalizedEncodec(device=device)

    def embed(path):
        wav = load_audio_24k(Path(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            return enc.encode_latent(wav, None, None).mean(dim=-1).squeeze(0).cpu().numpy()

    ref = np.load(B_ROOT / "fad/poly/reference_encodec.npz", allow_pickle=False)["emb"]
    mu = ref.mean(0)
    real_median = float(np.median(np.linalg.norm(ref - mu, axis=1)))
    model_d = {}
    for exp, (_, _, _, _, root) in COND.items():
        cache = root / "fad/poly/generated" / ("%s_encodec.npz" % exp)
        if cache.is_file():
            model_d[exp] = np.sort(np.linalg.norm(np.load(cache, allow_pickle=False)["emb"] - mu, axis=1))
    fads = {exp: read_fad(c[3], exp) for exp, c in COND.items()}

    refs = sorted((B_ROOT / "fad/poly/reference").glob("*.wav"))
    audio = OUT / "threeway_listen" / "audio"
    if audio.parent.exists():
        shutil.rmtree(audio.parent)
    audio.mkdir(parents=True)
    rng = random.Random(args.seed + 1)
    build_id = "%s-seed%d" % (time.strftime("%Y%m%d%H%M%S"), args.seed)
    trials, key = [], []
    for bi, ((title, exps), tracks) in enumerate(zip(BLOCKS, blocks), 1):
        order = list(tracks)
        rng.shuffle(order)
        for track in order:
            tid = "t%02d" % (len(trials) + 1)
            perm = list(exps)
            rng.shuffle(perm)
            files = [("ref", rng.choice(refs)), ("input", rows[(exps[0], track)]["input_audio"])]
            opts = {}
            for slot, exp in zip("ABC", perm):
                path = rows[(exp, track)]["output_audio"]
                files.append((slot, path))
                d = float(np.linalg.norm(embed(path) - mu))
                pct = None
                if exp in model_d:
                    pct = 100.0 * float(np.mean(model_d[exp] > d))   # share of the model's clips farther away
                label, kind, tset, _, _ = COND[exp]
                opts[slot] = dict(label=label, kind=kind, sigma=100 if kind == "s100" else 5, fad=fads[exp], set=tset, d=d, pct=pct,
                                  med=float(np.median(model_d[exp])) if exp in model_d else None)
            for slot, src in files:
                x, sr = load_mono(src)
                write_wav16(audio / ("%s_%s.wav" % (tid, slot)), level_match(x)[0], sr)
            trials.append(dict(id=tid, block=bi, title=title, opts=opts))
            key.append([build_id, tid, bi, track] + perm)
            print("%s block %d %s  A=%s (%.2f)  B=%s (%.2f)  C=%s (%.2f)"
                  % (tid, bi, track, perm[0], opts["A"]["d"], perm[1], opts["B"]["d"], perm[2], opts["C"]["d"]))

    with (OUT / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["build_id", "trial", "block", "sample_id", "A", "B", "C"])
        w.writerows(key)
    style = re.search(r"<style>.*?</style>", BLIND_PAGE, re.S).group(0)
    page = (PAGE.replace("__STYLE__", style).replace("__BUILD__", build_id)
            .replace("__REAL_MEDIAN__", "%.2f" % real_median)
            .replace("__TRIALS__", json.dumps(trials, ensure_ascii=False)))
    (OUT / "threeway_listen" / "index.html").write_text(page, encoding="utf-8")
    zip_path = shutil.make_archive(str(OUT / "threeway_listen"), "zip", root_dir=OUT, base_dir="threeway_listen")
    print("\n%d questions -> %s (%.0f MB)" % (len(trials), zip_path, Path(zip_path).stat().st_size / 1e6))
    missing = [e for e, v in fads.items() if v is None]
    if missing:
        print("no FAD on record for: %s (shown as 'not measured')" % ", ".join(missing))


PAGE = r"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>מבחן האזנה – שלוש דרכים</title>
__STYLE__
<style>
.choices { grid-template-columns:1fr 1fr 1fr; }
.block { display:inline-block; font-size:.85rem; font-weight:600; color:var(--accent);
         background:var(--soft); border-radius:999px; padding:3px 12px; margin-bottom:10px; }
.reveal { margin-top:12px; padding:12px 14px; border-radius:8px; background:var(--soft); }
.reveal .opt { margin:6px 0; }
.reveal .mine { font-weight:700; }
.small { font-size:.9rem; }
table.sum { border-collapse:collapse; width:100%; margin:8px 0 14px; }
table.sum th, table.sum td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; }
</style>
</head>
<body>
<main id="app"></main>
<script>
const BUILD = "__BUILD__";
const TRIALS = __TRIALS__;
const STORE = "threeway-listen-" + BUILD;
const SIG = v => `<bdi dir="ltr">σmax = ${v}</bdi>`;
const KIND = { nomidi: "בלי MIDI, " + SIG(5), s5: "עם MIDI, " + SIG(5), s100: "עם MIDI, " + SIG(100) };
let S = { answers: {}, pos: -1 };
try { const s = JSON.parse(localStorage.getItem(STORE)); if (s && s.answers) S = s; } catch (e) {}
function save() { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) {} }
const app = document.getElementById("app");
const answered = () => TRIALS.filter(t => (S.answers[t.id] || {}).choice).length;

function intro() {
  app.innerHTML = `
    <h1>מבחן האזנה – בלי MIDI, ועם MIDI בשתי רמות רעש</h1>
    <p class="sub">${TRIALS.length} שאלות בשלושה חלקים · כ־25 דקות · התשובות נשמרות בדפדפן</p>
    <div class="card"><h2>איך זה עובד</h2>
      <ol>
        <li><b>קטע ייחוס</b>: הקלטה אמיתית של כינור וצ׳לו, מיצירה אחרת. רק כדי להזכיר איך הכלים נשמעים.</li>
        <li><b>הקלט</b>: חליל ובסון, הקטע שהומר.</li>
        <li><b>A</b>, <b>B</b> ו־<b>C</b>: אותו קלט אחרי שלוש המרות שונות. הסדר אקראי בכל שאלה.</li>
        <li>בחר את ההמרה שנשמעת הכי הרבה כמו כינור וצ׳לו אמיתיים, ואז אפשר ללחוץ <b>הצג תשובה</b>.</li>
      </ol>
      <p>בכל שאלה: המרה אחת בלי MIDI עם ${SIG(5)}, אחת עם MIDI עם ${SIG(5)}, ואחת עם MIDI עם ${SIG(100)}. ההבדל בין שלושת החלקים הוא סוג ה־MIDI:</p>
      <ol>
        <li>רול נפרד לכל כלי, MIDI אמיתי.</li>
        <li>רול מאוחד לשני הכלים, MIDI אמיתי.</li>
        <li>מודל שאומן על MIDI אמיתי, אבל מקבל רול שגונרט מהאודיו – המצב של הקלטה אמיתית.</li>
      </ol>
      <p class="muted">שפוט את צליל הכלים (הטימבר), לא את המנגינה. כל הקבצים באותה עוצמה. עדיף אוזניות.
         כפתור התשובה נפתח רק אחרי שבחרת, כדי שהבחירה תהיה עיוורת.</p>
      <div class="foot"><button class="primary" id="go">${answered() ? "המשך" : "התחל"}</button></div>
    </div>`;
  document.getElementById("go").onclick = () => { if (S.pos < 0) S.pos = 0; save(); trial(); };
}

function player(src, label, hint) {
  return `<div class="player"><div class="lbl">${label} ${hint ? "<small>" + hint + "</small>" : ""}</div>
          <audio controls preload="none" src="audio/${src}"></audio></div>`;
}

function optLine(slot, o, mine) {
  const fad = o.fad == null ? "FAD לא נמדד" : `<bdi dir="ltr">FAD ${o.fad.toFixed(2)}</bdi> (סט ${o.set})`;
  const rank = o.pct == null ? "" : ` · קרוב יותר מ־${Math.round(o.pct)} אחוז מהקטעים של המודל (חציון ${o.med.toFixed(2)})`;
  return `<div class="opt ${mine ? "mine" : ""}"><b>${slot}</b>: ${o.label}, ${SIG(o.sigma)} · ${fad}${mine ? " · בחרת" : ""}
          <div class="muted small">מרחק הקטע הזה: <b>${o.d.toFixed(2)}</b>${rank}</div></div>`;
}

function trial() {
  if (S.pos >= TRIALS.length) return finish();
  const t = TRIALS[S.pos], a = S.answers[t.id] || {};
  const reveal = a.revealed ? `<div class="reveal">
      ${["A", "B", "C"].map(s => optLine(s, t.opts[s], a.choice === s)).join("")}
      <div class="muted small" style="margin-top:8px">FAD קיים רק לאוסף של קטעים; הוא של המודל, על 1,000 קטעים. סט A וסט B מושווים להקלטות ייחוס שונות, לכן השוואה ביניהם מקורבת.
        "מרחק" הוא המרחק של הקטע הזה מהממוצע של הקלטות כינור וצ׳לו אמיתיות, במרחב של EnCodec שה־FAD משתמש בו. נמוך יותר = קרוב יותר. הקלטה אמיתית טיפוסית: __REAL_MEDIAN__.</div>
    </div>` : "";
  const dis = a.revealed ? "disabled" : "";
  app.innerHTML = `
    <div class="row"><b>שאלה ${S.pos + 1} מתוך ${TRIALS.length}</b><span class="muted">נענו ${answered()}</span></div>
    <div class="progress"><div style="width:${100 * answered() / TRIALS.length}%"></div></div>
    <div class="card">
      <span class="block">חלק ${t.block} מתוך 3 · ${t.title}</span>
      ${player(t.id + "_ref.wav", "קטע ייחוס", "– כינור וצ׳לו אמיתיים, יצירה אחרת")}
      ${player(t.id + "_input.wav", "הקלט", "– חליל ובסון")}
      ${player(t.id + "_A.wav", "A", "")}
      ${player(t.id + "_B.wav", "B", "")}
      ${player(t.id + "_C.wav", "C", "")}
      <p class="q">איזו המרה נשמעת הכי הרבה כמו כינור וצ׳לו אמיתיים?</p>
      <div class="choices">
        ${["A", "B", "C"].map(s => `<button class="choice ${a.choice === s ? "on" : ""}" data-c="${s}" ${dis}>${s}</button>`).join("")}
      </div>
      <div class="conf">
        <button data-f="sure" class="${a.confidence === "sure" ? "on" : ""}" ${dis}>בטוח</button>
        <button data-f="guess" class="${a.confidence === "guess" ? "on" : ""}" ${dis}>מנחש</button>
      </div>
      <div class="foot"><button id="show" ${a.choice && !a.revealed ? "" : "disabled"}>הצג תשובה</button></div>
      ${reveal}
      <div class="foot">
        <button id="prev" ${S.pos === 0 ? "disabled" : ""}>הקודם</button>
        <button class="primary" id="next" ${a.choice ? "" : "disabled"}>${S.pos + 1 === TRIALS.length ? "סיכום" : "הבא"}</button>
      </div>
    </div>`;
  const audios = [...app.querySelectorAll("audio")];
  audios.forEach(el => el.addEventListener("play", () => audios.forEach(o => { if (o !== el) o.pause(); })));
  const set = (k, v) => { S.answers[t.id] = Object.assign({}, S.answers[t.id], { [k]: v, at: new Date().toISOString() }); save(); trial(); };
  app.querySelectorAll("[data-c]").forEach(b => b.onclick = () => set("choice", b.dataset.c));
  app.querySelectorAll("[data-f]").forEach(b => b.onclick = () => set("confidence", b.dataset.f));
  document.getElementById("show").onclick = () => set("revealed", true);
  document.getElementById("prev").onclick = () => { S.pos--; save(); trial(); };
  document.getElementById("next").onclick = () => { S.pos++; save(); trial(); };
}

function finish() {
  const rows = [1, 2, 3].map(b => {
    const ts = TRIALS.filter(t => t.block === b && (S.answers[t.id] || {}).choice);
    const n = k => ts.filter(t => t.opts[S.answers[t.id].choice].kind === k).length;
    const title = TRIALS.find(t => t.block === b).title;
    return `<tr><td>${b}. ${title}</td><td>${n("nomidi")}</td><td>${n("s5")}</td><td>${n("s100")}</td><td>${ts.length}</td></tr>`;
  }).join("");
  app.innerHTML = `
    <h1>סיכום</h1>
    <div class="card">
      <p>כמה פעמים בחרת בכל סוג המרה, בכל חלק:</p>
      <table class="sum"><tr><th>חלק</th><th>${KIND.nomidi}</th><th>${KIND.s5}</th><th>${KIND.s100}</th><th>נענו</th></tr>${rows}</table>
      <p class="muted">חמש שאלות בכל חלק לא מספיקות למובהקות סטטיסטית. זו בדיקה אם האוזן מסכימה עם המדדים, לא הוכחה.</p>
      <div class="foot"><button class="primary" id="dl">הורד תשובות</button><button id="back">חזרה לשאלות</button></div>
    </div>`;
  document.getElementById("dl").onclick = download;
  document.getElementById("back").onclick = () => { S.pos = 0; save(); trial(); };
}

function download() {
  const rows = [["build_id", "trial", "block", "choice", "chosen_kind", "chosen_label", "confidence", "revealed", "answered_at"]];
  TRIALS.forEach(t => { const a = S.answers[t.id] || {};
    if (a.choice) rows.push([BUILD, t.id, t.block, a.choice, t.opts[a.choice].kind, t.opts[a.choice].label,
                             a.confidence || "", a.revealed ? 1 : 0, a.at || ""]); });
  const csv = rows.map(r => r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(",")).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url; link.download = "threeway_answers.csv";
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

if (S.pos < 0) intro(); else trial();
</script>
</body>
</html>
"""


def link_s5():
    import os

    exp = "swap_realtrained_genroll"
    rows = [r for r in csv.DictReader(io.open(B_ROOT / "bridge_metrics_all.csv", encoding="utf-8"))
            if r["experiment"] == exp]
    if len(rows) < 1000:
        raise SystemExit("%s: %d clips in set B, expected 1000" % (exp, len(rows)))
    d = B_ROOT / "fad/poly/generated" / exp
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        src = Path(r["output_audio"])
        src = src if src.is_absolute() else Path.cwd() / src
        dst = d / ("%s.wav" % r["sample_id"])
        if not (dst.is_symlink() or dst.exists()):
            os.symlink(src, dst)
    print("%s: %d clips linked into %s" % (exp, len(rows), d))


def main():
    args = parse_args()
    if args.select:
        select(args)
    elif args.link_s5:
        link_s5()
    else:
        build(args)


if __name__ == "__main__":
    main()
