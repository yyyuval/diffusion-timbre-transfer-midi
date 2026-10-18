"""Blind listening test for the comparisons where EnCodec-FAD and CLAP-FAD disagree.

Both FADs are stable (evaluation/diagnose_fad.py), yet in some comparisons each
prefers the other model -- so at least one of them is not measuring timbre the
way a listener hears it. Ears decide. Each trial plays a real recording of the
target instrument(s), then the SAME input clip converted by the two models, as
A and B in random order; the listener picks the one that sounds more like the
real instrument(s).

What keeps it honest:
  - clips are drawn at random (seeded) from the clips the FAD scored, among
    those held out by BOTH models of a comparison -- never hand-picked;
  - file names and the page carry no condition; the key is written OUTSIDE the
    folder the listener gets;
  - A/B position and trial order are random; all three files of a trial are
    brought to the same loudness, so "louder" cannot win;
  - a control comparison, where both metrics agree by a wide margin, checks
    that the listener can hear a difference at all;
  - which model each metric prefers is read from the FAD result files, and the
    script stops if a comparison no longer disagrees (or the control no longer
    agrees).

Reading, fixed before anyone listens: per comparison, the share of trials in
which the listener chose EnCodec's pick, with an exact two-sided binomial test
against 50%. With 12 trials, 10 or more for one side is p < 0.05.
evaluation/score_listening_test.py does this.

  venv_yuval/bin/python evaluation/make_listening_test.py
  -> Outputs/listening_test/listening_test.zip   (give this to the listener)
     Outputs/listening_test/key.csv              (keep; do not open before scoring)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

A_TIMBRE = "Outputs/WAV_files/clap_validation_1000"
B_ROOT = "Outputs/WAV_files/merged_midi_test"

# name, root, case, model 1, model 2, trials, control?
COMPARISONS = [
    ("mono_sigma5_nomidi_vs_midi", A_TIMBRE, "mono", "mono_c2b_no_midi", "mono_c2b_original_midi", 12, False),
    ("poly_sigma5_nomidi_vs_stacked", A_TIMBRE, "poly", "no_midi", "original_midi", 12, False),
    ("poly_sigma100_merged_generated_vs_real", B_ROOT, "poly",
     "merged_generated_midi_s100", "merged_real_midi_s100", 12, False),
    ("control_poly_sigma100_nomidi_vs_stacked", A_TIMBRE, "poly", "no_midi_s100", "original_midi_s100", 6, True),
]
TARGET_HE = {"poly": "כינור + צ׳לו", "mono": "בסון"}
TARGET_RMS_DBFS = -26.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="Outputs/listening_test")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true",
                   help="rebuild over an existing test. Never do this after someone has "
                        "listened: the old answers would no longer match the key.")
    return p.parse_args()


def read_fad(root):
    """{(case, experiment): (encodec_fad, clap_fad)} from the files the report used."""
    def load(name, col):
        with io.open(Path(root) / "fad" / name, newline="", encoding="utf-8") as f:
            return {(r["case"], r["experiment"]): float(r[col]) for r in csv.DictReader(f) if r.get(col)}
    enc, clap = load("encodec_fad_results.csv", "fad"), load("clap_fad_200.csv", "clap_fad")
    return {k: (enc[k], clap[k]) for k in enc if k in clap}


def generated(root, case, exp):
    d = Path(root) / "fad" / case / "generated" / exp
    clips = {p.stem: p for p in sorted(d.glob("*.wav"))}
    if not clips:
        raise SystemExit("no generated clips in %s" % d)
    return clips


def load_mono(path):
    import torchaudio
    wav, sr = torchaudio.load(str(Path(path).resolve()))
    return wav.mean(dim=0).numpy().astype(np.float64), sr


def write_wav16(path, x, sr):
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def level_match(x):
    """Same RMS for every file; a file whose peaks would clip is lowered and counted."""
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    y = x * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = np.max(np.abs(y))
    return (y * 0.99 / peak, True) if peak > 0.99 else (y, False)


def main():
    args = parse_args()
    out = Path(args.out)
    key_path = out / "key.csv"
    if key_path.exists() and not args.force:
        raise SystemExit("%s exists -- a test was built already. Score it, or pass --force "
                         "if nobody has listened to it yet." % key_path)
    if out.exists():
        if any(out.iterdir()) and not (out / "build.json").exists():
            raise SystemExit("%s is not empty and is not a listening test -- choose another --out" % out)
        shutil.rmtree(out)
    audio_dir = out / "listening_test" / "audio"
    audio_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
    build_id = "%s-seed%d" % (time.strftime("%Y%m%d%H%M%S"), args.seed)

    trials, meta = [], []
    for name, root, case, m1, m2, n, control in COMPARISONS:
        fad = read_fad(root)
        (e1, c1), (e2, c2) = fad[(case, m1)], fad[(case, m2)]
        enc_pick, clap_pick = (m1 if e1 < e2 else m2), (m1 if c1 < c2 else m2)
        if control != (enc_pick == clap_pick):
            raise SystemExit("%s: EnCodec picks %s, CLAP picks %s -- not what this comparison is for"
                             % (name, enc_pick, clap_pick))
        g1, g2 = generated(root, case, m1), generated(root, case, m2)
        shared = sorted(set(g1) & set(g2))
        refs = sorted((Path(root) / "fad" / case / "reference").glob("*.wav"))
        if len(shared) < n or not refs:
            raise SystemExit("%s: %d shared clips, %d references -- not enough" % (name, len(shared), len(refs)))
        picked = rng.sample(shared, n)
        meta.append(dict(comparison=name, case=case, model_1=m1, model_2=m2, control=control,
                         encodec=(e1, e2), clap=(c1, c2), encodec_pick=enc_pick, clap_pick=clap_pick,
                         shared_clips=len(shared), trials=n))
        print("%-42s %s  EnCodec %.2f vs %.2f -> %s | CLAP %.4f vs %.4f -> %s | %d trials of %d shared clips"
              % (name, "CONTROL" if control else "", e1, e2, enc_pick, c1, c2, clap_pick, n, len(shared)))
        for sid in picked:
            first = rng.random() < 0.5
            trials.append(dict(comparison=name, case=case, control=control, sample_id=sid,
                               reference=rng.choice(refs),
                               A=m1 if first else m2, B=m2 if first else m1,
                               A_path=g1[sid] if first else g2[sid], B_path=g2[sid] if first else g1[sid],
                               enc_pick=enc_pick, clap_pick=clap_pick))
    rng.shuffle(trials)

    lowered = 0
    for i, t in enumerate(trials, 1):
        t["id"] = "t%02d" % i
        for slot, src in (("ref", t["reference"]), ("A", t["A_path"]), ("B", t["B_path"])):
            x, sr = load_mono(src)
            y, low = level_match(x)
            lowered += low
            write_wav16(audio_dir / ("%s_%s.wav" % (t["id"], slot)), y, sr)
    print("\n%d trials, %d files written to %s (%d lowered below the common level to avoid clipping)"
          % (len(trials), 3 * len(trials), audio_dir, lowered))

    with key_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["build_id", "trial", "comparison", "control", "case", "sample_id", "reference",
                    "A", "B", "encodec_choice", "clap_choice"])
        for t in trials:
            w.writerow([build_id, t["id"], t["comparison"], int(t["control"]), t["case"], t["sample_id"],
                        t["reference"].stem, t["A"], t["B"],
                        "A" if t["enc_pick"] == t["A"] else "B", "A" if t["clap_pick"] == t["A"] else "B"])
    (out / "build.json").write_text(json.dumps(dict(build_id=build_id, seed=args.seed, comparisons=meta),
                                               indent=2, ensure_ascii=False), encoding="utf-8")

    page = PAGE.replace("__BUILD__", build_id).replace("__TRIALS__", json.dumps(
        [dict(id=t["id"], target=TARGET_HE[t["case"]]) for t in trials], ensure_ascii=False))
    (out / "listening_test" / "index.html").write_text(page, encoding="utf-8")
    zip_path = shutil.make_archive(str(out / "listening_test"), "zip", root_dir=out, base_dir="listening_test")
    print("listener gets: %s  (%.0f MB)" % (zip_path, Path(zip_path).stat().st_size / 1e6))
    print("keep, do not open before scoring: %s" % key_path)


PAGE = r"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>מבחן האזנה – טימבר</title>
<style>
:root { --bg:#f6f5f2; --card:#ffffff; --ink:#1c1b19; --muted:#5f5b55; --line:#e3e0da;
        --accent:#1f4fbf; --accent-ink:#ffffff; --soft:#e5ecfb; --done:#1d7a46; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#171615; --card:#22211f; --ink:#f2f0ec; --muted:#a8a39b;
        --line:#3a3835; --accent:#7aa2f7; --accent-ink:#0e1424; --soft:#26324d; --done:#5fd08d; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); line-height:1.55;
       font-family: system-ui, "Segoe UI", Arial, sans-serif; }
main { max-width:760px; margin:0 auto; padding:28px 16px 72px; }
h1 { font-size:1.5rem; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin-bottom:16px; }
.card h2 { font-size:1.05rem; margin:0 0 10px; }
ol, ul { padding-inline-start:20px; margin:8px 0; }
li { margin:4px 0; }
label.name { display:block; font-weight:600; margin:12px 0 6px; }
input[type=text] { width:100%; max-width:320px; padding:10px 12px; font:inherit; color:var(--ink);
                   background:var(--bg); border:1px solid var(--line); border-radius:8px; }
button { font:inherit; cursor:pointer; border-radius:8px; padding:10px 18px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); }
button.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:600; }
button:disabled { opacity:.45; cursor:default; }
.progress { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:6px 0 18px; }
.progress > div { height:100%; background:var(--accent); width:0; transition:width .2s; }
.row { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
.player { margin:10px 0 16px; }
.player .lbl { font-weight:600; margin-bottom:4px; }
.player .lbl small { font-weight:400; color:var(--muted); }
audio { width:100%; }
.choices { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:10px 0 14px; }
.choice { padding:16px; font-size:1.15rem; font-weight:700; }
.choice.on { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
.conf { display:flex; gap:8px; flex-wrap:wrap; }
.conf button.on { background:var(--soft); border-color:var(--accent); }
.q { font-weight:600; margin:6px 0 0; }
.muted { color:var(--muted); }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(44px, 1fr)); gap:6px; margin-top:12px; }
.grid button { padding:6px 0; font-size:.85rem; }
.grid button.ans { border-color:var(--done); color:var(--done); font-weight:700; }
.grid button.cur { outline:2px solid var(--accent); }
.foot { margin-top:22px; display:flex; gap:10px; flex-wrap:wrap; }
</style>
</head>
<body>
<main id="app"></main>
<script>
const BUILD = "__BUILD__";
const TRIALS = __TRIALS__;
const STORE = "listening-test-" + BUILD;
let S = { listener: "", answers: {}, pos: -1 };
try { const s = JSON.parse(localStorage.getItem(STORE)); if (s && s.answers) S = s; } catch (e) {}
function save() { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) {} }
const app = document.getElementById("app");
const answered = () => TRIALS.filter(t => (S.answers[t.id] || {}).choice).length;
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function intro() {
  app.innerHTML = `
    <h1>מבחן האזנה – איזו המרה נשמעת כמו הכלי האמיתי?</h1>
    <p class="sub">${TRIALS.length} ניסיונות · כ־40–50 דקות · אפשר לעצור ולהמשיך, התשובות נשמרות בדפדפן</p>
    <div class="card"><h2>איך זה עובד</h2>
      <ol>
        <li>בכל ניסיון יש <b>קטע ייחוס</b>: הקלטה אמיתית של כלי היעד (כינור + צ׳לו, או בסון). זו יצירה אחרת – הוא שם רק כדי להזכיר איך הכלי נשמע.</li>
        <li>אחריו <b>A</b> ו־<b>B</b>: אותו קטע קלט, שהומר על ידי שני מודלים שונים. הסדר אקראי בכל ניסיון.</li>
        <li>בחר את ההמרה ש<b>נשמעת יותר כמו הכלים האמיתיים</b> בקטע הייחוס, וסמן אם אתה בטוח או מנחש.</li>
      </ol>
      <ul class="muted">
        <li>שפוט רק את <b>צליל הכלים</b> (הטימבר) – לא את המנגינה ולא את התווים.</li>
        <li>רעשים וארטיפקטים נחשבים רק במידה שהם גורמים לכלי להישמע פחות אמיתי.</li>
        <li>כל הקבצים הושוו לאותה עוצמה. עדיף אוזניות, בחדר שקט. מותר לשמוע כל קטע כמה פעמים שרוצים.</li>
        <li>אין תשובה "שווים" – אם באמת אין הבדל, בחר את מה שנראה לך וסמן "מנחש".</li>
      </ul>
      <label class="name" for="nm">השם שלך</label>
      <input type="text" id="nm" value="${esc(S.listener)}" autocomplete="off">
      <div class="foot"><button class="primary" id="go">${answered() ? "המשך מאיפה שעצרתי" : "התחל"}</button></div>
    </div>`;
  const nm = document.getElementById("nm");
  document.getElementById("go").onclick = () => {
    if (!nm.value.trim()) { nm.focus(); return; }
    S.listener = nm.value.trim();
    if (S.pos < 0) S.pos = 0;
    const firstOpen = TRIALS.findIndex(t => !(S.answers[t.id] || {}).choice);
    if (firstOpen >= 0 && answered()) S.pos = firstOpen;
    save(); trial();
  };
}

function player(src, label, hint) {
  return `<div class="player"><div class="lbl">${label} ${hint ? "<small>" + hint + "</small>" : ""}</div>
          <audio controls preload="none" src="audio/${src}"></audio></div>`;
}

function trial() {
  if (S.pos >= TRIALS.length) return finish();
  const t = TRIALS[S.pos], a = S.answers[t.id] || {};
  app.innerHTML = `
    <div class="row"><b>ניסיון ${S.pos + 1} מתוך ${TRIALS.length}</b>
      <span class="muted">נענו ${answered()} · ${esc(S.listener)}</span></div>
    <div class="progress"><div style="width:${100 * answered() / TRIALS.length}%"></div></div>
    <div class="card">
      ${player(t.id + "_ref.wav", "קטע ייחוס", "– הקלטה אמיתית של " + t.target)}
      ${player(t.id + "_A.wav", "A", "")}
      ${player(t.id + "_B.wav", "B", "")}
      <p class="q">איזו המרה נשמעת יותר כמו ${t.target} האמיתיים?</p>
      <div class="choices">
        <button class="choice ${a.choice === "A" ? "on" : ""}" data-c="A">A</button>
        <button class="choice ${a.choice === "B" ? "on" : ""}" data-c="B">B</button>
      </div>
      <div class="conf">
        <button data-f="sure" class="${a.confidence === "sure" ? "on" : ""}">בטוח</button>
        <button data-f="guess" class="${a.confidence === "guess" ? "on" : ""}">מנחש</button>
      </div>
      <div class="foot">
        <button id="prev" ${S.pos === 0 ? "disabled" : ""}>הקודם</button>
        <button class="primary" id="next" ${a.choice && a.confidence ? "" : "disabled"}>${S.pos + 1 === TRIALS.length ? "סיום" : "הבא"}</button>
      </div>
    </div>
    <div class="card"><div class="row"><span class="muted">מעבר לניסיון (ירוק = נענה)</span>
      <button id="dl">הורד תשובות</button></div>
      <div class="grid">${TRIALS.map((x, i) => `<button data-j="${i}" class="${(S.answers[x.id] || {}).choice ? "ans" : ""} ${i === S.pos ? "cur" : ""}">${i + 1}</button>`).join("")}</div>
    </div>`;
  const audios = [...app.querySelectorAll("audio")];
  audios.forEach(el => el.addEventListener("play", () => audios.forEach(o => { if (o !== el) o.pause(); })));
  const set = (k, v) => { S.answers[t.id] = Object.assign({}, S.answers[t.id], { [k]: v, at: new Date().toISOString() }); save(); trial(); };
  app.querySelectorAll("[data-c]").forEach(b => b.onclick = () => set("choice", b.dataset.c));
  app.querySelectorAll("[data-f]").forEach(b => b.onclick = () => set("confidence", b.dataset.f));
  app.querySelectorAll("[data-j]").forEach(b => b.onclick = () => { S.pos = +b.dataset.j; save(); trial(); });
  document.getElementById("prev").onclick = () => { S.pos--; save(); trial(); };
  document.getElementById("next").onclick = () => { S.pos++; save(); trial(); };
  document.getElementById("dl").onclick = download;
}

function finish() {
  const n = answered();
  app.innerHTML = `
    <h1>${n === TRIALS.length ? "סיימת – תודה!" : "כמעט"}</h1>
    <p class="sub">נענו ${n} מתוך ${TRIALS.length} ניסיונות.</p>
    <div class="card">
      <p>לחץ על <b>הורד תשובות</b> ושלח את הקובץ שירד (answers_….csv). אל תפתח את קובץ המפתח לפני שהתשובות נשלחו.</p>
      <div class="foot"><button class="primary" id="dl">הורד תשובות</button>
      <button id="back">חזרה לניסיונות</button></div>
    </div>`;
  document.getElementById("dl").onclick = download;
  document.getElementById("back").onclick = () => {
    const open = TRIALS.findIndex(t => !(S.answers[t.id] || {}).choice);
    S.pos = open >= 0 ? open : TRIALS.length - 1; save(); trial();
  };
}

function download() {
  const rows = [["build_id", "listener", "trial", "choice", "confidence", "answered_at"]];
  TRIALS.forEach(t => { const a = S.answers[t.id] || {};
    if (a.choice) rows.push([BUILD, S.listener, t.id, a.choice, a.confidence || "", a.at || ""]); });
  const csv = rows.map(r => r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(",")).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "answers_" + (S.listener || "listener").replace(/[^\w֐-׿-]+/g, "_") + ".csv";
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

if (S.pos < 0 || !S.listener) intro(); else trial();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
