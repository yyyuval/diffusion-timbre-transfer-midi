"""Ear check of the roll swap: a recording without real MIDI, two ways to convert it.

Both bridges get the SAME input and the SAME roll generated from it by Basic
Pitch -- the only roll a real recording would have. They differ in what their
models were trained on:

  merged_generated_midi_s100      trained on generated rolls    EnCodec-FAD 6.13
  swap_realtrained_genroll_s100   trained on real MIDI          EnCodec-FAD 3.30

Each trial plays a real violin + cello recording (another piece, to recall the
sound), the flute + bassoon input, and the two conversions as A and B in random
order. The listener picks the one that sounds more like real violin + cello;
after answering, a button reveals which model was which. The key is therefore
in the page by design -- this is a quick check by ear, not the blind test of
make_listening_test.py. Five trials cannot reach significance (5 of 5 is p =
0.06); read it as "do the ears agree with the FAD".

Clips are drawn at random (seeded) from the 1,000 inputs both bridges share,
never hand-picked, and every file is brought to the same loudness.

  venv_yuval/bin/python evaluation/make_rollswap_listen.py
  -> Outputs/rollswap_listen/rollswap_listen.zip
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.make_listening_test import PAGE as BLIND_PAGE, level_match, load_mono, write_wav16  # noqa: E402

B_ROOT = Path("Outputs/WAV_files/merged_midi_test")
OLD, NEW = "merged_generated_midi_s100", "swap_realtrained_genroll_s100"
LABEL = {
    OLD: "המודל שאומן על MIDI מגונרט (FAD 6.13)",
    NEW: "המודל שאומן על MIDI אמיתי (FAD 3.30)",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="Outputs/rollswap_listen")
    return p.parse_args()


def clip_distances():
    """Per-clip distance to the mean of the real violin + cello clips, in the same
    raw EnCodec space the FAD uses, from the cached embeddings.

    FAD itself exists only for a set of clips. Its mean term is the squared
    distance between the two sets' mean embeddings; this is the per-clip
    version of that. Also prints how much of each FAD is that mean term and
    how much is the covariance term, which tells a consistent shift (mean)
    from a minority of unusual clips (spread).
    """
    import numpy as np
    from scipy import linalg

    fad_dir = B_ROOT / "fad" / "poly"
    ref = np.load(fad_dir / "reference_encodec.npz", allow_pickle=False)["emb"]
    mu = ref.mean(0)
    real = np.linalg.norm(ref - mu, axis=1)
    out = {"real_median": float(np.median(real))}
    print("distance to the real mean: real violin + cello clips, median %.2f (n=%d)" % (out["real_median"], len(ref)))
    S_r = np.cov(ref, rowvar=False)
    for exp in (OLD, NEW):
        z = np.load(fad_dir / "generated" / ("%s_encodec.npz" % exp), allow_pickle=False)
        X, names = z["emb"], [str(n)[:-4] for n in z["names"]]
        d = np.linalg.norm(X - mu, axis=1)
        order = np.argsort(np.argsort(d))           # 0 = closest
        out[exp] = {"median": float(np.median(d)),
                    "clips": {n: (float(v), 100.0 * (len(d) - 1 - o) / (len(d) - 1))
                              for n, v, o in zip(names, d, order)}}
        S_g = np.cov(X, rowvar=False)
        mean_term = float(np.sum((X.mean(0) - mu) ** 2))
        cov_term = float(np.trace(S_r + S_g - 2 * linalg.sqrtm(S_r @ S_g).real))
        print("%-31s FAD %.2f = mean term %.2f + spread term %.2f | clip distance median %.2f, "
              "90th pct %.2f, max %.2f" % (exp, mean_term + cov_term, mean_term, cov_term,
                                           np.median(d), np.percentile(d, 90), d.max()))
    return out


def main():
    args = parse_args()
    dist = clip_distances()
    gen = {}
    for exp in (OLD, NEW):
        d = B_ROOT / "fad" / "poly" / "generated" / exp
        gen[exp] = {p.stem: p for p in sorted(d.glob("*.wav"))}
        if not gen[exp]:
            raise SystemExit("no clips in %s -- has the roll swap run?" % d)
    shared = sorted(set(gen[OLD]) & set(gen[NEW]))
    refs = sorted((B_ROOT / "fad" / "poly" / "reference").glob("*.wav"))
    inputs = {r["sample_id"]: r["input_audio"]
              for r in csv.DictReader(io.open(B_ROOT / "bridge_metrics_all.csv", encoding="utf-8"))
              if r["experiment"] == OLD}
    if len(shared) < args.n or not refs:
        raise SystemExit("%d shared clips, %d references -- not enough" % (len(shared), len(refs)))

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    audio_dir = out / "rollswap_listen" / "audio"
    audio_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
    build_id = "%s-seed%d" % (time.strftime("%Y%m%d%H%M%S"), args.seed)

    trials = []
    for i, sid in enumerate(rng.sample(shared, args.n), 1):
        new_first = rng.random() < 0.5
        a, b = (NEW, OLD) if new_first else (OLD, NEW)
        t = dict(id="t%d" % i, sample_id=sid, reference=rng.choice(refs), A=a, B=b)
        for slot, src in (("ref", t["reference"]), ("input", inputs[sid]), ("A", gen[a][sid]), ("B", gen[b][sid])):
            x, sr = load_mono(src)
            write_wav16(audio_dir / ("%s_%s.wav" % (t["id"], slot)), level_match(x)[0], sr)
        trials.append(t)
        print("%s  %s   A = %s (%.2f, closer than %.0f%% of its clips)   B = %s (%.2f, %.0f%%)"
              % (t["id"], sid, a, *dist[a]["clips"][sid], b, *dist[b]["clips"][sid]))

    with (out / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["build_id", "trial", "sample_id", "reference", "A", "B"])
        for t in trials:
            w.writerow([build_id, t["id"], t["sample_id"], t["reference"].stem, t["A"], t["B"]])

    style = re.search(r"<style>.*?</style>", BLIND_PAGE, re.S).group(0)
    page = PAGE.replace("__STYLE__", style).replace("__BUILD__", build_id).replace("__TRIALS__", json.dumps(
        [dict(id=t["id"], A=LABEL[t["A"]], B=LABEL[t["B"]], newSlot="A" if t["A"] == NEW else "B",
              dA=dist[t["A"]]["clips"][t["sample_id"]], dB=dist[t["B"]]["clips"][t["sample_id"]],
              medA=dist[t["A"]]["median"], medB=dist[t["B"]]["median"])
         for t in trials], ensure_ascii=False)).replace("__REAL_MEDIAN__", "%.2f" % dist["real_median"])
    (out / "rollswap_listen" / "index.html").write_text(page, encoding="utf-8")
    zip_path = shutil.make_archive(str(out / "rollswap_listen"), "zip", root_dir=out, base_dir="rollswap_listen")
    print("\n%d trials -> %s  (%.0f MB)" % (len(trials), zip_path, Path(zip_path).stat().st_size / 1e6))


PAGE = r"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>מבחן האזנה – הקלטה בלי MIDI</title>
__STYLE__
<style>
.reveal { margin-top:12px; padding:12px 14px; border-radius:8px; background:var(--soft); }
.reveal .hit { color:var(--done); font-weight:700; }
.score { font-size:1.25rem; font-weight:700; }
</style>
</head>
<body>
<main id="app"></main>
<script>
const BUILD = "__BUILD__";
const TRIALS = __TRIALS__;
const STORE = "rollswap-listen-" + BUILD;
let S = { answers: {}, pos: -1 };
try { const s = JSON.parse(localStorage.getItem(STORE)); if (s && s.answers) S = s; } catch (e) {}
function save() { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) {} }
const app = document.getElementById("app");
const answered = () => TRIALS.filter(t => (S.answers[t.id] || {}).choice).length;

function intro() {
  app.innerHTML = `
    <h1>מבחן האזנה – הקלטה בלי MIDI אמיתי</h1>
    <p class="sub">${TRIALS.length} ניסיונות · כ־10 דקות · התשובות נשמרות בדפדפן</p>
    <div class="card"><h2>מה בודקים</h2>
      <p>לשני המודלים נותנים את אותו קלט ואת אותו רול MIDI, שגונרט מהאודיו על ידי Basic Pitch, כמו שיהיה בהקלטה אמיתית.
         ההבדל הוא רק על מה הם אומנו. לפי ה־FAD, אחד מהם קרוב הרבה יותר לכינור וצ׳לו אמיתיים. כאן בודקים אם גם האוזן שומעת את זה.</p>
      <ol>
        <li><b>קטע ייחוס</b>: הקלטה אמיתית של כינור וצ׳לו, מיצירה אחרת. הוא שם רק כדי להזכיר איך הכלים נשמעים.</li>
        <li><b>הקלט</b>: חליל ובסון, הקטע שהומר.</li>
        <li><b>A</b> ו־<b>B</b>: אותו קלט אחרי ההמרה, בשני המודלים. הסדר אקראי.</li>
        <li>בחר את ההמרה שנשמעת יותר כמו כינור וצ׳לו אמיתיים, ואז אפשר ללחוץ על <b>הצג תשובה</b>.</li>
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

function trial() {
  if (S.pos >= TRIALS.length) return finish();
  const t = TRIALS[S.pos], a = S.answers[t.id] || {};
  const dline = (d, med) => `מרחק הקטע הזה: <b>${d[0].toFixed(2)}</b> · קרוב יותר מ־${Math.round(d[1])} אחוז מ־1,000 הקטעים של המודל (חציון המודל ${med.toFixed(2)})`;
  const reveal = a.revealed ? `<div class="reveal">
      <div><b>A</b>: ${t.A}</div><div class="muted">${dline(t.dA, t.medA)}</div>
      <div style="margin-top:6px"><b>B</b>: ${t.B}</div><div class="muted">${dline(t.dB, t.medB)}</div>
      <div style="margin-top:6px">${a.choice === t.newSlot
        ? '<span class="hit">בחרת במודל שאומן על MIDI אמיתי – כמו ה־FAD.</span>'
        : 'בחרת במודל שאומן על MIDI מגונרט – ההפך מה־FAD.'}</div>
      <div class="muted" style="margin-top:6px;font-size:.9rem">FAD קיים רק לאוסף של קטעים, לא לקטע בודד. "מרחק" הוא המרחק של הקטע
        מהממוצע של הקלטות כינור וצ׳לו אמיתיות, באותו מרחב של EnCodec שה־FAD משתמש בו. נמוך יותר = קרוב יותר.
        הקלטה אמיתית טיפוסית נמצאת במרחק __REAL_MEDIAN__.</div></div>` : "";
  app.innerHTML = `
    <div class="row"><b>ניסיון ${S.pos + 1} מתוך ${TRIALS.length}</b><span class="muted">נענו ${answered()}</span></div>
    <div class="progress"><div style="width:${100 * answered() / TRIALS.length}%"></div></div>
    <div class="card">
      ${player(t.id + "_ref.wav", "קטע ייחוס", "– כינור וצ׳לו אמיתיים, יצירה אחרת")}
      ${player(t.id + "_input.wav", "הקלט", "– חליל ובסון")}
      ${player(t.id + "_A.wav", "A", "")}
      ${player(t.id + "_B.wav", "B", "")}
      <p class="q">איזו המרה נשמעת יותר כמו כינור וצ׳לו אמיתיים?</p>
      <div class="choices">
        <button class="choice ${a.choice === "A" ? "on" : ""}" data-c="A" ${a.revealed ? "disabled" : ""}>A</button>
        <button class="choice ${a.choice === "B" ? "on" : ""}" data-c="B" ${a.revealed ? "disabled" : ""}>B</button>
      </div>
      <div class="conf">
        <button data-f="sure" class="${a.confidence === "sure" ? "on" : ""}" ${a.revealed ? "disabled" : ""}>בטוח</button>
        <button data-f="guess" class="${a.confidence === "guess" ? "on" : ""}" ${a.revealed ? "disabled" : ""}>מנחש</button>
      </div>
      <div class="foot">
        <button id="show" ${a.choice && !a.revealed ? "" : "disabled"}>הצג תשובה</button>
      </div>
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
  const done = TRIALS.filter(t => (S.answers[t.id] || {}).choice);
  const hits = done.filter(t => S.answers[t.id].choice === t.newSlot).length;
  app.innerHTML = `
    <h1>סיכום</h1>
    <div class="card">
      <p class="score">בחרת במודל שאומן על MIDI אמיתי ב־${hits} מתוך ${done.length} ניסיונות.</p>
      <p class="muted">לפי ה־FAD הוא קרוב יותר לכינור וצ׳לו אמיתיים (3.30 מול 6.13). חמישה ניסיונות לא מספיקים למובהקות סטטיסטית:
         גם 5 מתוך 5 יכול לקרות במקרה בסיכוי של כ־6%. זו בדיקה אם האוזן מסכימה עם המדד, לא הוכחה.</p>
      <div class="foot"><button class="primary" id="dl">הורד תשובות</button><button id="back">חזרה לניסיונות</button></div>
    </div>`;
  document.getElementById("dl").onclick = download;
  document.getElementById("back").onclick = () => { S.pos = 0; save(); trial(); };
}

function download() {
  const rows = [["build_id", "trial", "choice", "confidence", "chose_real_trained", "revealed", "answered_at"]];
  TRIALS.forEach(t => { const a = S.answers[t.id] || {};
    if (a.choice) rows.push([BUILD, t.id, a.choice, a.confidence || "", a.choice === t.newSlot ? 1 : 0,
                             a.revealed ? 1 : 0, a.at || ""]); });
  const csv = rows.map(r => r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(",")).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url; link.download = "rollswap_answers.csv";
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

if (S.pos < 0) intro(); else trial();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
