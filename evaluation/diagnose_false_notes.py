"""What kind of note is the extra note?

Precision sits well below recall everywhere -- 0.548 against 0.784 for the
best poly condition. The usual reading is that the model invents notes, but
the target-side ceilings show basic-pitch inventing them too: on real bassoon
it scores P 0.660 against R 0.969. So the question is not how many false
positives there are, it is whether the generated audio produces a DIFFERENT
KIND of false positive from real audio, and the fix depends on the answer.

Every unmatched estimated note is put in one bucket:

  octave    a reference note is sounding at that moment one or more octaves
            away. The transcriber locked onto a harmonic. Nothing is wrong
            with the audio's pitch content; this is transcriber behaviour and
            it should appear at the ceiling too.
  reattack  the SAME pitch is already sounding and this onset falls inside it.
            One held note read as two. Caused by amplitude or timbre wobble
            within a note -- a generation artifact, and the one worth fixing.
  late      the same pitch exists nearby in time but outside the tolerance.
            Onset drift, which the tolerance sweep already showed dominates.
  spurious  none of the above. A note that is simply not there.

Unmatched reference notes split the same way into `merged` (the pitch was
sounding, this onset was not found -- two notes read as one) and `missed`.

Reads the note events straight out of compute_onset_sensitivity.py's cache,
using its exact key, so a clip already transcribed costs nothing and no GPU
is touched. Clips not in the cache are transcribed unless --cached-only.

  python evaluation/diagnose_false_notes.py --cached-only
  python evaluation/diagnose_false_notes.py --limit 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_TAG = "events-v1-22050"  # must match compute_onset_sensitivity.py


def hz_to_midi(hz):
    hz = np.asarray(hz, dtype=float)
    out = np.full(hz.shape, -1.0)
    good = hz > 0
    out[good] = 69.0 + 12.0 * np.log2(hz[good] / 440.0)
    return np.rint(out).astype(int)


def cache_key(path, note_threshold):
    stat = path.stat()
    identity = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, note_threshold, CACHE_TAG]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class Events:
    """Note events for one wav, from the shared cache where possible."""

    def __init__(self, cache_dir, note_threshold, cached_only):
        self.cache = Path(cache_dir)
        self.thr = note_threshold
        self.cached_only = cached_only
        self.tracker = None
        self.hits = self.misses = 0

    def get(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        dest = self.cache / (cache_key(path, self.thr) + ".npz")
        if dest.exists():
            self.hits += 1
            with np.load(dest, allow_pickle=False) as a:
                return a["intervals"], a["pitches"], float(a["duration"])
        self.misses += 1
        if self.cached_only:
            return None
        import torchaudio
        from evaluation.content_metrics import frames_to_note_events, transcribe_wav
        from audio_diffusion_pytorch import PitchTracker
        if self.tracker is None:
            self.tracker = PitchTracker()
        wav, sr = torchaudio.load(str(path))
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        duration = wav.shape[-1] / 24000
        intervals, pitches = frames_to_note_events(transcribe_wav(wav, self.tracker),
                                                   threshold=self.thr)
        self.cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, intervals=intervals, pitches=pitches, duration=duration)
        return intervals, pitches, duration


def classify(ref_iv, ref_pitch, est_iv, est_pitch, tol, late_tol):
    """Greedy onset match, then a bucket for everything left over.

    Matching mirrors mir_eval.transcription with offset_ratio=None: a pair is
    eligible when the onsets fall within `tol` and the pitches round to the
    same semitone, and each note is used once. Greedy by onset distance is
    not the maximum matching mir_eval computes, so the counts here can differ
    from its F1 by a note or two; the shape of the error is what this is for.
    """
    ref_m, est_m = hz_to_midi(ref_pitch), hz_to_midi(est_pitch)
    ref_on, est_on = np.asarray(ref_iv)[:, 0], np.asarray(est_iv)[:, 0]
    ref_off = np.asarray(ref_iv)[:, 1]

    pairs = []
    for i in range(len(est_on)):
        for j in range(len(ref_on)):
            if est_m[i] == ref_m[j]:
                d = abs(est_on[i] - ref_on[j])
                if d <= tol:
                    pairs.append((d, i, j))
    pairs.sort()
    used_e, used_r = set(), set()
    for _, i, j in pairs:
        if i not in used_e and j not in used_r:
            used_e.add(i)
            used_r.add(j)

    matched_ref = np.zeros(len(ref_on), dtype=bool)
    matched_ref[list(used_r)] = True

    fp = Counter()
    for i in range(len(est_on)):
        if i in used_e:
            continue
        t, p = est_on[i], est_m[i]
        sounding = (ref_on <= t) & (t <= ref_off)
        same = sounding & (ref_m == p)
        # A second onset inside a note the transcriber already found is a
        # re-attack. The same onset inside a note nothing matched is that
        # note, heard late -- the two are identical in the audio and only the
        # bookkeeping tells them apart.
        if np.any(same & matched_ref):
            fp["reattack"] += 1
        elif np.any(same):
            fp["late"] += 1
        elif np.any(sounding & (np.abs(ref_m - p) % 12 == 0) & (ref_m != p)):
            fp["octave"] += 1
        elif np.any((ref_m == p) & (np.abs(ref_on - t) <= late_tol)):
            fp["late"] += 1
        else:
            fp["spurious"] += 1

    fn = Counter()
    for j in range(len(ref_on)):
        if j in used_r:
            continue
        t, p = ref_on[j], ref_m[j]
        if np.any((est_m == p) & (np.asarray(est_iv)[:, 0] <= t) & (t <= np.asarray(est_iv)[:, 1])):
            fn["merged"] += 1
        else:
            fn["missed"] += 1

    return len(used_e), fp, fn, len(est_on), len(ref_on)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("Outputs/WAV_files/pitch_validation_200"))
    p.add_argument("--bridge-csv", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--note-threshold", type=float, default=0.5)
    p.add_argument("--onset-tolerance", type=float, default=0.05)
    p.add_argument("--late-tolerance", type=float, default=0.15,
                   help="an unmatched note of the right pitch within this is onset drift")
    p.add_argument("--limit", type=int, default=None, help="rows of the bridge CSV to read")
    p.add_argument("--cached-only", action="store_true",
                   help="never transcribe; skip clips the cache does not hold")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    bridge = args.bridge_csv or args.root / "bridge_metrics_all.csv"
    if not bridge.is_file():
        raise SystemExit("no %s" % bridge)
    cache = args.cache or args.root / "onset_sensitivity" / "cache"
    events = Events(cache, args.note_threshold, args.cached_only)

    from evaluation.content_metrics import midi_paths_from_audio_paths, notes_from_midi_files

    rows = list(csv.DictReader(io.open(bridge, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    agg = defaultdict(lambda: dict(tp=0, fp=Counter(), fn=Counter(), est=0, ref=0, clips=0))
    seen_inputs = set()
    for n, row in enumerate(rows, 1):
        midi = midi_paths_from_audio_paths(row["input_stems"].split(";"))
        jobs = [("generated", row["experiment"], row["output_audio"])]
        key = (row["case"], row["sample_id"])
        if key not in seen_inputs:
            jobs.append(("real_input", "ceiling_" + row["case"], row["input_audio"]))
            seen_inputs.add(key)
        for kind, exp, path in jobs:
            got = events.get(path)
            if got is None:
                continue
            est_iv, est_pitch, duration = got
            ref_iv, ref_pitch = notes_from_midi_files(midi, max_time=duration)
            if len(ref_iv) == 0 or len(est_iv) == 0:
                continue
            tp, fp, fn, n_est, n_ref = classify(
                ref_iv, ref_pitch, est_iv, est_pitch, args.onset_tolerance, args.late_tolerance)
            a = agg[(kind, exp)]
            a["tp"] += tp
            a["fp"] += fp
            a["fn"] += fn
            a["est"] += n_est
            a["ref"] += n_ref
            a["clips"] += 1
        if n % 200 == 0:
            print("  %d/%d rows (cache %d hit, %d miss)" % (n, len(rows), events.hits, events.misses),
                  flush=True)

    if not agg:
        raise SystemExit("nothing scored. With --cached-only this means the cache is empty -- "
                         "run compute_onset_sensitivity.py first, or drop the flag.")

    print("\ncache: %d hits, %d misses" % (events.hits, events.misses))
    print("\nfalse positives as a share of all estimated notes "
          "(onset tolerance %.0f ms)\n" % (args.onset_tolerance * 1000))
    head = "%-34s %6s %7s %8s %8s %8s %8s %8s %8s" % (
        "condition", "clips", "P", "octave", "reattack", "late", "spurious", "merged", "missed")
    print(head)
    print("-" * len(head))
    out_rows = []
    for (kind, exp), a in sorted(agg.items(), key=lambda kv: (kv[0][0] != "real_input", kv[0][1])):
        est, ref = max(a["est"], 1), max(a["ref"], 1)
        rec = dict(kind=kind, experiment=exp, clips=a["clips"], n_est=a["est"], n_ref=a["ref"],
                   precision=a["tp"] / est, recall=a["tp"] / ref,
                   **{"fp_" + k: a["fp"][k] / est for k in ("octave", "reattack", "late", "spurious")},
                   **{"fn_" + k: a["fn"][k] / ref for k in ("merged", "missed")})
        out_rows.append(rec)
        print("%-34s %6d %7.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
              % (exp, a["clips"], rec["precision"], rec["fp_octave"], rec["fp_reattack"],
                 rec["fp_late"], rec["fp_spurious"], rec["fn_merged"], rec["fn_missed"]))

    print("\nRead a generated row against its ceiling_ row, not against zero. A column that")
    print("matches the ceiling is the transcriber; a column well above it is the model.")

    out = args.out_csv or args.root / "false_note_diagnosis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("\n-> %s" % out)


if __name__ == "__main__":
    main()
