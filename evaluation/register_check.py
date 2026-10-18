"""Do the inputs and the FAD reference play in the same register?

The bridge keeps the input's notes, so every output plays a flute + bassoon
(or cello) part in violin + cello (or bassoon) timbre, while the FAD reference
is violin + cello (or bassoon) playing parts written for them. If the two
note distributions differ, even a perfect timbre transfer is some distance
from the reference -- for an embedding that is sensitive to register.

Reads the dataset's own MIDI for the first 17.07 s of each clip (what the
models and the FAD saw), duration-weighted. No audio, no GPU.

  venv_yuval/bin/python evaluation/register_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bridge_200 import (CASES, CLIP_LENGTH, DATASET_ROOT, SAMPLING_RATE,  # noqa: E402
                                   midi_path_for, stem_paths)

ROOTS = ["Outputs/WAV_files/clap_validation_1000", "Outputs/WAV_files/merged_midi_test"]
CLIP_SEC = CLIP_LENGTH / SAMPLING_RATE


def notes_in_clip(track, stems):
    """{stem: (pitches, weights)}: every note's pitch, weighted by its duration inside the clip."""
    import pretty_midi
    out = {}
    for audio in stem_paths(DATASET_ROOT, track, stems):
        pitches, weights = [], []
        for inst in pretty_midi.PrettyMIDI(midi_path_for(audio)).instruments:
            for n in inst.notes:
                dur = min(n.end, CLIP_SEC) - max(n.start, 0.0)
                if dur > 0:
                    pitches.append(n.pitch)
                    weights.append(dur)
        out[Path(audio).stem] = (np.array(pitches, float), np.array(weights, float))
    return out


def wquantile(x, w, q):
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    return float(x[order][np.searchsorted(cw, q)])


def name(p):
    import pretty_midi
    return pretty_midi.note_number_to_name(int(round(p)))


def summarise(tracks, stems):
    per_stem = {}
    clip_means = []
    for t in tracks:
        notes = notes_in_clip(t, stems)
        allp = np.concatenate([p for p, _ in notes.values()])
        allw = np.concatenate([w for _, w in notes.values()])
        if allw.sum() > 0:
            clip_means.append(np.average(allp, weights=allw))
        for s, (p, w) in notes.items():
            per_stem.setdefault(s, ([], []))
            per_stem[s][0].append(p)
            per_stem[s][1].append(w)
    pooled_p = np.concatenate([np.concatenate(v[0]) for v in per_stem.values()])
    pooled_w = np.concatenate([np.concatenate(v[1]) for v in per_stem.values()])
    stems_out = {s: (np.average(np.concatenate(v[0]), weights=np.concatenate(v[1])),
                     wquantile(np.concatenate(v[0]), np.concatenate(v[1]), 0.1),
                     wquantile(np.concatenate(v[0]), np.concatenate(v[1]), 0.9))
                 for s, v in per_stem.items()}
    return dict(n=len(clip_means), mean=float(np.average(pooled_p, weights=pooled_w)),
                p10=wquantile(pooled_p, pooled_w, 0.1), p50=wquantile(pooled_p, pooled_w, 0.5),
                p90=wquantile(pooled_p, pooled_w, 0.9), clip_sd=float(np.std(clip_means)), stems=stems_out)


def line(label, s):
    return ("  %-34s n=%4d  mean %5.1f (%-4s)  10%%/50%%/90%% %s / %s / %s  per-clip sd %.1f"
            % (label, s["n"], s["mean"], name(s["mean"]), name(s["p10"]), name(s["p50"]), name(s["p90"]), s["clip_sd"]))


def main():
    for root in ROOTS:
        for case in ("poly", "mono"):
            case_dir = Path(root) / "fad" / case
            if not (case_dir / "reference").is_dir():
                continue
            inputs = sorted({p.stem for p in (case_dir / "generated").glob("*/*.wav")})
            refs = sorted(p.stem for p in (case_dir / "reference").glob("*.wav"))
            c = CASES[case]
            si, sr = summarise(inputs, c["stems"]), summarise(refs, c["reference_stems"])
            print("\n" + "=" * 110)
            print("  %s / %s   (MIDI note numbers; 60 = C4, 12 = one octave)" % (Path(root).name, case))
            print("=" * 110)
            print(line("inputs  (%s)" % c["source"], si))
            print(line("reference (%s)" % c["reference_ensemble"], sr))
            print("  -> the inputs sit %+.1f semitones from the reference on average (median %+.0f)"
                  % (si["mean"] - sr["mean"], si["p50"] - sr["p50"]))
            for label, s in (("inputs", si), ("reference", sr)):
                for stem, (m, lo, hi) in s["stems"].items():
                    print("     %-9s %-10s mean %5.1f (%-4s)  10%%-90%% %s-%s" % (label, stem, m, name(m), name(lo), name(hi)))


if __name__ == "__main__":
    main()
