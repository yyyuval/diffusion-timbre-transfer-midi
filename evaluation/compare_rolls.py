"""How wrong is the generated (basic-pitch) merged roll, and is it wrong the
same way for both ensembles?

The generated-MIDI target model trained on rolls transcribed from violin +
cello, but at transfer time it reads rolls transcribed from flute + bassoon.
If basic-pitch errs differently on the two ensembles, the target model meets
an error pattern it never saw -- one candidate cause of its poor EnCodec-FAD at
sigma_max = 100. This compares, cell by cell, each track's generated merged
roll against its real merged roll (the dataset's MIDI, merged), over the first
1280 frames (17.07 s at 75 fps) -- the stretch the bridge reads at test time.

Extra cells (generated on, real off) are put in one bucket each, first match
wins, judged against the real notes sounding in that frame:
  octave     a real note 1-3 octaves away                  (+-12, 24, 36)
  harmonic   a real note this far BELOW it: the 3rd, 5th or 6th harmonic of
             that note (+19, +28, +31 semitones)
  neighbour  a real note 1 or 2 semitones away
  other      none of these

CPU only, reads the caches the models trained on.

  python evaluation/compare_rolls.py
  python evaluation/compare_rolls.py --wood-tracks evaluation/merged_test_lists/track_list_poly_merged.txt --n-string 1000
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FPS = 75
FRAMES = 1280
OCTAVE = (12, 24, 36)
HARMONIC = (19, 28, 31)
NEIGHBOUR = (1, 2)
DOMAINS = {
    "flute_bassoon": ("midi_cache_wood_fps75_gt_merged", "midi_cache_wood_fps75_bp_merged",
                      ["flute", "bassoon"]),
    "violin_cello": ("midi_cache_string_fps75_gt_merged", "midi_cache_string_fps75_bp_merged",
                     ["violin", "cello"]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wood-tracks", default="evaluation/merged_test_lists/track_list_poly_merged.txt",
                   help="flute + bassoon tracks: by default the bridge's own test inputs")
    p.add_argument("--string-tracks", default=None,
                   help="violin + cello tracks; default: --n-string drawn (seeded) from "
                        "every track both string caches hold")
    p.add_argument("--n-string", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", default="Outputs/WAV_files/merged_midi_test/roll_swap")
    return p.parse_args()


def read_list(path):
    lines = io.open(path, encoding="utf-8").read().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def load_roll(cache, track, name):
    path = Path(cache) / track / (name + "_pianoroll.npy")
    if not path.is_file():
        return None
    roll = np.load(path).astype(bool)[:, :FRAMES]
    if roll.shape[1] < FRAMES:
        roll = np.pad(roll, ((0, 0), (0, FRAMES - roll.shape[1])))
    return roll


def shifted(roll, k):
    """roll moved k pitch rows up (k > 0) or down, zero-filled."""
    out = np.zeros_like(roll)
    if k > 0:
        out[k:] = roll[:-k]
    elif k < 0:
        out[:k] = roll[-k:]
    else:
        out[:] = roll
    return out


def near(real, offsets, both_ways):
    hit = np.zeros_like(real)
    for k in offsets:
        hit |= shifted(real, k)
        if both_ways:
            hit |= shifted(real, -k)
    return hit


def onsets(roll):
    prev = np.concatenate([np.zeros((roll.shape[0], 1), bool), roll[:, :-1]], axis=1)
    return int((roll & ~prev).sum())


def compare(real, gen):
    extra = gen & ~real
    oct_ = extra & near(real, OCTAVE, True)
    harm = extra & ~oct_ & near(real, HARMONIC, False)   # real note k below -> shift up by k
    neigh = extra & ~oct_ & ~harm & near(real, NEIGHBOUR, True)
    sounding = real.any(0) | gen.any(0)
    return dict(
        tp=int((gen & real).sum()), fp=int(extra.sum()), fn=int((real & ~gen).sum()),
        fp_octave=int(oct_.sum()), fp_harmonic=int(harm.sum()), fp_neighbour=int(neigh.sum()),
        fp_other=int((extra & ~oct_ & ~harm & ~neigh).sum()),
        poly_real=float(real[:, sounding].sum(0).mean()) if sounding.any() else 0.0,
        poly_gen=float(gen[:, sounding].sum(0).mean()) if sounding.any() else 0.0,
        onsets_real=onsets(real), onsets_gen=onsets(gen),
    )


def summarise(domain, rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("domain", "track", "poly_real", "poly_gen")}
    p = s["tp"] / max(s["tp"] + s["fp"], 1)
    r = s["tp"] / max(s["tp"] + s["fn"], 1)
    secs = len(rows) * FRAMES / FPS
    return dict(
        domain=domain, tracks=len(rows),
        precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12),
        **{"fp_share_" + k: s["fp_" + k] / max(s["fp"], 1)
           for k in ("octave", "harmonic", "neighbour", "other")},
        polyphony_real=float(np.mean([x["poly_real"] for x in rows])),
        polyphony_gen=float(np.mean([x["poly_gen"] for x in rows])),
        onsets_per_s_real=s["onsets_real"] / secs, onsets_per_s_gen=s["onsets_gen"] / secs,
    )


def main():
    args = parse_args()
    from audio_data_pytorch.datasets.wav_dataset import merged_midi_name

    rng = np.random.default_rng(args.seed)
    track_lists = {"flute_bassoon": read_list(args.wood_tracks)}
    if args.string_tracks:
        track_lists["violin_cello"] = read_list(args.string_tracks)
    else:
        real_cache, gen_cache, _ = DOMAINS["violin_cello"]
        both = sorted({p.name for p in Path(real_cache).iterdir() if p.is_dir()}
                      & {p.name for p in Path(gen_cache).iterdir() if p.is_dir()})
        pick = rng.choice(len(both), min(args.n_string, len(both)), replace=False)
        track_lists["violin_cello"] = [both[i] for i in sorted(pick)]

    per_track, summary = [], []
    for domain, tracks in track_lists.items():
        real_cache, gen_cache, instruments = DOMAINS[domain]
        name = merged_midi_name(instruments)
        rows, missing = [], 0
        for track in tracks:
            real, gen = load_roll(real_cache, track, name), load_roll(gen_cache, track, name)
            if real is None or gen is None:
                missing += 1
                continue
            rows.append(dict(domain=domain, track=track, **compare(real, gen)))
        if missing:
            print("%s: %d of %d tracks lack a roll in one of the caches -- skipped"
                  % (domain, missing, len(tracks)))
        if not rows:
            raise SystemExit("%s: no track has both rolls" % domain)
        per_track += rows
        summary.append(summarise(domain, rows))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path, rows in ((out / "roll_comparison_tracks.csv", per_track),
                       (out / "roll_comparison.csv", summary)):
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("\ngenerated merged roll against the real one, first %d frames, cell by cell\n" % FRAMES)
    head = "%-14s %6s %6s %6s %6s | extra cells: %6s %8s %9s %6s | notes at once real/gen | onsets/s real/gen" % (
        "ensemble", "tracks", "P", "R", "F1", "octave", "harmonic", "neighbour", "other")
    print(head)
    print("-" * len(head))
    for s in summary:
        print("%-14s %6d %6.3f %6.3f %6.3f |              %6.3f %8.3f %9.3f %6.3f | %10.2f / %-10.2f | %6.2f / %.2f"
              % (s["domain"], s["tracks"], s["precision"], s["recall"], s["f1"],
                 s["fp_share_octave"], s["fp_share_harmonic"], s["fp_share_neighbour"],
                 s["fp_share_other"], s["polyphony_real"], s["polyphony_gen"],
                 s["onsets_per_s_real"], s["onsets_per_s_gen"]))
    print("\nextra-cell shares sum to 1 per row. -> %s" % out)


if __name__ == "__main__":
    main()
