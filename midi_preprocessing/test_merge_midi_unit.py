"""Tests for merged MIDI. The pure ones need only numpy; the rest skip unless
pretty_midi (round trip) or torch + torchaudio (the dataset) are installed, so
the whole file runs on the server and the pure part runs anywhere.

    python midi_preprocessing/test_merge_midi_unit.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from merge_midi import merge_notes, or_of_rolls, rolls_equal  # noqa: E402

FPS = 75


class Skip(Exception):
    pass


def render(notes, fps=FPS):
    """pretty_midi's rendering rule: frames int(start*fs) up to int(end*fs)."""
    end = max((n[2] for n in notes), default=0.0)
    roll = np.zeros((128, int(fps * end)), dtype=np.uint8)
    for pitch, start, stop, _ in notes:
        roll[pitch, int(start * fps):int(stop * fps)] = 1
    return roll


# ----------------------------------------------------------------- pure logic
def test_overlapping_same_pitch_fuses_into_one_note():
    out = merge_notes([(60, 1.0, 2.0, 64), (60, 1.5, 3.0, 80)])
    assert out == [(60, 1.0, 3.0, 80)]


def test_note_inside_another_is_absorbed():
    assert merge_notes([(60, 1.0, 4.0, 64), (60, 2.0, 3.0, 64)]) == [(60, 1.0, 4.0, 64)]


def test_touching_notes_stay_two_so_a_reattack_is_kept():
    out = merge_notes([(60, 1.0, 2.0, 64), (60, 2.0, 3.0, 64)])
    assert out == [(60, 1.0, 2.0, 64), (60, 2.0, 3.0, 64)]


def test_different_pitches_never_fuse():
    out = merge_notes([(60, 1.0, 2.0, 64), (67, 1.0, 2.0, 64)])
    assert out == [(60, 1.0, 2.0, 64), (67, 1.0, 2.0, 64)]


def test_chain_of_overlaps_fuses_all_the_way():
    out = merge_notes([(60, 0.0, 1.0, 64), (60, 0.5, 1.5, 64), (60, 1.2, 2.0, 64)])
    assert out == [(60, 0.0, 2.0, 64)]


def test_zero_length_notes_are_dropped():
    assert merge_notes([(60, 1.0, 1.0, 64), (62, 1.0, 2.0, 64)]) == [(62, 1.0, 2.0, 64)]


def test_output_is_in_time_order():
    out = merge_notes([(72, 2.0, 3.0, 64), (48, 0.0, 1.0, 64), (60, 1.0, 2.0, 64)])
    assert [n[1] for n in out] == [0.0, 1.0, 2.0]


def test_merged_roll_is_the_or_of_the_inputs():
    """The property everything rests on, over random chorale-like material."""
    rng = np.random.default_rng(2026)
    for _ in range(200):
        parts = []
        for _ in range(2):
            notes, t = [], 0.0
            for _ in range(rng.integers(1, 30)):
                t += float(rng.uniform(0.0, 0.6))
                length = float(rng.uniform(0.01, 1.2))
                notes.append((int(rng.integers(55, 70)), t, t + length, 64))
            parts.append(notes)
        merged = merge_notes(parts[0] + parts[1])
        assert rolls_equal(render(merged), or_of_rolls([render(p) for p in parts]))


def test_or_of_rolls_pads_to_the_longest():
    a = np.zeros((128, 3), dtype=np.uint8); a[60, 0] = 1
    b = np.zeros((128, 5), dtype=np.uint8); b[67, 4] = 1
    out = or_of_rolls([a, b])
    assert out.shape == (128, 5) and out[60, 0] == 1 and out[67, 4] == 1


# ---------------------------------------------------- pretty_midi round trip
def test_written_file_renders_to_the_same_roll():
    try:
        import pretty_midi
    except ImportError:
        raise Skip("pretty_midi not installed")
    from merge_midi import merge_midi, render_roll

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for name, notes in (("1_flute", [(72, 0.0, 1.0), (74, 1.0, 2.5)]),
                            ("4_bassoon", [(48, 0.0, 2.0), (72, 0.5, 1.5)])):
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes = [pretty_midi.Note(velocity=64, pitch=p, start=s, end=e) for p, s, e in notes]
            pm.instruments.append(inst)
            path = tmp / (name + ".mid")
            pm.write(str(path))
            paths.append(path)

        merged, pms = merge_midi(paths)
        assert len(merged.instruments) == 1
        # the two 72s overlap (0.0-1.0 and 0.5-1.5) and must have fused
        assert sum(1 for n in merged.instruments[0].notes if n.pitch == 72) == 1

        roll = render_roll(merged, FPS, 0.0)
        assert rolls_equal(roll, or_of_rolls([render_roll(pm, FPS, 0.0) for pm in pms]))

        out = tmp / "merged.mid"
        merged.write(str(out))
        reread = render_roll(pretty_midi.PrettyMIDI(str(out)), FPS, 0.0)
        assert rolls_equal(roll, reread)


# ------------------------------------------------------------------ dataset
def _fake_track(root: Path, track: str, frames: int):
    import torch
    import torchaudio

    stems = root / track / "stems_audio"
    stems.mkdir(parents=True)
    for name in ("1_flute", "4_bassoon"):
        torchaudio.save(str(stems / (name + ".wav")), 0.1 * torch.randn(1, 24000 * 3), 24000)
    return frames


def test_dataset_serves_one_merged_roll_and_still_stacks_by_default():
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from audio_data_pytorch.datasets.wav_dataset import WAVDataset, merged_midi_name
    except ImportError:
        raise Skip("torch / torchaudio not installed")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data, cache = tmp / "data", tmp / "cache"
        _fake_track(data, "woodwind_track000001", 225)
        track_cache = cache / "woodwind_track000001"
        track_cache.mkdir(parents=True)
        merged = np.zeros((128, 225), dtype=np.uint8); merged[60, :] = 1
        np.save(track_cache / (merged_midi_name(["flute", "bassoon"]) + "_pianoroll.npy"), merged)
        for name, pitch in (("1_flute", 72), ("4_bassoon", 48)):
            r = np.zeros((128, 225), dtype=np.uint8); r[pitch, :] = 1
            np.save(track_cache / (name + "_pianoroll.npy"), r)

        common = dict(path=str(data), sample_rate=24000, random_crop_size=24000,
                      midi_cache_root=str(cache), mix_instruments=["flute", "bassoon"],
                      check_silence=False)

        roll = WAVDataset(midi_merged=True, **common)[0][3]
        assert tuple(roll.shape) == (128, 75), roll.shape
        assert float(roll[60].sum()) == 75 and float(roll.sum()) == 75

        stacked = WAVDataset(**common)[0][3]
        assert tuple(stacked.shape) == (256, 75), stacked.shape

        try:
            WAVDataset(path=str(data), sample_rate=24000, random_crop_size=24000,
                       midi_cache_root=str(cache), midi_merged=True)
        except ValueError:
            pass
        else:
            raise AssertionError("midi_merged without mix_instruments must be refused")


if __name__ == "__main__":
    passed = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok    %s" % name)
                passed += 1
            except Skip as why:
                print("skip  %s  (%s)" % (name, why))
                skipped += 1
    print("\n%d passed, %d skipped" % (passed, skipped))
