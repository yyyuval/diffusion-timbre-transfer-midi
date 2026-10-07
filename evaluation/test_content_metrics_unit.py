"""Pure-function tests — no GPU, no dataset, no basic-pitch weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.content_metrics import (  # noqa: E402
    BASIC_PITCH_FPS,
    frames_to_multipitch,
    frames_to_note_events,
    midi_to_hz,
    score_transcription,
)


def test_frames_to_note_events_single_note():
    t = int(BASIC_PITCH_FPS * 1.0)  # 1 second of frames
    note = np.zeros((t, 88), dtype=np.float32)
    # MIDI 60 = C4 -> key index 60-21 = 39
    note[10:40, 39] = 0.9
    intervals, pitches = frames_to_note_events(note, threshold=0.5)
    assert intervals.shape == (1, 2)
    assert abs(pitches[0] - midi_to_hz(60)) < 1e-6
    assert abs(intervals[0, 0] - 10 / BASIC_PITCH_FPS) < 1e-6
    assert abs(intervals[0, 1] - 40 / BASIC_PITCH_FPS) < 1e-6


def test_polyphonic_two_notes():
    t = 50
    note = np.zeros((t, 88), dtype=np.float32)
    note[5:20, 39] = 0.9   # C4
    note[5:20, 43] = 0.9   # E4 simultaneous
    intervals, pitches = frames_to_note_events(note, threshold=0.5)
    assert intervals.shape[0] == 2
    assert set(np.round(pitches, 5)) == set(np.round([midi_to_hz(60), midi_to_hz(64)], 5))


def test_transcription_perfect_match():
    intervals = np.array([[0.1, 0.5], [0.6, 1.0]])
    pitches = np.array([midi_to_hz(60), midi_to_hz(64)])
    scores = score_transcription(intervals, pitches, intervals.copy(), pitches.copy())
    assert scores["f1"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0


def test_multipitch_shape():
    note = np.zeros((10, 88), dtype=np.float32)
    note[:, 39] = 0.8
    times, freqs = frames_to_multipitch(note, threshold=0.5)
    assert len(times) == 10
    assert all(len(f) == 1 for f in freqs)


if __name__ == "__main__":
    test_frames_to_note_events_single_note()
    test_polyphonic_two_notes()
    test_transcription_perfect_match()
    test_multipitch_shape()
    print("OK")
