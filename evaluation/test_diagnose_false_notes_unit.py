"""Pure-function tests — no GPU, no dataset, no basic-pitch weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.diagnose_false_notes import classify, hz_to_midi  # noqa: E402

TOL, LATE = 0.05, 0.15


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def one_ref():
    """A single reference note, MIDI 60, sounding from 1.0 s to 2.0 s."""
    return np.array([[1.0, 2.0]]), np.array([hz(60)])


def run(est_iv, est_pitch):
    ref_iv, ref_pitch = one_ref()
    return classify(ref_iv, ref_pitch, np.array(est_iv), np.array(est_pitch), TOL, LATE)


def test_hz_to_midi_roundtrip():
    assert list(hz_to_midi([hz(60), hz(72), hz(21)])) == [60, 72, 21]
    assert hz_to_midi([0.0])[0] == -1


def test_exact_match_has_no_errors():
    tp, fp, fn, _, _ = run([[1.0, 2.0]], [hz(60)])
    assert (tp, dict(fp), dict(fn)) == (1, {}, {})


def test_second_onset_inside_a_matched_note_is_a_reattack():
    tp, fp, fn, _, _ = run([[1.0, 1.4], [1.5, 2.0]], [hz(60), hz(60)])
    assert tp == 1 and fp["reattack"] == 1 and not fn


def test_harmonic_an_octave_up_is_an_octave_error():
    tp, fp, fn, _, _ = run([[1.0, 2.0], [1.2, 2.0]], [hz(60), hz(72)])
    assert tp == 1 and fp["octave"] == 1 and not fn


def test_late_onset_inside_an_unmatched_note_is_late_not_reattack():
    """The audio is identical to a re-attack; only the unmatched reference
    note distinguishes them, which is why matching has to run first."""
    tp, fp, fn, _, _ = run([[1.1, 2.0]], [hz(60)])
    assert tp == 0 and fp["reattack"] == 0 and fp["late"] == 1 and fn["missed"] == 1


def test_unrelated_note_is_spurious():
    tp, fp, fn, _, _ = run([[1.0, 2.0], [5.0, 5.5]], [hz(60), hz(67)])
    assert tp == 1 and fp["spurious"] == 1 and not fn


def test_wrong_pitch_elsewhere_is_spurious_and_the_reference_is_missed():
    tp, fp, fn, _, _ = run([[5.0, 5.5]], [hz(67)])
    assert tp == 0 and fp["spurious"] == 1 and fn["missed"] == 1


def test_one_estimate_cannot_claim_two_reference_notes():
    ref_iv = np.array([[1.0, 1.4], [1.5, 2.0]])
    ref_pitch = np.array([hz(60), hz(60)])
    tp, fp, fn, _, _ = classify(ref_iv, ref_pitch, np.array([[1.0, 2.0]]),
                                np.array([hz(60)]), TOL, LATE)
    assert tp == 1 and fn["merged"] == 1


if __name__ == "__main__":
    passed = 0
    for name, fn_ in sorted(globals().items()):
        if name.startswith("test_") and callable(fn_):
            fn_()
            print("ok  %s" % name)
            passed += 1
    print("\n%d passed" % passed)
