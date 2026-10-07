"""Polyphonic content metrics for timbre-transfer evaluation.

Primary:  mir_eval.transcription  (note events, 50 ms onset tolerance)
Secondary: mir_eval.multipitch    (frame-aligned, no time tolerance)

Reference side is always the dataset's ground-truth MIDI (stems_midi/),
never a transcription of the input. Estimate side is basic-pitch on audio,
using the raw [T, 88] matrix (no octave fold, no argmax).

Audio fed to basic-pitch is resampled to 22050 Hz here. Do not change
pitch_tracking_utils.py — its 24 kHz quirks cancel for DPD/JD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pretty_midi
import torch
import torchaudio

# basic-pitch constants
BASIC_PITCH_SR = 22050
BASIC_PITCH_HOP = 256
BASIC_PITCH_FPS = BASIC_PITCH_SR / BASIC_PITCH_HOP  # ≈ 86.13
MIDI_NOTE_OFFSET = 21  # A0
N_PIANO_KEYS = 88

PathLike = Union[str, Path]


def midi_to_hz(midi_note: float) -> float:
    return float(440.0 * (2.0 ** ((midi_note - 69.0) / 12.0)))


def load_wav_mono(path: PathLike, target_sr: int) -> torch.Tensor:
    """Load mono float waveform at target_sr. Shape [1, N]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def crop_or_pad(wav: torch.Tensor, n_samples: int) -> torch.Tensor:
    if wav.shape[-1] >= n_samples:
        return wav[..., :n_samples]
    pad = n_samples - wav.shape[-1]
    return torch.nn.functional.pad(wav, (0, pad))


def sum_stems(
    paths: Sequence[PathLike],
    target_sr: int = 24000,
    clip_length: Optional[int] = 409600,
) -> torch.Tensor:
    """Replicate Run_timbre_transfer mixture loading (no AddFifth).

    Sum stems at 24 kHz, peak-normalise to 0.95 only when >1 stem and peak>0.95,
    then crop/pad from the start to clip_length.
    """
    waveform = None
    for path in paths:
        stem = load_wav_mono(path, target_sr)
        waveform = stem if waveform is None else waveform + stem
    assert waveform is not None

    if len(paths) > 1:
        peak = float(waveform.abs().max())
        if peak > 0.95:
            waveform = 0.95 * waveform / peak

    if clip_length is not None:
        waveform = crop_or_pad(waveform, clip_length)
    return waveform


def midi_paths_from_audio_paths(audio_paths: Sequence[PathLike]) -> List[Path]:
    """stems_audio/4_cello.wav -> stems_midi/4_cello.mid"""
    out = []
    for p in audio_paths:
        p = Path(p)
        midi = p.parent.parent / "stems_midi" / (p.stem + ".mid")
        if not midi.is_file():
            raise FileNotFoundError("Missing ground-truth MIDI: %s" % midi)
        out.append(midi)
    return out


def notes_from_midi_files(
    midi_paths: Sequence[PathLike],
    max_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge note events from one or more MIDI files.

    Returns
    -------
    intervals : [N, 2] float64  (onset, offset) seconds
    pitches_hz : [N] float64
    """
    intervals: List[List[float]] = []
    pitches_hz: List[float] = []

    for path in midi_paths:
        pm = pretty_midi.PrettyMIDI(str(path))
        for inst in pm.instruments:
            for note in inst.notes:
                onset = float(note.start)
                offset = float(note.end)
                if max_time is not None:
                    if onset >= max_time:
                        continue
                    offset = min(offset, max_time)
                if offset <= onset:
                    continue
                intervals.append([onset, offset])
                pitches_hz.append(midi_to_hz(note.pitch))

    if not intervals:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(intervals, dtype=np.float64), np.asarray(pitches_hz, dtype=np.float64)


def frames_to_note_events(
    note_prob: np.ndarray,
    threshold: float = 0.5,
    fps: float = BASIC_PITCH_FPS,
    midi_offset: int = MIDI_NOTE_OFFSET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert [T, 88] note probabilities to mir_eval note events.

    Contiguous runs above threshold become a single note. Offset is the end
    of the last active frame; mir_eval.transcription is typically called with
    offset_ratio=None so endings are ignored.
    """
    if note_prob.ndim != 2 or note_prob.shape[1] != N_PIANO_KEYS:
        raise ValueError("expected note_prob shape [T, 88], got %s" % (note_prob.shape,))

    active = note_prob >= threshold
    intervals: List[List[float]] = []
    pitches_hz: List[float] = []

    for key in range(N_PIANO_KEYS):
        col = active[:, key]
        if not col.any():
            continue
        # pad edges so runs are closed
        padded = np.concatenate([[False], col, [False]])
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        hz = midi_to_hz(midi_offset + key)
        for s, e in zip(starts, ends):
            onset = s / fps
            offset = e / fps
            if offset <= onset:
                offset = onset + (1.0 / fps)
            intervals.append([onset, offset])
            pitches_hz.append(hz)

    if not intervals:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(intervals, dtype=np.float64), np.asarray(pitches_hz, dtype=np.float64)


def frames_to_multipitch(
    note_prob: np.ndarray,
    threshold: float = 0.5,
    fps: float = BASIC_PITCH_FPS,
    midi_offset: int = MIDI_NOTE_OFFSET,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Frame-wise multipitch lists for mir_eval.multipitch."""
    t_frames = note_prob.shape[0]
    times = np.arange(t_frames, dtype=np.float64) / fps
    freqs: List[np.ndarray] = []
    for t in range(t_frames):
        keys = np.where(note_prob[t] >= threshold)[0]
        if keys.size == 0:
            freqs.append(np.zeros((0,), dtype=np.float64))
        else:
            freqs.append(np.asarray([midi_to_hz(midi_offset + int(k)) for k in keys], dtype=np.float64))
    return times, freqs


def midi_to_multipitch(
    midi_paths: Sequence[PathLike],
    times: np.ndarray,
    max_time: Optional[float] = None,
) -> List[np.ndarray]:
    """Sample ground-truth MIDI at the given frame times (Hz lists)."""
    intervals, pitches_hz = notes_from_midi_files(midi_paths, max_time=max_time)
    out: List[np.ndarray] = []
    for t in times:
        if intervals.shape[0] == 0:
            out.append(np.zeros((0,), dtype=np.float64))
            continue
        mask = (intervals[:, 0] <= t) & (t < intervals[:, 1])
        out.append(pitches_hz[mask].astype(np.float64))
    return out


def transcribe_wav(
    wav_24k_or_any: torch.Tensor,
    tracker,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Resample to 22050 and run basic-pitch. Returns [T, 88] probabilities.

    `tracker` is an audio_diffusion_pytorch.PitchTracker instance (or any
    object with `.pt_model`).
    """
    if wav_24k_or_any.dim() == 1:
        wav = wav_24k_or_any.unsqueeze(0)
    elif wav_24k_or_any.dim() == 2:
        wav = wav_24k_or_any
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    elif wav_24k_or_any.dim() == 3:
        wav = wav_24k_or_any.squeeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    else:
        raise ValueError("unexpected wav shape %s" % (tuple(wav_24k_or_any.shape),))

    # Assume input is 24 kHz (repo default). Callers with other rates should
    # resample before calling, or pass already-22050 audio via resample below
    # only when numel matches a 24 kHz clip length heuristic — safer to always
    # resample from an explicit source_sr. We treat the tensor as 24 kHz.
    wav_22k = torchaudio.functional.resample(wav, 24000, BASIC_PITCH_SR)
    wav_22k = wav_22k.unsqueeze(0).cpu()  # [1, 1, N]; basic-pitch stays on CPU

    with torch.no_grad():
        out = tracker.pt_model(wav_22k)
    note = out["note"][0].detach().cpu().numpy()
    if note.shape[-1] != N_PIANO_KEYS:
        raise RuntimeError("basic-pitch note dim is %s, expected 88" % (note.shape,))
    return note.astype(np.float32)


def score_transcription(
    ref_intervals: np.ndarray,
    ref_pitches_hz: np.ndarray,
    est_intervals: np.ndarray,
    est_pitches_hz: np.ndarray,
    onset_tolerance: float = 0.05,
    pitch_tolerance: float = 50.0,
) -> dict:
    """mir_eval.transcription with offsets ignored (offset_ratio=None)."""
    import mir_eval

    if ref_intervals.shape[0] == 0 and est_intervals.shape[0] == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "avg_overlap_ratio": 0.0}
    if ref_intervals.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "avg_overlap_ratio": 0.0}
    if est_intervals.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "avg_overlap_ratio": 0.0}

    p, r, f, ao = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches_hz,
        est_intervals,
        est_pitches_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=None,
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "avg_overlap_ratio": float(ao),
    }


def score_multipitch(
    ref_times: np.ndarray,
    ref_freqs: List[np.ndarray],
    est_times: np.ndarray,
    est_freqs: List[np.ndarray],
) -> dict:
    import mir_eval

    scores = mir_eval.multipitch.metrics(ref_times, ref_freqs, est_times, est_freqs)
    # mir_eval returns (Precision, Recall, Accuracy) plus chroma variants etc.
    # Unpack defensively by length.
    precision = float(scores[0])
    recall = float(scores[1])
    accuracy = float(scores[2])
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": float(f1),
    }


def score_audio_against_midi(
    wav_24k: torch.Tensor,
    midi_paths: Sequence[PathLike],
    tracker,
    note_threshold: float = 0.5,
    max_time: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Full content score: transcription + multipitch for one clip."""
    if max_time is None:
        max_time = float(wav_24k.shape[-1]) / 24000.0

    note_prob = transcribe_wav(wav_24k, tracker, device=device)
    est_intervals, est_pitches = frames_to_note_events(note_prob, threshold=note_threshold)
    ref_intervals, ref_pitches = notes_from_midi_files(midi_paths, max_time=max_time)

    tr = score_transcription(ref_intervals, ref_pitches, est_intervals, est_pitches)

    est_times, est_mp = frames_to_multipitch(note_prob, threshold=note_threshold)
    # Align multipitch to the same time grid as the estimate
    ref_mp = midi_to_multipitch(midi_paths, est_times, max_time=max_time)
    mp = score_multipitch(est_times, ref_mp, est_times, est_mp)

    return {
        "transcription_precision": tr["precision"],
        "transcription_recall": tr["recall"],
        "transcription_f1": tr["f1"],
        "multipitch_precision": mp["precision"],
        "multipitch_recall": mp["recall"],
        "multipitch_f1": mp["f1"],
        "multipitch_accuracy": mp["accuracy"],
        "n_ref_notes": int(ref_intervals.shape[0]),
        "n_est_notes": int(est_intervals.shape[0]),
        "note_threshold": float(note_threshold),
    }


# Ensemble name -> stem filenames used by the poly bridge configs
ENSEMBLE_STEMS = {
    "flute_bassoon": ["1_flute.wav", "4_bassoon.wav"],
    "violin_cello": ["1_violin.wav", "4_cello.wav"],
    "flute": ["1_flute.wav"],
    "bassoon": ["4_bassoon.wav"],
    "violin": ["1_violin.wav"],
    "cello": ["4_cello.wav"],
}


def resolve_stem_paths(dataset_root: PathLike, track_name: str, ensemble: str) -> List[Path]:
    stems = ENSEMBLE_STEMS.get(ensemble)
    if stems is None:
        raise KeyError(
            "Unknown ensemble %r. Known: %s" % (ensemble, sorted(ENSEMBLE_STEMS))
        )
    root = Path(dataset_root) / track_name / "stems_audio"
    paths = [root / s for s in stems]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing stems for %s/%s: %s" % (track_name, ensemble, missing))
    return paths
