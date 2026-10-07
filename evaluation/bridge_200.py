"""Shared configuration and pure helpers for the 200-clip evaluation.

Imported by make_track_lists.py, run_bridge_batched.py and
prepare_fad_sets_200.py. Nothing here touches a GPU or loads a model, so it
can be unit-tested without the server.
"""

from __future__ import annotations

import csv
import glob
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_PATH = "/home/shared_workspace/diffusion-timbre-transfer"
DATASET_ROOT = (
    "/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/"
    "cocochorales_tiny_v1_zipped/main_dataset"
)
DEFAULT_OUT_ROOT = "Outputs/WAV_files/metrics_200"

# Must equal the constants in Run_timbre_transfer.py; the runner asserts it.
SAMPLING_RATE = 24000
CLIP_LENGTH = 409600
NUM_STEPS = 100
SIGMA_MIN = 0.001
RHO = 9.0
MIDI_MODE = "multiscale"
MIDI_FPS = 75

# A bridge direction: which stems go in, which statistics each model was
# trained with, and which real audio a FAD reference set is drawn from.
CASES = {
    "poly": dict(
        family="woodwind",
        source="flute_bassoon",
        target="violin_cello",
        stems=["1_flute.wav", "4_bassoon.wav"],
        midi_bins=256,
        source_mean="checkpoints/flute_bassoon_mean.pt",
        source_std="checkpoints/flute_bassoon_std.pt",
        target_mean="checkpoints/violin_cello_mean.pt",
        target_std="checkpoints/violin_cello_std.pt",
        reference_family="string",
        reference_stems=["1_violin.wav", "4_cello.wav"],
        reference_ensemble="violin_cello",
        track_list="evaluation/track_list_200_poly.txt",
        reference_script="Run_timbre_transfer.py",
    ),
    "mono": dict(
        family="string",
        source="cello",
        target="bassoon",
        stems=["4_cello.wav"],
        midi_bins=128,
        source_mean="checkpoints/mean_cello.pt",
        source_std="checkpoints/std_cello.pt",
        target_mean="checkpoints/mean_tensor_enc_bassoon.pt",
        target_std="checkpoints/std_tensor_enc_bassoon.pt",
        reference_family="woodwind",
        reference_stems=["4_bassoon.wav"],
        reference_ensemble="bassoon",
        track_list="evaluation/track_list_200_mono.txt",
        reference_script=None,
    ),
    # The direction Run_timbre_transfer_mono.py runs. Not part of the main
    # matrix; kept so --verify can check the single-stem path against a script.
    "mono_b2c": dict(
        family="woodwind",
        source="bassoon",
        target="cello",
        stems=["4_bassoon.wav"],
        midi_bins=128,
        source_mean="checkpoints/mean_tensor_enc_bassoon.pt",
        source_std="checkpoints/std_tensor_enc_bassoon.pt",
        target_mean="checkpoints/mean_cello.pt",
        target_std="checkpoints/std_cello.pt",
        reference_family="string",
        reference_stems=["4_cello.wav"],
        reference_ensemble="cello",
        track_list="evaluation_mono/track_list_20_mono.txt",
        reference_script="Run_timbre_transfer_mono.py",
    ),
}

# Checkpoint specs are either a path relative to the repo, or "run:<dir>",
# which resolves to the best epoch=*.ckpt under our_checkpoints/<dir>/ckpts/.
# The poly paths are copied verbatim from Run_timbre_transfer.py.
#
# A pair is one (source, target) set of weights. sigma_handoff does not change
# the weights, so the runner loads each pair once and sweeps the handoffs.
MODEL_PAIRS = {
    "no_midi": dict(
        case="poly", use_midi=False, trained_sigma=5,
        source_ckpt="our_checkpoints/flute_bassoon_nomidi_85ep/ckpts/epoch=83-valid_loss=0.592.ckpt",
        target_ckpt="our_checkpoints/violin_cello_nomidi_85ep/ckpts/epoch=80-valid_loss=0.637.ckpt",
    ),
    "original_midi": dict(
        case="poly", use_midi=True, trained_sigma=5,
        source_ckpt="our_checkpoints/flute_bassoon_gt_ms_85ep/ckpts/epoch=83-valid_loss=0.586.ckpt",
        target_ckpt="our_checkpoints/violin_cello_gt_ms_85ep/ckpts/epoch=80-valid_loss=0.632.ckpt",
    ),
    "no_midi_s100": dict(
        case="poly", use_midi=False, trained_sigma=100,
        source_ckpt="our_checkpoints/flute_bassoon_nomidi_85ep_sigma100/ckpts/epoch=74-valid_loss=0.667.ckpt",
        target_ckpt="our_checkpoints/violin_cello_nomidi_85ep_sigma100/ckpts/epoch=82-valid_loss=0.699.ckpt",
    ),
    "original_midi_s100": dict(
        case="poly", use_midi=True, trained_sigma=100,
        source_ckpt="our_checkpoints/flute_bassoon_gt_ms_85ep_sigma100/ckpts/epoch=79-valid_loss=0.593.ckpt",
        target_ckpt="our_checkpoints/violin_cello_gt_ms_85ep_sigma100/ckpts/epoch=82-valid_loss=0.612.ckpt",
    ),
    "mono_c2b_no_midi": dict(
        case="mono", use_midi=False, trained_sigma=5,
        source_ckpt="run:cello_nomidi_85ep",
        target_ckpt="run:bassoon_nomidi_85ep",
    ),
    "mono_c2b_original_midi": dict(
        case="mono", use_midi=True, trained_sigma=5,
        source_ckpt="run:cello_gt_ms_85ep",
        target_ckpt="run:bassoon_gt_ms_85ep",
    ),
    "mono_c2b_no_midi_s100": dict(
        case="mono", use_midi=False, trained_sigma=100,
        source_ckpt="run:cello_nomidi_85ep_sigma100",
        target_ckpt="run:bassoon_nomidi_85ep_sigma100",
    ),
    "mono_c2b_original_midi_s100": dict(
        case="mono", use_midi=True, trained_sigma=100,
        source_ckpt="run:cello_gt_ms_85ep_sigma100",
        target_ckpt="run:bassoon_gt_ms_85ep_sigma100",
    ),
    "mono_no_midi": dict(
        case="mono_b2c", use_midi=False, trained_sigma=5,
        source_ckpt="run:bassoon_nomidi_85ep",
        target_ckpt="run:cello_nomidi_85ep",
    ),
    "mono_original_midi": dict(
        case="mono_b2c", use_midi=True, trained_sigma=5,
        source_ckpt="run:bassoon_gt_ms_85ep",
        target_ckpt="run:cello_gt_ms_85ep",
    ),
    "mono_no_midi_s100": dict(
        case="mono_b2c", use_midi=False, trained_sigma=100,
        source_ckpt="run:bassoon_nomidi_85ep_sigma100",
        target_ckpt="run:cello_nomidi_85ep_sigma100",
    ),
    "mono_original_midi_s100": dict(
        case="mono_b2c", use_midi=True, trained_sigma=100,
        source_ckpt="run:bassoon_gt_ms_85ep_sigma100",
        target_ckpt="run:cello_gt_ms_85ep_sigma100",
    ),
}

DEFAULT_PAIRS = [
    "original_midi_s100", "no_midi_s100",
    "mono_c2b_original_midi_s100", "mono_c2b_no_midi_s100",
]
BASELINE_PAIRS = [
    "original_midi", "no_midi",
    "mono_c2b_original_midi", "mono_c2b_no_midi",
]
DEFAULT_HANDOFFS = [5.0, 25.0, 50.0, 75.0, 100.0]

CSV_FIELDS = [
    "timestamp", "experiment", "pair", "case", "sample_id", "source", "target",
    "output_audio", "input_audio", "DPD", "JD", "num_steps", "use_midi",
    "midi_mode", "midi_bins", "trained_sigma", "sigma_handoff",
    "source_ckpt", "target_ckpt", "input_stems", "batch_size",
]


def experiment_name(pair: str, trained_sigma: float, sigma_handoff: float) -> str:
    """Same suffix rule as Run_timbre_transfer.py, so names line up across CSVs."""
    if sigma_handoff > trained_sigma:
        raise ValueError(
            "sigma_handoff=%g is above trained sigma_max=%g. These weights were "
            "never trained at that noise level." % (sigma_handoff, trained_sigma))
    if sigma_handoff != trained_sigma:
        return "%s_h%g" % (pair, sigma_handoff)
    return pair


def build_jobs(pairs: Sequence[str], handoffs: Sequence[float]) -> List[tuple]:
    """[(pair, [handoffs])]. Handoffs above a pair's trained sigma are dropped
    for that pair, so --baseline pairs (sigma 5) only get the values they can run."""
    jobs = []
    for pair in pairs:
        if pair not in MODEL_PAIRS:
            raise KeyError("unknown pair %r. Known: %s" % (pair, sorted(MODEL_PAIRS)))
        trained = MODEL_PAIRS[pair]["trained_sigma"]
        usable = [float(h) for h in handoffs if float(h) <= trained]
        if not usable:
            usable = [float(trained)]
        jobs.append((pair, usable))
    return jobs


def parse_job_specs(specs: Sequence[str]) -> List[tuple]:
    """'original_midi_s100:5,25,50' -> ('original_midi_s100', [5.0, 25.0, 50.0])."""
    jobs = []
    for spec in specs:
        pair, _, hs = spec.partition(":")
        if pair not in MODEL_PAIRS:
            raise KeyError("unknown pair %r. Known: %s" % (pair, sorted(MODEL_PAIRS)))
        handoffs = [float(h) for h in hs.split(",") if h] if hs else [float(MODEL_PAIRS[pair]["trained_sigma"])]
        for h in handoffs:
            experiment_name(pair, MODEL_PAIRS[pair]["trained_sigma"], h)
        jobs.append((pair, handoffs))
    return jobs


def format_job_spec(pair: str, handoffs: Sequence[float]) -> str:
    return "%s:%s" % (pair, ",".join("%g" % h for h in handoffs))


def split_jobs(jobs: Sequence[tuple], n_workers: int) -> List[List[tuple]]:
    """Split (pair, handoff) units into n contiguous, near-equal shares.

    Every handoff costs the same (NUM_STEPS is fixed), so equal counts mean equal
    time. Contiguity keeps each share to as few pairs as possible, and every
    pair it touches is loaded once.
    """
    units = [(pair, h) for pair, hs in jobs for h in hs]
    n_workers = max(1, min(n_workers, len(units)))
    base, extra = divmod(len(units), n_workers)
    shares, start = [], 0
    for i in range(n_workers):
        size = base + (1 if i < extra else 0)
        share = []
        for pair, h in units[start:start + size]:
            if share and share[-1][0] == pair:
                share[-1][1].append(h)
            else:
                share.append((pair, [h]))
        shares.append(share)
        start += size
    return shares


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


_VALID_LOSS = re.compile(r"valid_loss=([0-9.]+?)(?:-v\d+)?\.ckpt$")


def resolve_ckpt(spec: str, repo_path: str = REPO_PATH) -> str:
    """A repo-relative path, or "run:<dir>" -> lowest valid_loss epoch=*.ckpt in it."""
    if not spec.startswith("run:"):
        path = spec if os.path.isabs(spec) else os.path.join(repo_path, spec)
        if not os.path.isfile(path):
            raise FileNotFoundError("checkpoint not found: %s" % path)
        return path
    run_dir = spec[len("run:"):]
    pattern = os.path.join(repo_path, "our_checkpoints", run_dir, "ckpts", "epoch=*.ckpt")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            "no epoch=*.ckpt under our_checkpoints/%s/ckpts/ -- has it finished "
            "training? Pass --source-ckpt/--target-ckpt to point at a file." % run_dir)

    def loss(path):
        m = _VALID_LOSS.search(os.path.basename(path))
        return float(m.group(1).rstrip(".")) if m else float("inf")

    return min(hits, key=lambda p: (loss(p), p))


def stem_paths(dataset_root: str, track: str, stems: Sequence[str]) -> List[str]:
    return [os.path.join(dataset_root, track, "stems_audio", s) for s in stems]


def midi_path_for(audio_path: str) -> str:
    """stems_audio/4_cello.wav -> stems_midi/4_cello.mid (Run_timbre_transfer.derive_midi_paths)."""
    stems_dir, wav_name = os.path.split(audio_path)
    track_dir = os.path.dirname(stems_dir)
    return os.path.join(track_dir, "stems_midi", os.path.splitext(wav_name)[0] + ".mid")


def track_has_files(dataset_root: str, track: str, stems: Sequence[str]) -> bool:
    for audio in stem_paths(dataset_root, track, stems):
        midi = midi_path_for(audio)
        for path in (audio, midi):
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                return False
    return True


def list_family_tracks(dataset_root: str, family: str) -> List[str]:
    prefix = family + "_track"
    with os.scandir(dataset_root) as it:
        return sorted(e.name for e in it if e.is_dir() and e.name.startswith(prefix))


def pick_tracks(
    candidates: Sequence[str],
    n: int,
    seed: int,
    accept=None,
    exclude: Iterable[str] = (),
) -> List[str]:
    """Seeded shuffle of the sorted candidates, keep the first n that pass `accept`.

    Sorting first makes the result independent of directory listing order.
    Returned sorted, so the list reads naturally and diffs cleanly.
    """
    excluded = set(exclude)
    pool = sorted(c for c in candidates if c not in excluded)
    random.Random(seed).shuffle(pool)
    picked = []
    for track in pool:
        if accept is None or accept(track):
            picked.append(track)
            if len(picked) == n:
                break
    if len(picked) < n:
        raise RuntimeError("only %d usable tracks, %d requested" % (len(picked), n))
    return sorted(picked)


def read_track_list(path) -> List[str]:
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def load_input_waveform(audio_paths: Sequence[str]):
    """Run_timbre_transfer.py's input preprocessing, line for line, on CPU.

    Load each stem, resample to 24 kHz, sum, rescale to 0.95 only if there were
    several stems and the sum clips, then crop [:CLIP_LENGTH] from the start or
    zero-pad. Returns [1, C, CLIP_LENGTH], the shape the script feeds EnCodec.
    """
    import torch
    import torchaudio

    waveform_raw = None
    for audio_path in audio_paths:
        stem, orig_sr = torchaudio.load(audio_path)
        if orig_sr != SAMPLING_RATE:
            stem = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=SAMPLING_RATE
            )(stem)
        waveform_raw = stem if waveform_raw is None else waveform_raw + stem

    if len(audio_paths) > 1:
        peak = waveform_raw.abs().max()
        if peak > 0.95:
            waveform_raw = 0.95 * waveform_raw / peak

    waveform = waveform_raw.unsqueeze(0)
    if waveform.shape[-1] >= CLIP_LENGTH:
        return waveform[..., :CLIP_LENGTH]
    return torch.nn.functional.pad(waveform, (0, CLIP_LENGTH - waveform.shape[-1]))


def output_paths(out_root, experiment: str, sample_id: str, source: str, target: str):
    """(output, input) wav paths; the output name follows Run_timbre_transfer's archive pattern."""
    audio_dir = Path(out_root) / "audio"
    prefix = "%s_%s_" % (safe_name(experiment), sample_id)
    out = audio_dir / (prefix + "generated_%s_from_%s.wav" % (target, source))
    inp = audio_dir / (prefix + "input_%s.wav" % source)
    return out, inp


def condition_csv(out_root, experiment: str) -> Path:
    """One CSV per experiment, so parallel processes never share a file."""
    return Path(out_root) / "csv" / ("%s.csv" % safe_name(experiment))


def read_done(csv_path) -> set:
    done = set()
    path = Path(csv_path)
    if not path.is_file():
        return done
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["experiment"], row["sample_id"]))
    return done


def append_row(csv_path, row: Dict, fieldnames: Sequence[str] = CSV_FIELDS) -> None:
    """Append one row; refuse to write under a header that does not match."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        with path.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != list(fieldnames):
            raise RuntimeError(
                "%s has header %s, expected %s. Move it aside rather than mixing "
                "column layouts." % (path, header, list(fieldnames)))
        write_header = False
    else:
        write_header = True
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def merge_csvs(out_root, merged_path: Optional[str] = None) -> Path:
    """Concatenate every per-experiment CSV, last row wins on (experiment, sample_id)."""
    csv_dir = Path(out_root) / "csv"
    merged = Path(merged_path) if merged_path else Path(out_root) / "bridge_metrics_all.csv"
    rows = {}
    for path in sorted(csv_dir.glob("*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows[(row["experiment"], row["sample_id"])] = row
    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({k: rows[key].get(k, "") for k in CSV_FIELDS})
    return merged


def chunks(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield list(items[i:i + size])
