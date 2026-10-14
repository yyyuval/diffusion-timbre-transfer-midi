"""Run the official basic-pitch command-line tool, and say why when it fails.

Used by the single-stem cache builders. Note that the official tool is NOT
installed in venv_yuval on the server; create_mixture_bp_midi_cache.py uses the
in-repo PyTorch port instead, which needs nothing installed. Two things this
fixes that the single-stem scripts got wrong:

The binary is looked up next to the Python that is running, not on PATH. Under
nohup, without the venv activated, a bare "basic-pitch" resolves to nothing or
to a different install, and the failure looked identical to "produced no MIDI".

The tool's own output is kept. It used to go to /dev/null, so every failure was
reported as "No MIDI file created" with the reason thrown away.

The existing single-stem generated caches came from this tool.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def find_basic_pitch(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    beside_python = Path(sys.executable).with_name("basic-pitch")
    if beside_python.exists():
        return str(beside_python)
    on_path = shutil.which("basic-pitch")
    if on_path:
        return on_path
    sys.exit(
        "basic-pitch was not found next to %s or on PATH. Install it into this "
        "venv (pip install basic-pitch) or pass --basic_pitch_bin." % sys.executable
    )


def run_basic_pitch(
    output_dir: Path,
    audio_paths: Sequence[Path],
    binary: Optional[str] = None,
    cpu_only: bool = False,
) -> subprocess.CompletedProcess:
    """One call for many files, so the model is loaded once.

    Writes <output_dir>/<audio stem>_basic_pitch.mid for each input. The tool
    refuses to overwrite, so callers pass only files whose output is absent.
    A failure on one file does not stop the others; check which outputs exist.

    cpu_only hides the GPUs from the tool. TensorFlow otherwise reserves the
    whole memory of every visible GPU, which takes them from training runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [find_basic_pitch(binary), str(output_dir)] + [str(p) for p in audio_paths]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def expected_output(output_dir: Path, audio_path: Path) -> Path:
    return Path(output_dir) / (Path(audio_path).stem + "_basic_pitch.mid")


def output_tail(proc: subprocess.CompletedProcess, lines: int = 25) -> str:
    text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip().splitlines()
    return "\n".join("    | " + line for line in text[-lines:])
