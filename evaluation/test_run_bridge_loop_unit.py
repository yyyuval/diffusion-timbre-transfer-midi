"""The runner's job loop with the models faked out: batching, handoff sweep,
resume, and which output lands under which (experiment, sample_id).

  python -m pytest evaluation/test_run_bridge_loop_unit.py -q
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import bridge_200 as b  # noqa: E402
from evaluation import run_bridge_batched as r  # noqa: E402


class FakeBridge:
    loads, transfers = [], []

    def __init__(self, pair, source_ckpt, target_ckpt, device, rtt):
        FakeBridge.loads.append(pair)
        self.case = b.CASES[b.MODEL_PAIRS[pair]["case"]]

    def prepare(self, tracks, dataset_root):
        x = torch.stack([torch.full((1, 4), float(int(t[-3:]))) for t in tracks])
        return x, None, [b.stem_paths(dataset_root, t, self.case["stems"]) for t in tracks]

    def encode(self, x):
        return x

    def transfer(self, emb, midi, handoff):
        FakeBridge.transfers.append((handoff, emb[:, 0, 0].tolist()))
        return emb * 1000 + handoff


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import torchaudio

    saved = {}

    def fake_save(path, tensor, sr):
        Path(path).write_bytes(b"wav")
        saved[str(path)] = tensor.clone()

    monkeypatch.setattr(torchaudio, "save", fake_save)
    monkeypatch.setattr(r, "Bridge", FakeBridge)
    monkeypatch.setattr(r, "import_reference_module", lambda: None)
    FakeBridge.loads.clear()
    FakeBridge.transfers.clear()

    for d in ("flute_bassoon_gt_ms_85ep_sigma100", "violin_cello_gt_ms_85ep_sigma100"):
        ck = tmp_path / "our_checkpoints" / d / "ckpts"
        ck.mkdir(parents=True)
    (tmp_path / "our_checkpoints/flute_bassoon_gt_ms_85ep_sigma100/ckpts/epoch=79-valid_loss=0.593.ckpt").write_bytes(b"")
    (tmp_path / "our_checkpoints/violin_cello_gt_ms_85ep_sigma100/ckpts/epoch=82-valid_loss=0.612.ckpt").write_bytes(b"")
    (tmp_path / "evaluation").mkdir()
    tracks = ["woodwind_track%06d" % i for i in range(1, 8)]
    (tmp_path / "evaluation" / "track_list_200_poly.txt").write_text("\n".join(tracks) + "\n")

    args = argparse.Namespace(
        repo_path=str(tmp_path), out_root="out", dataset_root="/data", batch_size=3, n=None,
        no_dpd=True, source_ckpt=None, target_ckpt=None,
        track_list_poly="evaluation/track_list_200_poly.txt",
        track_list_mono="evaluation/track_list_200_mono.txt",
        track_list_mono_b2c="evaluation_mono/track_list_20_mono.txt",
    )
    return args, tracks, saved


def rows_of(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_sweep_loads_pair_once_and_writes_every_condition(setup, monkeypatch):
    args, tracks, saved = setup
    monkeypatch.chdir(args.repo_path)
    r.run_jobs(args, [("original_midi_s100", [25.0, 100.0])])

    assert FakeBridge.loads == ["original_midi_s100"]
    assert len(FakeBridge.transfers) == 2 * 3  # 7 tracks at batch 3 -> 3 batches x 2 handoffs
    for h, exp in ((25.0, "original_midi_s100_h25"), (100.0, "original_midi_s100")):
        rows = rows_of(Path(args.repo_path) / "out" / "csv" / ("%s.csv" % exp))
        assert [x["sample_id"] for x in rows] == tracks
        for row in rows:
            assert float(row["sigma_handoff"]) == h and row["trained_sigma"] == "100"
            assert row["use_midi"] == "True" and row["case"] == "poly"
            assert row["input_stems"].endswith("/stems_audio/4_bassoon.wav")
            assert row["source_ckpt"].endswith("epoch=79-valid_loss=0.593.ckpt")
            out = saved[str(Path(row["output_audio"]))]
            assert float(out[0, 0]) == int(row["sample_id"][-3:]) * 1000 + h
            assert Path(args.repo_path, row["input_audio"]).exists()
            assert Path(row["input_audio"]).name.startswith(exp + "_" + row["sample_id"] + "_input_")


def test_resume_skips_done_clips_per_handoff(setup, monkeypatch):
    args, tracks, saved = setup
    monkeypatch.chdir(args.repo_path)
    done_csv = b.condition_csv(Path(args.repo_path) / "out", "original_midi_s100_h25")
    for t in tracks[:4]:
        b.append_row(done_csv, {"experiment": "original_midi_s100_h25", "sample_id": t})
    r.run_jobs(args, [("original_midi_s100", [25.0, 100.0])])

    h25 = [ids for h, ids in FakeBridge.transfers if h == 25.0]
    assert sorted(i for ids in h25 for i in ids) == [5.0, 6.0, 7.0]
    h100 = [ids for h, ids in FakeBridge.transfers if h == 100.0]
    assert sorted(i for ids in h100 for i in ids) == [float(i) for i in range(1, 8)]
    assert len(rows_of(done_csv)) == 7

    FakeBridge.loads.clear()
    r.run_jobs(args, [("original_midi_s100", [25.0, 100.0])])
    assert FakeBridge.loads == [], "nothing left to do, so the models must not be loaded"


def test_split_on_oom_halves_and_reassembles():
    calls = []

    def fn(x, extra):
        calls.append(x.shape[0])
        if x.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("fake")
        return x * 2 + extra

    x = torch.arange(5.0).view(5, 1)
    out = r.split_on_oom(fn, x, torch.ones(5, 1))
    assert torch.equal(out, x * 2 + 1)
    assert calls == [5, 2, 3, 1, 2]
