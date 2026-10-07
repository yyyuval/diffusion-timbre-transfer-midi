"""Pure-function tests for the 200-clip pipeline -- no GPU, no dataset, no checkpoints.

  python -m pytest evaluation/test_bridge_200_unit.py -q
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import bridge_200 as b  # noqa: E402


# ---------------------------------------------------------------- naming / jobs

def test_experiment_name_follows_run_timbre_transfer():
    assert b.experiment_name("original_midi_s100", 100, 100.0) == "original_midi_s100"
    assert b.experiment_name("original_midi_s100", 100, 25.0) == "original_midi_s100_h25"
    assert b.experiment_name("no_midi_s100", 100, 5) == "no_midi_s100_h5"
    with pytest.raises(ValueError):
        b.experiment_name("original_midi", 5, 25.0)


def test_default_grid_is_20_conditions():
    jobs = b.build_jobs(b.DEFAULT_PAIRS, b.DEFAULT_HANDOFFS)
    units = [(p, h) for p, hs in jobs for h in hs]
    assert len(units) == 20
    assert all(b.MODEL_PAIRS[p]["trained_sigma"] == 100 for p, _ in units)
    assert {b.MODEL_PAIRS[p]["case"] for p, _ in units} == {"poly", "mono"}


def test_baseline_pairs_only_get_runnable_handoffs():
    jobs = dict(b.build_jobs(b.BASELINE_PAIRS, b.DEFAULT_HANDOFFS))
    assert all(hs == [5.0] for hs in jobs.values())


def test_job_spec_roundtrip():
    spec = b.format_job_spec("mono_c2b_no_midi_s100", [5.0, 25.0, 100.0])
    assert spec == "mono_c2b_no_midi_s100:5,25,100"
    assert b.parse_job_specs([spec]) == [("mono_c2b_no_midi_s100", [5.0, 25.0, 100.0])]
    assert b.parse_job_specs(["original_midi"]) == [("original_midi", [5.0])]
    with pytest.raises(ValueError):
        b.parse_job_specs(["original_midi:25"])
    with pytest.raises(KeyError):
        b.parse_job_specs(["nope:5"])


@pytest.mark.parametrize("n_gpus", [1, 2, 3, 4, 5, 8, 30])
def test_split_covers_every_unit_once(n_gpus):
    jobs = b.build_jobs(b.DEFAULT_PAIRS, b.DEFAULT_HANDOFFS)
    shares = b.split_jobs(jobs, n_gpus)
    units = sorted((p, h) for share in shares for p, hs in share for h in hs)
    assert units == sorted((p, h) for p, hs in jobs for h in hs)
    sizes = [sum(len(hs) for _, hs in share) for share in shares]
    assert max(sizes) - min(sizes) <= 1
    for share in shares:
        pairs = [p for p, _ in share]
        assert len(pairs) == len(set(pairs)), "a pair would be loaded twice in one process"


def test_split_four_gpus_is_one_pair_each():
    shares = b.split_jobs(b.build_jobs(b.DEFAULT_PAIRS, b.DEFAULT_HANDOFFS), 4)
    assert [len(s) for s in shares] == [1, 1, 1, 1]
    assert all(len(s[0][1]) == 5 for s in shares)


# ---------------------------------------------------------------- checkpoints

def test_resolve_ckpt(tmp_path):
    ck = tmp_path / "our_checkpoints" / "cello_nomidi_85ep_sigma100" / "ckpts"
    ck.mkdir(parents=True)
    for name in ("epoch=9-valid_loss=0.701.ckpt", "epoch=77-valid_loss=0.655.ckpt",
                 "epoch=80-valid_loss=0.690-v1.ckpt", "last.ckpt"):
        (ck / name).write_bytes(b"x")
    got = b.resolve_ckpt("run:cello_nomidi_85ep_sigma100", str(tmp_path))
    assert got.endswith("epoch=77-valid_loss=0.655.ckpt")
    assert b.resolve_ckpt("our_checkpoints/cello_nomidi_85ep_sigma100/ckpts/last.ckpt",
                          str(tmp_path)).endswith("last.ckpt")
    with pytest.raises(FileNotFoundError):
        b.resolve_ckpt("run:bassoon_nomidi_85ep_sigma100", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        b.resolve_ckpt("our_checkpoints/missing.ckpt", str(tmp_path))


# ---------------------------------------------------------------- fake dataset

def write_midi(path, notes):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for start, end, pitch in notes:
        inst.notes.append(pretty_midi.Note(velocity=64, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def make_track(root, name, stems, midi_notes=((0.5, 1.0, 60),), skip_midi=(), empty_audio=()):
    for stem in stems:
        wav = root / name / "stems_audio" / (stem + ".wav")
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"" if stem in empty_audio else b"RIFF")
        if stem not in skip_midi:
            write_midi(root / name / "stems_midi" / (stem + ".mid"), midi_notes)


@pytest.fixture
def fake_dataset(tmp_path):
    root = tmp_path / "main_dataset"
    for i in range(12):
        make_track(root, "woodwind_track%06d" % i, ["1_flute", "2_oboe", "3_clarinet", "4_bassoon"])
        make_track(root, "string_track%06d" % i, ["1_violin", "2_violin", "3_viola", "4_cello"])
    make_track(root, "woodwind_track900001", ["1_flute", "4_bassoon"], skip_midi=("4_bassoon",))
    make_track(root, "woodwind_track900002", ["1_flute", "4_bassoon"], empty_audio=("1_flute",))
    make_track(root, "woodwind_track900003", ["1_flute"])
    make_track(root, "woodwind_track900004", ["1_flute", "4_bassoon"], midi_notes=((30.0, 31.0, 60),))
    make_track(root, "string_track900001", ["1_violin", "2_violin", "3_viola", "4_cello"],
               midi_notes=())
    make_track(root, "brass_track000001", ["1_trumpet"])
    (root / "train").mkdir()
    return root


def test_track_filtering(fake_dataset):
    root = str(fake_dataset)
    woodwind = b.list_family_tracks(root, "woodwind")
    assert len(woodwind) == 16 and all(t.startswith("woodwind_track") for t in woodwind)
    stems = b.CASES["poly"]["stems"]
    assert b.track_has_files(root, "woodwind_track000003", stems)
    assert not b.track_has_files(root, "woodwind_track900001", stems)
    assert not b.track_has_files(root, "woodwind_track900002", stems)
    assert not b.track_has_files(root, "woodwind_track900003", stems)


def test_pick_tracks_is_deterministic_and_respects_exclude():
    cands = ["t%03d" % i for i in range(50)]
    a = b.pick_tracks(cands, 10, seed=7)
    assert a == b.pick_tracks(list(reversed(cands)), 10, seed=7)
    assert a == sorted(a) and len(set(a)) == 10
    assert a != b.pick_tracks(cands, 10, seed=8)
    ex = b.pick_tracks(cands, 10, seed=7, exclude=a)
    assert not set(ex) & set(a)
    odd = b.pick_tracks(cands, 5, seed=7, accept=lambda t: int(t[1:]) % 2 == 1)
    assert all(int(t[1:]) % 2 == 1 for t in odd)
    with pytest.raises(RuntimeError):
        b.pick_tracks(cands, 51, seed=7)


def test_make_track_lists_cli(fake_dataset, tmp_path):
    out = tmp_path / "lists"
    cmd = [sys.executable, str(ROOT / "evaluation" / "make_track_lists.py"),
           "--dataset-root", str(fake_dataset), "--n", "12", "--out-dir", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    poly = b.read_track_list(out / "track_list_12_poly.txt")
    mono = b.read_track_list(out / "track_list_12_mono.txt")
    assert poly == ["woodwind_track%06d" % i for i in range(12)]
    assert mono == ["string_track%06d" % i for i in range(12)]
    subprocess.run(cmd, check=True, capture_output=True)
    assert b.read_track_list(out / "track_list_12_poly.txt") == poly
    res = subprocess.run(cmd[:-4] + ["--n", "13", "--out-dir", str(out)], capture_output=True)
    assert res.returncode != 0, "only 12 usable tracks per family; 13 must fail"


# ---------------------------------------------------------------- preprocessing

@pytest.fixture
def fake_audio(monkeypatch):
    store = {}
    import torchaudio

    monkeypatch.setattr(torchaudio, "load", lambda path, *a, **k: (store[str(path)].clone(), 16000))
    return store


def reference_preprocess(stems):
    """Run_timbre_transfer.py main(), lines 357-388, transcribed independently."""
    import torchaudio

    waveform_raw = None
    for s in stems:
        stem = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(s)
        waveform_raw = stem if waveform_raw is None else waveform_raw + stem
    if len(stems) > 1:
        peak = waveform_raw.abs().max()
        if peak > 0.95:
            waveform_raw = 0.95 * waveform_raw / peak
    w = waveform_raw.unsqueeze(0)
    if w.shape[-1] >= b.CLIP_LENGTH:
        return w[..., :b.CLIP_LENGTH]
    return torch.nn.functional.pad(w, (0, b.CLIP_LENGTH - w.shape[-1]))


def test_preprocess_mixture_that_clips(fake_audio):
    g = torch.Generator().manual_seed(0)
    n16 = int(32 * 16000)
    fake_audio["a"] = 0.8 * torch.rand(1, n16, generator=g) - 0.4 + 0.5
    fake_audio["b"] = 0.8 * torch.rand(1, n16, generator=g) - 0.4 + 0.5
    out = b.load_input_waveform(["a", "b"])
    assert out.shape == (1, 1, b.CLIP_LENGTH)
    assert torch.equal(out, reference_preprocess([fake_audio["a"], fake_audio["b"]]))
    full = torch.cat([fake_audio["a"], fake_audio["b"]]).sum(0)
    assert out.abs().max() <= 0.95 + 1e-6 and full.abs().max() > 0.95


def test_preprocess_single_stem_is_never_rescaled(fake_audio):
    fake_audio["loud"] = torch.full((1, 16000 * 30), 0.99)
    out = b.load_input_waveform(["loud"])
    assert torch.equal(out, reference_preprocess([fake_audio["loud"]]))
    assert out.abs().max() > 0.95


def test_preprocess_quiet_mixture_untouched_and_short_clip_padded(fake_audio):
    fake_audio["q1"] = torch.full((1, 16000 * 5), 0.1)
    fake_audio["q2"] = torch.full((1, 16000 * 5), 0.2)
    out = b.load_input_waveform(["q1", "q2"])
    assert out.shape == (1, 1, b.CLIP_LENGTH)
    assert torch.equal(out, reference_preprocess([fake_audio["q1"], fake_audio["q2"]]))
    assert torch.all(out[..., 24000 * 5 + 100:] == 0)


def test_preprocess_matches_content_metrics_sum_stems(fake_audio):
    from evaluation.content_metrics import sum_stems

    g = torch.Generator().manual_seed(1)
    fake_audio["x"] = torch.randn(1, 16000 * 20, generator=g) * 0.6
    fake_audio["y"] = torch.randn(1, 16000 * 20, generator=g) * 0.6
    ours = b.load_input_waveform(["x", "y"]).squeeze(0)
    theirs = sum_stems(["x", "y"], target_sr=24000, clip_length=b.CLIP_LENGTH)
    assert torch.allclose(ours, theirs, atol=1e-6)


# ---------------------------------------------------------------- CSV

def test_csv_resume_and_header_guard(tmp_path):
    path = b.condition_csv(tmp_path, "original_midi_s100_h25")
    assert path.name == "original_midi_s100_h25.csv"
    assert b.read_done(path) == set()
    b.append_row(path, {"experiment": "original_midi_s100_h25", "sample_id": "t1", "sigma_handoff": 25.0})
    b.append_row(path, {"experiment": "original_midi_s100_h25", "sample_id": "t2"})
    assert b.read_done(path) == {("original_midi_s100_h25", "t1"), ("original_midi_s100_h25", "t2")}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == b.CSV_FIELDS and rows[0]["sigma_handoff"] == "25.0"

    bad = tmp_path / "csv" / "bad.csv"
    bad.write_text("timestamp,experiment,sample_id\n1,x,y\n")
    with pytest.raises(RuntimeError):
        b.append_row(bad, {"experiment": "x", "sample_id": "z"})


def test_merge_last_row_wins(tmp_path):
    b.append_row(b.condition_csv(tmp_path, "e1"), {"experiment": "e1", "sample_id": "t1", "DPD": 1})
    b.append_row(b.condition_csv(tmp_path, "e1"), {"experiment": "e1", "sample_id": "t1", "DPD": 2})
    b.append_row(b.condition_csv(tmp_path, "e2"), {"experiment": "e2", "sample_id": "t1", "DPD": 3})
    merged = b.merge_csvs(tmp_path)
    with merged.open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["experiment"], r["DPD"]) for r in rows] == [("e1", "2"), ("e2", "3")]


def test_output_paths_prefix():
    out, inp = b.output_paths("root", "original_midi_s100_h25", "woodwind_track000001",
                              "flute_bassoon", "violin_cello")
    assert out == Path("root/audio/original_midi_s100_h25_woodwind_track000001_"
                       "generated_violin_cello_from_flute_bassoon.wav")
    assert inp.name == "original_midi_s100_h25_woodwind_track000001_input_flute_bassoon.wav"


def test_midi_path_for_matches_derive_rule():
    assert b.midi_path_for("/d/t/stems_audio/4_cello.wav") == "/d/t/stems_midi/4_cello.mid"


# ---------------------------------------------------------------- summary

def test_summary_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    out_root = tmp_path / "m200"
    rows, content = [], []
    for case, pair_midi, pair_nomidi in (("poly", "original_midi_s100", "no_midi_s100"),
                                         ("mono", "mono_c2b_original_midi_s100", "mono_c2b_no_midi_s100")):
        for pair, use_midi in ((pair_midi, True), (pair_nomidi, False)):
            for h in b.DEFAULT_HANDOFFS:
                exp = b.experiment_name(pair, 100, h)
                for i in range(30):
                    dpd = rng.normal(2 + h / 50 - (0.5 if use_midi else 0), 0.3)
                    rows.append({"experiment": exp, "pair": pair, "case": case,
                                 "sample_id": "t%02d" % i, "DPD": dpd, "JD": rng.uniform(0, .3),
                                 "use_midi": use_midi, "trained_sigma": 100, "sigma_handoff": h})
                    if case == "poly":
                        f1 = float(np.clip(rng.normal(0.6 - h / 400 + (0.1 if use_midi else 0), .05), 0, 1))
                        content.append({"experiment": exp, "sample_id": "t%02d" % i,
                                        "transcription_f1": f1, "transcription_precision": f1,
                                        "transcription_recall": f1, "multipitch_f1": f1})
    for r in rows:
        b.append_row(b.condition_csv(out_root, r["experiment"]), r)
    b.merge_csvs(out_root)
    (out_root / "content").mkdir()
    import pandas as pd
    pd.DataFrame(content).to_csv(out_root / "content" / "all.csv", index=False)
    (out_root / "ceiling").mkdir()
    pd.DataFrame({"transcription_f1": [0.8, 0.7], "transcription_precision": [.8, .7],
                  "transcription_recall": [.8, .7]}).to_csv(out_root / "ceiling" / "ceiling_poly.csv")
    (out_root / "fad").mkdir()
    exps = sorted({r["experiment"] for r in rows})
    pd.DataFrame({"experiment": exps, "clap_fad": rng.uniform(1, 5, len(exps)),
                  "n_reference": 200, "n_generated": 30}).to_csv(out_root / "fad" / "clap_fad_200.csv")

    env = dict(__import__("os").environ, MPLCONFIGDIR=str(tmp_path / "mpl"))
    res = subprocess.run([sys.executable, str(ROOT / "evaluation" / "summarize_200.py"),
                          "--out-root", str(out_root)], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    table = pd.read_csv(out_root / "summary" / "summary_200.csv")
    assert len(table) == 20 and table["n"].eq(30).all()
    assert table["clap_fad"].notna().all()
    paired = pd.read_csv(out_root / "summary" / "paired_midi_vs_nomidi_200.csv")
    f1 = paired[(paired["case"] == "poly") & (paired["metric"] == "transcription_f1")]
    assert len(f1) == 5 and (f1["delta_midi_minus_no_midi"] > 0).all()
    dpd = paired[(paired["case"] == "mono") & (paired["metric"] == "DPD")]
    assert len(dpd) == 5 and (dpd["delta_midi_minus_no_midi"] < 0).all()
    assert (dpd["midi_better_on"] > 15).all()
    pngs = sorted(p.name for p in (out_root / "summary" / "plots").glob("*.png"))
    assert "poly_transcription_f1_vs_handoff.png" in pngs
    assert "mono_DPD_vs_handoff.png" in pngs and "mono_clap_fad_vs_handoff.png" in pngs
    assert "MIDI is better" in res.stdout or "original_midi_s100" in res.stdout
