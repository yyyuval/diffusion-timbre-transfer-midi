"""Batched timbre-transfer bridge for the 200-clip evaluation.

Same bridge as Run_timbre_transfer.py, but each model pair (+ EnCodec) is loaded
ONCE, clips go through in batches, and every sigma_handoff is swept off the
same loaded weights and the same encoded batch. Input preprocessing and the MIDI
rolls come from the same code (bridge_200.load_input_waveform replicates the
script's lines; derive_midi_paths / load_piano_roll are imported from it).

Run --verify first: one clip through Run_timbre_transfer.py and through this
runner, outputs compared.

  python -u evaluation/run_bridge_batched.py --verify --gpu 0
  python -u evaluation/run_bridge_batched.py --gpu 0 --jobs original_midi_s100:5,25,50,75,100
  python -u evaluation/run_bridge_batched.py --gpu 0 --pairs no_midi_s100 --handoffs 5 25
  python evaluation/run_bridge_batched.py --print-split 0 3 5 6     # show the per-GPU commands
  python evaluation/run_bridge_batched.py --launch-gpus 0 3 5 6     # start them, detached
  python evaluation/run_bridge_batched.py --list-ckpts --baseline
  python evaluation/run_bridge_batched.py --merge
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bridge_200 import (  # noqa: E402
    BASELINE_PAIRS,
    CASES,
    CLIP_LENGTH,
    CSV_FIELDS,
    DATASET_ROOT,
    DEFAULT_HANDOFFS,
    DEFAULT_OUT_ROOT,
    DEFAULT_PAIRS,
    MIDI_FPS,
    MIDI_MODE,
    MODEL_PAIRS,
    NUM_STEPS,
    REPO_PATH,
    RHO,
    SAMPLING_RATE,
    SIGMA_MIN,
    append_row,
    build_jobs,
    chunks,
    condition_csv,
    experiment_name,
    format_job_spec,
    load_input_waveform,
    merge_csvs,
    output_paths,
    parse_job_specs,
    read_done,
    read_track_list,
    resolve_ckpt,
    safe_name,
    split_jobs,
    stem_paths,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default=None,
                   help="physical GPU index; sets CUDA_VISIBLE_DEVICES before torch is imported")
    p.add_argument("--pairs", nargs="+", default=None,
                   help="model pairs (default: the four sigma_max=100 pairs). Known: %s"
                        % ", ".join(MODEL_PAIRS))
    p.add_argument("--baseline", action="store_true",
                   help="also run the sigma_max=5 pairs (only at handoff 5)")
    p.add_argument("--handoffs", nargs="+", type=float, default=DEFAULT_HANDOFFS)
    p.add_argument("--jobs", nargs="+", default=None,
                   help="explicit pair:h1,h2 specs; overrides --pairs/--handoffs/--baseline")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n", type=int, default=None, help="only the first N tracks of each list")
    p.add_argument("--track-list-dir", default=None,
                   help="prefer <dir>/track_list_<case>_<midi|nomidi>.txt, the "
                        "per-architecture lists make_track_lists.py --pool-dir "
                        "writes. Falls back to the --track-list-<case> default "
                        "when the file is absent.")
    p.add_argument("--track-list-poly", default=CASES["poly"]["track_list"])
    p.add_argument("--track-list-mono", default=CASES["mono"]["track_list"])
    p.add_argument("--track-list-mono-b2c", default=CASES["mono_b2c"]["track_list"])
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--repo-path", default=REPO_PATH)
    p.add_argument("--dataset-root", default=DATASET_ROOT)
    p.add_argument("--source-ckpt", default=None, help="override, only with a single pair")
    p.add_argument("--target-ckpt", default=None, help="override, only with a single pair")
    p.add_argument("--no-dpd", action="store_true", help="skip DPD/JD (they run on CPU between batches)")

    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-pair", default="original_midi_s100")
    p.add_argument("--verify-handoff", type=float, default=25.0)
    p.add_argument("--verify-track", default=None)
    p.add_argument("--verify-batch", type=int, default=4,
                   help="size of the batch the verified clip is also run inside")

    p.add_argument("--merge", action="store_true", help="merge per-experiment CSVs and exit")
    p.add_argument("--list-ckpts", action="store_true", help="resolve checkpoints and exit")
    p.add_argument("--print-split", nargs="+", default=None, metavar="GPU",
                   help="print one nohup command per GPU and exit")
    p.add_argument("--launch-gpus", nargs="+", default=None, metavar="GPU",
                   help="start one detached process per GPU and exit")
    return p.parse_args()


def selected_jobs(args):
    if args.jobs:
        return parse_job_specs(args.jobs)
    pairs = list(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
    if args.baseline:
        pairs += [p for p in BASELINE_PAIRS if p not in pairs]
    return build_jobs(pairs, args.handoffs)


def track_list_for(args, cfg):
    """The clip list for one model pair, as a repo-relative path.

    A pair's list depends on its architecture as well as its direction: the
    MIDI and no-MIDI models of the same dataset landed on different validation
    splits, so each has its own pool of clips it never trained on. sigma_max
    does not change the split, so a pair and its _s100 twin share a list and
    their numbers stay comparable.
    """
    case_name = cfg["case"] if isinstance(cfg, dict) else cfg
    fallback = {
        "poly": args.track_list_poly,
        "mono": args.track_list_mono,
        "mono_b2c": args.track_list_mono_b2c,
    }[case_name]
    if not args.track_list_dir or not isinstance(cfg, dict):
        return fallback
    arch = "midi" if cfg["use_midi"] else "nomidi"
    rel = os.path.join(args.track_list_dir, "track_list_%s_%s.txt" % (case_name, arch))
    if os.path.isfile(rel) or os.path.isfile(os.path.join(args.repo_path, rel)):
        return rel
    print("no %s -- falling back to %s" % (rel, fallback))
    return fallback


def ckpts_for(args, pair, n_pairs):
    cfg = MODEL_PAIRS[pair]
    if (args.source_ckpt or args.target_ckpt) and n_pairs != 1:
        raise SystemExit("--source-ckpt/--target-ckpt need exactly one pair")
    src = resolve_ckpt(args.source_ckpt or cfg["source_ckpt"], args.repo_path)
    tgt = resolve_ckpt(args.target_ckpt or cfg["target_ckpt"], args.repo_path)
    return src, tgt


# ---------------------------------------------------------------- launching

def passthrough_args(args):
    out = ["--batch-size", str(args.batch_size), "--out-root", args.out_root,
           "--repo-path", args.repo_path, "--dataset-root", args.dataset_root,
           "--track-list-poly", args.track_list_poly, "--track-list-mono", args.track_list_mono,
           "--track-list-mono-b2c", args.track_list_mono_b2c]
    if args.track_list_dir:
        out += ["--track-list-dir", args.track_list_dir]
    if args.n is not None:
        out += ["--n", str(args.n)]
    if args.no_dpd:
        out.append("--no-dpd")
    return out


def worker_commands(args, gpus):
    shares = split_jobs(selected_jobs(args), len(gpus))
    script = str(Path(__file__).resolve())
    cmds = []
    for gpu, share in zip(gpus, shares):
        specs = [format_job_spec(pair, hs) for pair, hs in share]
        cmd = [sys.executable, "-u", script, "--jobs", *specs] + passthrough_args(args)
        cmds.append((gpu, specs, cmd))
    return cmds


def print_split(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.out_root, "logs")
    print("mkdir -p %s" % log_dir)
    for gpu, specs, cmd in worker_commands(args, args.print_split):
        log = os.path.join(log_dir, "gen_gpu%s_%s.log" % (gpu, stamp))
        print("CUDA_VISIBLE_DEVICES=%s nohup %s > %s 2>&1 &" % (gpu, " ".join(cmd), log))


def launch(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.repo_path) / args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_file = log_dir / ("pids_%s.txt" % stamp)
    with pid_file.open("w") as pf:
        for gpu, specs, cmd in worker_commands(args, args.launch_gpus):
            log = log_dir / ("gen_gpu%s_%s.log" % (gpu, stamp))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with open(log, "w") as lf:
                # New session: the worker survives the terminal closing, like nohup.
                proc = subprocess.Popen(cmd, env=env, cwd=args.repo_path, stdout=lf,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            pf.write("%d %s %s\n" % (proc.pid, gpu, " ".join(specs)))
            print("GPU %s  pid %d  %s\n    log: %s" % (gpu, proc.pid, " ".join(specs), log))
    print("pids -> %s" % pid_file)


# ---------------------------------------------------------------- the bridge

def import_reference_module():
    """Run_timbre_transfer.py runs its CONFIG at import. Give it a known-valid
    experiment and no handoff override so the import cannot raise, then restore."""
    saved = {k: os.environ.get(k) for k in ("EXPERIMENT_NAME", "SIGMA_HANDOFF")}
    os.environ["EXPERIMENT_NAME"] = "no_midi"
    os.environ.pop("SIGMA_HANDOFF", None)
    try:
        import Run_timbre_transfer as rtt
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    expected = dict(SAMPLING_RATE=SAMPLING_RATE, CLIP_LENGTH=CLIP_LENGTH, NUM_STEPS=NUM_STEPS,
                    SIGMA_MIN=SIGMA_MIN, RHO=RHO, MIDI_MODE=MIDI_MODE, MIDI_FPS=MIDI_FPS,
                    ADD_FIFTH=False)
    drift = {k: (getattr(rtt, k), v) for k, v in expected.items() if getattr(rtt, k) != v}
    if drift:
        raise RuntimeError("Run_timbre_transfer.py constants differ from bridge_200.py "
                           "(script, ours): %s" % drift)
    return rtt


class Bridge:
    def __init__(self, pair, source_ckpt, target_ckpt, device, rtt):
        import torch
        from main.module_base_latent_cond import Model, AudioDiffusionModel
        from audio_diffusion_pytorch import KDistribution, NormalizedEncodec

        cfg = MODEL_PAIRS[pair]
        self.pair = pair
        self.case = CASES[cfg["case"]]
        self.use_midi = cfg["use_midi"]
        self.trained_sigma = cfg["trained_sigma"]
        self.source_ckpt, self.target_ckpt = source_ckpt, target_ckpt
        self.device = device
        self.rtt = rtt

        sigma_distribution = KDistribution(
            sigma_min=SIGMA_MIN, sigma_max=self.trained_sigma, rho=RHO)

        def load(ckpt, mean_path, std_path, label):
            print("  loading %s: %s" % (label, ckpt), flush=True)
            diffusion = AudioDiffusionModel(
                diffusion_sigma_distribution=sigma_distribution,
                use_midi=self.use_midi,
                midi_mode=MIDI_MODE,
                midi_bins=self.case["midi_bins"],
            )
            model = Model(model=diffusion, mean_path=mean_path, std_path=std_path,
                          use_midi=self.use_midi)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state["state_dict"], strict=True)
            del state
            return model.to(device)

        self.source = load(source_ckpt, self.case["source_mean"], self.case["source_std"], "SOURCE")
        self.target = load(target_ckpt, self.case["target_mean"], self.case["target_std"], "TARGET")
        self.encodec = NormalizedEncodec(device=device)

    def prepare(self, tracks, dataset_root):
        """Inputs [B, 1, N] on device, stacked rolls [B, bins, T] or None."""
        import torch

        paths = [stem_paths(dataset_root, t, self.case["stems"]) for t in tracks]
        x = torch.cat([load_input_waveform(p) for p in paths], dim=0).to(self.device)
        midi = None
        if self.use_midi:
            target_frames = int(round(CLIP_LENGTH / SAMPLING_RATE * MIDI_FPS))
            midi = torch.cat([
                self.rtt.load_piano_roll(self.rtt.derive_midi_paths(p), target_frames,
                                         MIDI_FPS, self.case["midi_bins"], self.device)
                for p in paths
            ], dim=0)
        return x, midi, paths

    def encode(self, x):
        return split_on_oom(
            lambda xb, _: self.encodec.encode_latent(xb, self.source.mean, self.source.std), x)

    def transfer(self, emb, midi, handoff):
        from audio_diffusion_pytorch import KarrasSampler, KarrasSamplerReverse, KarrasSchedule

        schedule = KarrasSchedule(sigma_min=SIGMA_MIN, sigma_max=handoff, rho=RHO)

        def run(e, m):
            noisy = self.source.model.sample(
                noise=e, midi=m, sampler=KarrasSamplerReverse(),
                sigma_schedule=schedule, num_steps=NUM_STEPS)
            generated = self.target.model.sample(
                noise=noisy, midi=m, sampler=KarrasSampler(),
                sigma_schedule=schedule, num_steps=NUM_STEPS)
            return self.encodec.decode_latent(generated, self.target.mean, self.target.std)

        return split_on_oom(run, emb, midi)


def split_on_oom(fn, x, extra=None):
    """fn(x[, extra]) on the whole batch; on CUDA OOM, halve and retry."""
    import torch

    try:
        return fn(x, extra)
    except torch.cuda.OutOfMemoryError:
        if x.shape[0] == 1:
            raise
        torch.cuda.empty_cache()
        k = x.shape[0] // 2
        print("  CUDA OOM at batch %d, splitting into %d + %d" % (x.shape[0], k, x.shape[0] - k),
              flush=True)
        lo = split_on_oom(fn, x[:k], None if extra is None else extra[:k])
        hi = split_on_oom(fn, x[k:], None if extra is None else extra[k:])
        return torch.cat([lo, hi], dim=0)


def pitch_metrics(tracker, input_waveform, target_waveform_np, source, target):
    """Exactly Run_timbre_transfer.py's call, including plot=True (Bug 9)."""
    import torch

    tracking_output = tracker.cals_pitch_metric(
        input_waveform,
        torch.tensor(target_waveform_np),
        "both",
        plot=True,
        pair=(f"Input {source}", f"Generated {target}"),
    )
    dtw, jaccard = tracking_output[0][0]
    return float(dtw), float(jaccard)


def save_input(path, input_waveform, first_copy):
    """Inputs are identical across handoffs: hardlink to the first copy instead
    of writing the same 1.6 MB again."""
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    if first_copy is not None and first_copy.exists():
        try:
            os.link(first_copy, path)
            return
        except OSError:
            shutil.copyfile(first_copy, path)
            return
    torchaudio.save(str(path), input_waveform.squeeze(0).cpu(), SAMPLING_RATE)


def run_jobs(args, jobs):
    import matplotlib
    matplotlib.use("Agg")
    import torch
    import torchaudio

    rtt = import_reference_module()
    os.chdir(args.repo_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device %s | CUDA_VISIBLE_DEVICES=%s" % (device, os.environ.get("CUDA_VISIBLE_DEVICES")))
    print("jobs: %s" % " ".join(format_job_spec(p, hs) for p, hs in jobs), flush=True)

    tracker = None
    if not args.no_dpd:
        from audio_diffusion_pytorch import PitchTracker
        tracker = PitchTracker()

    for pair, handoffs in jobs:
        cfg = MODEL_PAIRS[pair]
        case = CASES[cfg["case"]]
        tracks = read_track_list(track_list_for(args, cfg))
        if args.n is not None:
            tracks = tracks[:args.n]
        exps = {h: experiment_name(pair, cfg["trained_sigma"], h) for h in handoffs}
        done = {h: read_done(condition_csv(args.out_root, exps[h])) for h in handoffs}
        todo = [t for t in tracks if any((exps[h], t) not in done[h] for h in handoffs)]

        print("\n" + "=" * 70)
        print("  %s | %s -> %s | MIDI %s | trained sigma %g | handoffs %s"
              % (pair, case["source"], case["target"], cfg["use_midi"], cfg["trained_sigma"],
                 ", ".join("%g" % h for h in handoffs)))
        print("  %d tracks, %d with work left" % (len(tracks), len(todo)))
        print("=" * 70, flush=True)
        if not todo:
            continue

        source_ckpt, target_ckpt = ckpts_for(args, pair, len(jobs))
        bridge = Bridge(pair, source_ckpt, target_ckpt, device, rtt)

        t_start, n_clips = time.time(), 0
        total = sum(1 for t in todo for h in handoffs if (exps[h], t) not in done[h])
        for batch in chunks(todo, args.batch_size):
            t_batch = time.time()
            x, midi, paths = bridge.prepare(batch, args.dataset_root)
            emb = bridge.encode(x)
            first_inputs = {}
            for h in handoffs:
                idx = [i for i, t in enumerate(batch) if (exps[h], t) not in done[h]]
                if not idx:
                    continue
                wav = bridge.transfer(emb[idx], None if midi is None else midi[idx], h)
                for j, i in enumerate(idx):
                    track = batch[i]
                    input_waveform = x[i:i + 1]
                    target_waveform_np = wav[j:j + 1].cpu().detach().squeeze(0).numpy()
                    out_path, in_path = output_paths(
                        args.out_root, exps[h], track, case["source"], case["target"])
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(str(out_path), torch.tensor(target_waveform_np).cpu(), SAMPLING_RATE)
                    save_input(in_path, input_waveform, first_inputs.get(track))
                    first_inputs.setdefault(track, in_path)

                    dpd = jd = ""
                    if tracker is not None:
                        dpd, jd = pitch_metrics(tracker, input_waveform, target_waveform_np,
                                                case["source"], case["target"])
                    append_row(condition_csv(args.out_root, exps[h]), {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment": exps[h],
                        "pair": pair,
                        "case": cfg["case"],
                        "sample_id": track,
                        "source": case["source"],
                        "target": case["target"],
                        "output_audio": str(out_path),
                        "input_audio": str(in_path),
                        "DPD": dpd,
                        "JD": jd,
                        "num_steps": NUM_STEPS,
                        "use_midi": cfg["use_midi"],
                        "midi_mode": MIDI_MODE if cfg["use_midi"] else "",
                        "midi_bins": case["midi_bins"] if cfg["use_midi"] else "",
                        "trained_sigma": cfg["trained_sigma"],
                        "sigma_handoff": h,
                        "source_ckpt": source_ckpt,
                        "target_ckpt": target_ckpt,
                        "input_stems": ";".join(paths[i]),
                        "batch_size": len(idx),
                    })
                    done[h].add((exps[h], track))
                    n_clips += 1
            elapsed = time.time() - t_start
            eta = elapsed / n_clips * (total - n_clips) if n_clips else float("nan")
            print("  [%s] batch of %d tracks in %.0fs | %d/%d clips | elapsed %.1f min | ETA %.1f min"
                  % (pair, len(batch), time.time() - t_batch, n_clips, total,
                     elapsed / 60, eta / 60), flush=True)

        del bridge
        torch.cuda.empty_cache()

    print("\nall jobs done", flush=True)


# ---------------------------------------------------------------- verify

def compare(ref, ours):
    import torch

    n = min(ref.shape[-1], ours.shape[-1])
    ref, ours = ref[..., :n].double(), ours[..., :n].double()
    err = ours - ref
    noise = float((err ** 2).sum())
    snr = float("inf") if noise == 0 else 10 * torch.log10((ref ** 2).sum() / noise).item()
    return bool(torch.equal(ref, ours)), float(err.abs().max()), snr


def verify(args):
    import torch
    import torchaudio
    import csv as csv_mod

    pair, handoff = args.verify_pair, args.verify_handoff
    cfg = MODEL_PAIRS[pair]
    case = CASES[cfg["case"]]
    script = case["reference_script"]
    if script is None:
        raise SystemExit("%s has no reference script to verify against. Verify a poly pair "
                         "(Run_timbre_transfer.py) or a mono_* bassoon->cello pair "
                         "(Run_timbre_transfer_mono.py); the code path is the same." % pair)
    exp = experiment_name(pair, cfg["trained_sigma"], handoff)
    tracks = read_track_list(os.path.join(args.repo_path, track_list_for(args, cfg)))
    track = args.verify_track or tracks[0]
    others = [t for t in tracks if t != track][:max(0, args.verify_batch - 1)]

    vroot = (Path(args.repo_path) / args.out_root / "verify" / safe_name(exp)).resolve()
    ref_root = vroot / "reference"
    ref_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  VERIFY %s (handoff %g) on %s" % (pair, handoff, track))
    print("  step 1/2: %s in a subprocess" % script)
    print("=" * 70, flush=True)
    env = dict(os.environ, TRACK_NAME=track, EXPERIMENT_NAME=pair,
               METRICS_ROOT=str(ref_root), SIGMA_HANDOFF="%g" % handoff)
    subprocess.run([sys.executable, "-u", os.path.join(args.repo_path, script)],
                   env=env, cwd=args.repo_path, check=True)

    rtt = import_reference_module()
    os.chdir(args.repo_path)
    ref_out = ref_root / "audio" / ("%s_%s_generated_%s_from_%s.wav"
                                    % (safe_name(exp), track, case["target"], case["source"]))
    ref_in = Path(rtt.WAV_DIR) / ("input_%s.wav" % case["source"])
    ref_csv = ref_root / ("inference_metrics.csv" if script == "Run_timbre_transfer.py"
                          else "inference_metrics_mono.csv")
    ref_row = None
    with ref_csv.open(newline="") as f:
        for row in csv_mod.DictReader(f):
            if row["experiment"] == exp and row["sample_id"] == track:
                ref_row = row

    print("=" * 70)
    print("  step 2/2: batched runner, alone and inside a batch of %d" % (1 + len(others)))
    print("=" * 70, flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_ckpt, target_ckpt = ckpts_for(args, pair, 1)
    bridge = Bridge(pair, source_ckpt, target_ckpt, device, rtt)

    def run(batch):
        x, midi, _ = bridge.prepare(batch, args.dataset_root)
        wav = bridge.transfer(bridge.encode(x), midi, handoff)
        return x[0:1], wav[0:1].cpu().detach().squeeze(0).numpy()

    def roundtrip(tensor, name):
        path = vroot / name
        torchaudio.save(str(path), tensor, SAMPLING_RATE)
        return torchaudio.load(str(path))[0]

    x1, out1 = run([track])
    xb, outb = run([track] + others)
    ours_in = roundtrip(x1.squeeze(0).cpu(), "ours_input.wav")
    ours_single = roundtrip(torch.tensor(out1), "ours_single.wav")
    ours_batched = roundtrip(torch.tensor(outb), "ours_batched.wav")
    ref_input = torchaudio.load(str(ref_in))[0]
    ref_output = torchaudio.load(str(ref_out))[0]

    results = {
        "input": compare(ref_input, ours_in),
        "output, batch of 1": compare(ref_output, ours_single),
        "output, batch of %d" % (1 + len(others)): compare(ref_output, ours_batched),
    }
    print("\n%-26s %-7s %-12s %s" % ("", "exact", "max |diff|", "SNR vs script (dB)"))
    for name, (exact, maxdiff, snr) in results.items():
        print("%-26s %-7s %-12.3g %.1f" % (name, exact, maxdiff, snr))

    if ref_row is not None and not args.no_dpd:
        from audio_diffusion_pytorch import PitchTracker
        dpd, jd = pitch_metrics(PitchTracker(), x1, out1, case["source"], case["target"])
        print("\nDPD script %.4f  ours %.4f | JD script %.4f  ours %.4f"
              % (float(ref_row["DPD"]), dpd, float(ref_row["JD"]), jd))

    in_ok = results["input"][0]
    single = results["output, batch of 1"]
    batched = list(results.values())[2]
    single_ok = single[0] or single[2] >= 60
    batched_ok = batched[0] or batched[2] >= 40
    print()
    if not in_ok:
        print("FAIL: the input preprocessing differs from the script's. Do not launch.")
    elif not single_ok:
        print("FAIL: same input, different output at batch 1 -- the bridge differs. Do not launch.")
    elif not batched_ok:
        print("FAIL: batch of 1 matches but batching changes the output beyond float noise. "
              "Launch with --batch-size 1, or investigate before trusting batched runs.")
    else:
        print("PASS: input identical; batch-1 output %s; batched output %s."
              % ("bit-identical" if single[0] else "equal to float precision (SNR %.0f dB)" % single[2],
                 "bit-identical" if batched[0] else
                 "equal up to GPU float noise from the larger batch (SNR %.0f dB)" % batched[2]))
    return 0 if (in_ok and single_ok and batched_ok) else 1


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.merge:
        merged = merge_csvs(Path(args.repo_path) / args.out_root)
        print("merged -> %s" % merged)
        return 0
    if args.print_split:
        print_split(args)
        return 0
    if args.launch_gpus:
        launch(args)
        return 0

    jobs = selected_jobs(args)
    if args.list_ckpts:
        missing = 0
        for pair, handoffs in jobs:
            try:
                src, tgt = ckpts_for(args, pair, len(jobs))
                print("%-30s source %s\n%-30s target %s" % (pair, src, "", tgt))
            except FileNotFoundError as e:
                missing += 1
                print("%-30s MISSING: %s" % (pair, e))
        return 1 if missing else 0
    if args.verify:
        return verify(args)

    for sub in ("audio", "csv", "logs"):
        (Path(args.repo_path) / args.out_root / sub).mkdir(parents=True, exist_ok=True)
    run_jobs(args, jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
