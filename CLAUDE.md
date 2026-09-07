# CLAUDE.md — Diffusion Timbre Transfer (Music Proj 2026)

This file gives Claude (or any new assistant / teammate) the full context needed to work inside this repository and dataset environment. It is intentionally detailed so that a fresh conversation can pick up exactly where the last one left off, with no loss of context.

> **Status (2026-09-06):**
> - **Phase 1 — AddFifth polyphony augmentation: DONE.** Cello and bassoon each trained 100 epochs with the fifth added. `Run_timbre_transfer.py` runs the full cello→bassoon bridge end to end.
> - **Phase 2 — MIDI piano-roll conditioning: IN PROGRESS.** The pipeline is wired end to end (cache builders → dataset → gated UNet conditioning → W&B logging), but **no full MIDI-conditioned model has been trained yet** — only a 1-epoch smoke test. A crop-alignment bug that would have silently defeated the conditioning was found and fixed on 2026-09-06 (see **Bug 10**); any MIDI run started before that date is invalid.

---

## 👥 Team & Environment

| Field | Value |
|---|---|
| Developers | Shlomo (`benshise`) & Yuval (`chenyuv`) |
| Host / Server | `dgx01` (Lab container environment) |
| Python Env | `venv_yuval` at `/home/shared_workspace/diffusion-timbre-transfer/venv_yuval` |
| Python binary | `/home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/python` |
| Python version | 3.12 |
| Repo Path | `/home/shared_workspace/diffusion-timbre-transfer` |
| Dataset Root | `/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj` |
| Our outputs (checkpoints) | `/home/shared_workspace/diffusion-timbre-transfer/our_checkpoints` |
| Our outputs (inference WAV) | `/home/shared_workspace/diffusion-timbre-transfer/Outputs/WAV_files` |
| Our outputs (inference PNG) | `/home/shared_workspace/diffusion-timbre-transfer/Outputs/WAV_files/png_files` |
| MIDI caches | `/home/shared_workspace/diffusion-timbre-transfer/midi_cache_<instrument>_fps75[_addfifth]` |

### Activating the environment
```bash
source /home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/activate
```

> **GPU machine:** `dgx01` has **8× Tesla V100-SXM2-32GB** GPUs (indices 0–7). Each has ~32 GB. Training one model uses ~30 GB (almost the whole GPU). Multiple lab users share these GPUs.

---

## 🔑 Permissions — IMPORTANT, this caused a lot of friction

The shared workspace has a tangled ownership situation. Read this before debugging any "Permission denied" error.

- The `our_checkpoints/` directory itself is owned by `benshise` with permissions `drwxrwsrwx` (note the **setgid `s` bit** on the group).
- Because of the **setgid bit**, files created *inside* `our_checkpoints/` inherit the `dsi` group, and subdirectories created during training runs end up owned by whichever user ran the training. As a result, **neither `benshise` nor `chenyuv` can `chmod` each other's subdirectories** — both get "Permission denied".
- `sudo` is **not** available to the developers.
- `screen` and `tmux` are **NOT installed** on this machine.

### Consequences / working rules
1. **Don't try to `chmod -R 777 our_checkpoints`** — it will partially fail on subdirectories owned by the other user. This is expected, not a new bug.
2. To avoid permission collisions, **each training run uses a unique `exp_tag`** so it writes to a fresh subdirectory that the running user owns.
3. **Inference outputs** (WAV + PNG) are saved to `Outputs/WAV_files/...`. Earlier we temporarily used `/home/benshise/` (always writable by Shlomo) when the shared dir had issues — that's a valid fallback if shared write fails again.
4. The original `logs/` folder is owned by yet another user — we never had write access. That's why all training output goes to `our_checkpoints/`.
5. `torchaudio.save(...)` and `matplotlib.savefig(...)` do **NOT** create missing directories. Any output folder must be created first with `mkdir -p`.
6. W&B writes its own run dir. Each user points it at their own home: `loggers.wandb.save_dir=/home/chenyuv/wandb_logs` (or `/home/benshise/...`).

---

## 📂 Repository Layout

```
diffusion-timbre-transfer/
├── audio_data_pytorch/          ← dataset / dataloader classes
│   ├── datasets/
│   │   └── wav_dataset.py       ← WAVDataset (EDITED — loads MIDI piano rolls)
│   ├── transforms/
│   │   └── all.py               ← AllTransform + our AddFifth class (EDITED)
│   ├── declipping.py
│   └── utils.py
├── audio_diffusion_pytorch/     ← model code
│   ├── basic-pitch-torch/       ← cloned from github during setup (gitignored)
│   ├── diffusion.py
│   ├── encodec_utils.py
│   ├── model.py
│   ├── modules.py               ← UNet1d (EDITED — midi_encoder + midi_gate)
│   ├── pitch_tracking_utils.py
│   └── utils.py
├── main/
│   ├── module_base_latent_cond.py  ← Model/Datamodule/SampleLogger (EDITED)
│   └── utils.py
├── exp/                         ← Hydra experiment configs
│   ├── bassoon_latent.yaml      ← EDITED
│   ├── cello_latent.yaml        ← EDITED
│   ├── flute_latent.yaml / violin_latent.yaml / trumpet_latent.yaml
│   ├── pitch_flute_latent.yaml
│   ├── callbacks/base.yaml
│   ├── datamodule/base.yaml     ← EDITED (random_crop_size + midi_fps, see Bug 10)
│   ├── loggers/
│   │   ├── tensorboard.yaml
│   │   └── wandb.yaml           ← EDITED — this is the ACTIVE logger
│   ├── model/latent.yaml
│   └── trainer/full.yaml        ← EDITED
├── midi_preprocessing/          ← NEW — build the MIDI piano-roll caches
│   ├── midi_to_pianoroll.py               ← single .mid → .npy piano roll
│   ├── create_cello_midi_cache.py         ← wav → basic-pitch → .npy, whole dataset
│   ├── create_bassoon_midi_cache.py       ← same, bassoon stems
│   └── create_addfifth_pianoroll_cache.py ← adds +7 semitones to a cached roll
├── debug_midi/                  ← NEW — visual sanity checks
│   ├── debug_compare_audio_midi_crop.py    ← spectrogram vs piano roll, same window
│   └── export_cropped_piano_roll_to_midi.py
├── scripts/
│   └── create_fifth_polyphony.py  ← offline audio +fifth (librosa), for test inputs
├── checkpoints/                 ← mean/std tensors and pretrained ckpts (gitignored)
├── our_checkpoints/             ← OUR trained models (gitignored)
├── Outputs/WAV_files/png_files/ ← inference outputs (gitignored)
├── config.yaml                  ← root Hydra config
├── train.py                     ← training entry point (EDITED)
├── Run_timbre_transfer.py       ← OUR single-file inference script
├── inference.ipynb              ← original notebook (the bridge that does the transfer)
└── requirements.txt
```

---

## 🗄️ Dataset

**Active path (TINY, already extracted):**
```
/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/cocochorales_tiny_v1_zipped/main_dataset
```

⚠️ **Important:** The path is `main_dataset/` directly — **NOT** `main_dataset/train/`. The `train/`, `valid/`, `test/` subfolders contain only un-extracted `.tar.bz2` archives.

Each track folder has the structure:
```
<family>_track<ID>/
└── stems_audio/
    ├── 1_<instrument>.wav
    ├── 2_<instrument>.wav
    ├── 3_<instrument>.wav
    └── 4_<instrument>.wav
```

Relevant stems for our project:
- **Bassoon** — `woodwind_track*/stems_audio/4_bassoon.wav` (**9982 tracks**, confirmed in training logs)
- **Cello** — `string_track*/stems_audio/4_cello.wav` (**11395 tracks**, confirmed in training logs)

Source stems are **16 kHz**; everything is resampled to **24 kHz** to match Encodec.

---

## 🧠 How the Model Works

This repo implements **Latent Diffusion Bridges** for unsupervised timbre transfer.

> Paper: https://arxiv.org/abs/2409.06096
> The paper trains on **CocoChorales** (unpaired monophonic single-instrument audio), one diffusion model per instrument, with a Gaussian prior. At inference, a SOURCE model maps the input to its Gaussian prior, and a TARGET model reconstructs from that prior — facilitating timbre transfer. Evaluation in the paper is by **FAD (Fréchet Audio Distance)** and **DPD (pitch distance)** — NOT by training loss. The noise level σ controls the melody-preservation vs timbre-transfer tradeoff.

**Conceptual point:** "cello → bassoon" is NOT a single training run.

1. Audio is encoded into a latent space (Encodec, `facebook/encodec_24khz`, 24 kHz). Latent length is **1280** for a 409600-sample (17 s) clip.
2. A **separate diffusion model is trained per instrument** — one for cello, one for bassoon.
3. The **actual transfer happens at inference**: reverse-diffuse the input to noise with the SOURCE model, then forward-diffuse that noise to the TARGET instrument with the TARGET model.

So the task = **train 2 models** + **run the bridge at inference**.

### Model size (from training logs)
- `encodec_model`: 14.9 M params (frozen, eval mode)
- `model` (AudioDiffusionModel / UNet1d / XDiffusion): 302 M params (trainable)
- `model_ema`: 604 M params
- Total ~619 M. Trainable 302 M. Non-trainable 317 M.

---

## 🎼 PROJECT GOAL 1 — AddFifth / polyphony augmentation

**What we do differently from the base repo:** each training example gets an **additional voice a perfect fifth higher (7 semitones up)** mixed in, so it sounds like **two instruments playing together**. At inference the input is augmented the same way so it matches the training distribution.

### ⚠️ Where the fifth is added — this has changed twice, read carefully

| Placement | Status | Why |
|---|---|---|
| 1. In the dataloader (`base.yaml` transforms) | ❌ abandoned | Runs once per sample **on CPU** in the dataloader workers. CPU saturates, GPU starves: ~4 min/epoch → **15+ min/epoch, stuck at 0/570**. |
| 2. On GPU in `training_step` | ⚠️ **currently commented out** | Was the fix for (1) — one GPU op per batch, restored ~7 min/epoch. Still present at `module_base_latent_cond.py:194` and `:290` but **disabled**. |
| 3. Baked into the offline caches | ✅ **current** | For Phase 2 the fifth lives in the **piano-roll cache** (`*_fps75_addfifth`, built by `create_addfifth_pianoroll_cache.py`), not applied to audio at train time. |

> **If you re-enable the GPU AddFifth**, uncomment `batch[0] = self.add_fifth_transform(batch[0])` in **both** `training_step` and `validation_step`. Do NOT move it back into the dataloader. Sanity check: the Lightning summary shows **"Modules in train mode: 1270"** (vs 1269) when AddFifth is registered as a submodule.

### The `AddFifth` class — `audio_data_pytorch/transforms/all.py`

Uses precomputed `Resample` transforms built once in `__init__`, **not** `torchaudio.functional.pitch_shift` per call (see Bug 2).

```python
class AddFifth(nn.Module):
    def __init__(self, sample_rate: int = 24000, fifth_gain: float = 0.8):
        super().__init__()
        self.fifth_gain = fifth_gain
        ratio = 2 ** (7 / 12)  # 7 semitones up = perfect fifth
        orig_freq = 10000
        new_freq = round(orig_freq / ratio)  # ~ 6674
        self.downsample = torchaudio.transforms.Resample(orig_freq, new_freq)
        self.upsample   = torchaudio.transforms.Resample(new_freq, orig_freq)

    def forward(self, x: Tensor) -> Tensor:
        x_fifth = self.upsample(self.downsample(x))
        min_len = min(x.shape[-1], x_fifth.shape[-1])
        x, x_fifth = x[..., :min_len], x_fifth[..., :min_len]
        x_mix = x + self.fifth_gain * x_fifth
        max_val = torch.max(torch.abs(x_mix))
        if max_val > 0:
            x_mix = 0.95 * x_mix / max_val
        return x_mix
```

> `sample_rate` is accepted for API compatibility but unused — the ratio comes from the hardcoded `orig_freq`/`new_freq`. Harmless. Resample-based shifting is an approximation of true pitch shift; fine for augmentation.

For making **one-off polyphonic test files** offline, `scripts/create_fifth_polyphony.py` does the same thing properly with `librosa.effects.pitch_shift` (slow, but exact):
```bash
python scripts/create_fifth_polyphony.py --input real_cello.wav --output real_cello_fifth.wav
```

---

## 🎹 PROJECT GOAL 2 — MIDI piano-roll conditioning (current work)

Give the diffusion UNet an explicit **note-level conditioning signal** — a 128×T binary piano roll — alongside the audio, so the model knows *which notes* it should be producing rather than inferring everything from the latent.

### The pipeline, end to end

```
stem wav --basic-pitch--> .mid --pretty_midi--> [128, T] binary roll --> .npy cache
                                                                             |
                                              create_addfifth_pianoroll_cache.py
                                                        (adds +7 semitone row)
                                                                             |
                                                                             v
                                            midi_cache_<inst>_fps75_addfifth/
                                                                             |
                                        WAVDataset.__getitem__ loads + crops it
                                                                             |
                                                                             v
                     Model.training_step: F.interpolate to latent length (1280)
                                                                             |
                                                                             v
                UNet1d.forward: x = x + midi_gate * midi_encoder(midi)   <- after to_in
```

### Step 1 — build the raw caches (once per instrument)

`basic-pitch` transcribes each stem to MIDI, then `pretty_midi.get_piano_roll(fs=fps)` renders it and thresholds on velocity.

```bash
python midi_preprocessing/create_cello_midi_cache.py \
  --dataset_root /dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/cocochorales_tiny_v1_zipped/main_dataset \
  --cache_root  /home/shared_workspace/diffusion-timbre-transfer/midi_cache_cello_fps75 \
  --fps 75 --velocity_threshold 40
```

> ⚠️ **`--fps 75` is not the default.** Both `create_*_midi_cache.py` and `midi_to_pianoroll.py` default to **50**. The caches we use are **75 fps** — the directory names say so (`..._fps75`), and `WAVDataset(midi_fps=75)` / `base.yaml` must agree. Passing the wrong `--fps` produces a cache that is silently time-scaled wrong.

Layout produced: `<cache_root>/<track_name>/<stem_name>_pianoroll.npy`, each `[128, T]` `uint8`.

**Two stages with very different costs.** `basic-pitch` transcribing wav to `.mid` is the slow part (hours over the full dataset). Rendering `.mid` to `.npy` is near-instant — and it is where **both** `--fps` and `--velocity_threshold` are applied. So:

- By default the script **deletes each `.mid` once its `.npy` is written**. Nothing downstream reads them, and keeping them costs one directory per track — tens of thousands across a dataset.
- Pass **`--keep_midi`** if you might rebuild at a different `--fps` or `--velocity_threshold`. On the next run the kept `.mid` is reused and transcription is skipped entirely, turning hours into seconds.
- A `.mid` whose render *failed* is never deleted, so you can inspect it.

### Step 2 — add the fifth to the rolls

```bash
python midi_preprocessing/create_addfifth_pianoroll_cache.py
```
Edit `SRC_CACHE` / `DST_CACHE` at the top of the file (module constants, not CLI args). It shifts every active note up 7 rows and ORs it in:
```python
out[7:, :] = np.maximum(out[7:, :], roll[:-7, :])
```

### Step 3 — the dataset reads it

`WAVDataset.get_midi_cache_path()` maps a wav path to its cached roll. **The cache roots are hardcoded absolute paths inside that method**, keyed off `"cello"` / `"bassoon"` appearing in the filename — anything else raises `ValueError`. If you add an instrument or move the caches, edit that method.

`__getitem__` returns a **4-tuple**:
```python
return waveform, instrument_name[2:], self.wavs[idx], piano_roll
#      batch[0]  batch[1]             batch[2]        batch[3]
```
(`instrument_name[2:]` strips the leading `4_` from e.g. `4_cello`.) A missing cache file degrades gracefully to a zero roll rather than crashing.

### Step 4 — conditioning inside the UNet

`audio_diffusion_pytorch/modules.py`:
```python
# in UNet1d.__init__
self.midi_encoder = nn.Sequential(
    nn.Conv1d(128, midi_hidden_channels, 3, padding=1),
    nn.SiLU(),
    nn.Conv1d(midi_hidden_channels, midi_hidden_channels, 3, padding=1),
)
# Starts at zero, so the model initially behaves exactly like the original model
self.midi_gate = nn.Parameter(torch.zeros(1))

# in UNet1d.forward, right after x = self.to_in(x, mapping)
if midi is not None:
    midi_features = self.midi_encoder(midi)
    if midi_features.shape[-1] != x.shape[-1]:
        midi_features = F.interpolate(midi_features, size=x.shape[-1], mode="nearest")
    x = x + self.midi_gate * midi_features
```

**`midi_gate` is the health indicator for this whole feature.** It is a single learnable scalar starting at 0, logged every step as `train/midi_gate` and printed every 500 steps. If the conditioning carries real information the gate should move away from zero. **A gate that stays flat at ~0 means the model has decided MIDI is useless** — which is exactly what a misaligned or wrongly-scaled roll looks like. Check this first when evaluating a MIDI run.

---

## ⚙️ Modified Files (summary)

| File | What changed |
|---|---|
| `audio_data_pytorch/transforms/all.py` | Added `AddFifth`; fixed `Resample` guard (Bug 3) |
| `audio_data_pytorch/datasets/wav_dataset.py` | `get_midi_cache_path()`, piano-roll load, 4-tuple return, `midi_fps` param, crop-offset alignment (Bug 10) |
| `audio_diffusion_pytorch/modules.py` | `midi_encoder` + `midi_gate` in `UNet1d`; `midi` kwarg in `forward` |
| `main/module_base_latent_cond.py` | AddFifth wiring (now commented out), MIDI resize + threading into the model, `train/midi_gate` logging, validation conditioning (Bug 10) |
| `exp/datamodule/base.yaml` | `random_crop_size` + `midi_fps` on the dataset (Bug 10); `add_fifth` lines commented out |
| `exp/trainer/full.yaml` | `gpus: -1` changed to `devices: 1` |
| `exp/cello_latent.yaml`, `exp/bassoon_latent.yaml` | dataset path, `exp_tag`, `logs_dir`, `devices: 1`, `loggers: wandb` |
| `train.py` | `CUDA_VISIBLE_DEVICES` + Hydra overrides before imports |

### `train.py` — set GPU and overrides BEFORE imports
`CUDA_VISIBLE_DEVICES` MUST be set before any CUDA/torch import or it is ignored.
```python
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"   # pick a FREE GPU (check nvidia-smi)

sys.argv += [
    "exp=bassoon_latent",                  # or cello_latent
    "trainer.max_epochs=100",
    "datamodule.batch_size=16",
    "datamodule.num_workers=8",
    "exp_tag=bassoon_midi_v1",             # unique tag per run (permissions!)
    "logs_dir=/home/shared_workspace/diffusion-timbre-transfer/our_checkpoints",
    "loggers.wandb.save_dir=/home/chenyuv/wandb_logs",
]
# ... then dotenv, hydra, pytorch_lightning imports follow ...
```

> ⚠️ **`train.py` is currently configured as a 1-epoch smoke test** (`max_epochs=1`, `+trainer.limit_train_batches=1`, `+trainer.limit_val_batches=0`, tag `bassoon_midi_addfifth_wandb_test_yuval`). Remove those limits before a real run.

---

## 🐞 Bugs Encountered & Fixes (full history — read before touching AddFifth or MIDI)

**Bug 1 — `_init_` instead of `__init__` (single underscores).**
The original `AddFifth` and `AllTransform` used `def _init_(...)` / `super()._init_()`. Python treats `_init_` as an ordinary method, so attributes were never stored and `forward()` raised `AttributeError: ... has no attribute 'sample_rate'`. Fix: dunder `__init__` everywhere.

**Bug 2 — `AF.pitch_shift` was the training slowdown.**
`torchaudio.functional.pitch_shift` does STFT pitch shifting on CPU, rebuilding expensive state every call in the dataloader workers — the main cause of 15+ min/epoch. Fix: precomputed `torchaudio.transforms.Resample` built once in `__init__`.

**Bug 3 — `target_rates` (plural) NameError.**
In `AllTransform.__init__` the `Resample` guard read `source_rate != target_rates`. Fix: singular `target_rate`.

**Bug 4 — saving WAV to a read-only path.**
`torchaudio.save(...)` failed with `LibsndfileError: ... System error`. It was **permissions**, not audio. Fix: save somewhere you own, and `mkdir -p` it first — the save functions do not create directories.

**Bug 5 — `add_fifth_transform` "has no attribute" / device pinning.**
- `.to('cuda')` in `__init__` pins the transform before Lightning moves the model, causing a device mismatch. Remove it; Lightning handles submodules.
- A stray `self.length = length` line raised `NameError` *before* `self.add_fifth_transform` was assigned, making the attribute look absent. Remove it.
- Edits sometimes silently did not save — verify with `cat -n ... | head -120`.

**Bug 6 — `devices: [1]` vs `devices: 1`.**
`devices: [1]` means "GPU physical index 1". Combined with `CUDA_VISIBLE_DEVICES=N` (which remaps the chosen GPU to index 0), Lightning crashes: `You requested gpu: [1] But your machine only has: [0]`. Fix: scalar `devices: 1`.

**Bug 7 — CUDA out of memory even though "GPU is free".**
The OOM message lists the processes holding the memory — often another lab user. Run `nvidia-smi` immediately before launching, pick a GPU at ~3 MiB, launch at once.

**Bug 8 — `CUDA_VISIBLE_DEVICES=N` + `cuda:N` mismatch.**
Setting `CUDA_VISIBLE_DEVICES=3` makes that GPU appear as `cuda:0`; asking for `torch.device("cuda:3")` then fails. Rule: set the env var to the physical GPU, always reference **`cuda:0`** in code.

**Bug 9 — `cals_pitch_metric(... plot=False)` returns a float, not a pair.**
`tracking_output[0][0]` then fails with `TypeError: cannot unpack non-iterable float object`. Call it with `plot=True` so it returns `(metrics, images)` and `dtw, jaccard = tracking_output[0][0]` unpacks.

**Bug 10 — MIDI piano roll was misaligned with the audio crop. (found & fixed 2026-09-06)**

The most damaging bug so far, because nothing crashed — the conditioning was simply meaningless.

*What happened.* Two independent crops that never talked to each other:
- `random_crop_size` was passed only to **`AllTransform`**, not to `WAVDataset`. So the dataset loaded the **whole** file (`frame_offset` forced to 0) and `RandomCrop` inside the transform chain later picked a **random** 409600-sample window — then threw the offset away (`randomcrop.py:24`).
- The piano roll was cropped `piano_roll[:, :target_midi_frames]` — **always from frame 0**.

CocoChorales tracks run well past 17 s, so the offset was non-zero on essentially every sample: the model was conditioned on one part of the track while denoising a different part. The gate's rational response is to stay at zero and ignore MIDI entirely.

*Why the debug script missed it.* `debug_midi/debug_compare_audio_midi_crop.py` cropped with `waveform[:, :final_audio_samples]` under the comment `# crop like training: keep beginning`. Training does not keep the beginning. The tool validated the single offset where alignment holds by construction.

*The fix.*
1. `exp/datamodule/base.yaml` — pass `random_crop_size: ${length}` **to the dataset**, so `optimized_random_crop` chooses the window and knows the offset. (The `RandomCrop` in `transforms` stays: it is now a no-op on correctly-sized input but still zero-pads short tracks.)
2. `optimized_random_crop` returns `crop_start_sec` as a third value.
3. `__getitem__` slices `piano_roll[:, midi_start : midi_start + target_midi_frames]` where `midi_start = round(crop_start_sec * self.midi_fps)`.
4. `midi_fps` is now a constructor arg (default 75) set from `base.yaml`, instead of a bare `midi_fps = 75` inside the loop.
5. The debug script now crops from a random offset like training and slices the roll to match; set `crop_start_sec` at the top to pin a window.

Two related issues fixed at the same time:
- `validation_step` never built or passed `midi`, so `valid_loss` was measured **unconditioned** while `train_loss` was conditioned — the two were not comparable. It now mirrors `training_step`.
- `midi_resized` was only bound inside the `latent == True` branch, so a `latent: False` run would `NameError` when calling `self.model(...)`. It is now initialised to `None` before the branch.

*Consequence.* Any MIDI-conditioned checkpoint trained before 2026-09-06 was trained on noise conditioning and should be discarded.

**Bug 11 — `model.use_midi=false` crashes under DDP.**
The unconditioned baseline never passes `midi`, so the gated block in `UNet1d.forward` never runs and `midi_encoder` / `midi_gate` produce no gradient. `exp/trainer/full.yaml` sets `find_unused_parameters: False`, and DDP refuses:
`RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss`
Fix: add `trainer.strategy.find_unused_parameters=true` to that run (no `+` prefix — the key already exists). It only changes DDP's gradient-sync bookkeeping, not the math, and at `devices: 1` there is no sync happening anyway. Only the baseline needs it.

> Worth fixing properly at some point: either default that flag to `true`, or stop using DDPStrategy for single-GPU runs, where it is pure overhead.

---

## 🔬 Is the MIDI conditioning actually doing anything?

Three switches exist to answer this. All default to the normal path, so ordinary runs are unaffected.

| Override | Effect |
|---|---|
| `use_midi=true` | **the master switch.** Off by default. When off, no MIDI parameters are built, nothing is loaded, nothing is passed — the model is byte-for-byte the upstream architecture |
| `midi_cache_root=<path>` | which roll cache to read. Required when `use_midi=true`; a path that does not exist now raises at startup instead of silently serving zero rolls |
| `midi_mode=multiscale\|single` | conditioning architecture. `multiscale` (default) injects at every resolution; `single` is the original one-shot injection, kept for comparison |
| `midi_gate_init=<float>` | starting value for the gates. `0.0` (default) means the model begins identical to the unconditioned one |
| `datamodule.dataset.midi_shuffle=true` | **control**: pair each clip with a roll from a *different* track. Same format, same density, wrong notes. The window is drawn from within that track's own length, so a short track does not come back mostly zeros — that would be the no-MIDI condition, not the wrong-MIDI one |

### The architecture, and why it was changed

The original scheme injected MIDI **once**, right after `to_in` at length 1280, through a single scalar gate. From there the signal had to survive five downsampling stages to the bottleneck at length 5 — a **256× compression** — with nothing re-introducing it. That is close to the weakest conditioning mechanism available, and a negative result from it says nothing about MIDI conditioning in general.

`midi_mode=multiscale` injects at **every** resolution:

```
shared trunk (Conv-SiLU-Conv, 128 pitches -> 128ch) runs once at full length
  then per level i:  x_i = x_i + gate_i * proj_i(pool(trunk_out, len(x_i)))
```

Six injection points, channel widths `[128, 256, 512, 1024, 1024, 1024]`, one gate each. Downsampling uses average pooling, not nearest — with rolls this dense, nearest-neighbour would discard most of the signal, while averaging preserves "how much of this window is sounding". `train/midi_gate_L0..L5` shows which scales the model actually finds notes useful at.

### The two metrics

- **`train/midi_gate`** — the learned scalar, starts at 0. Movement means the model is turning the channel up.
- **`train/midi_rel_magnitude`** — `norm(gate * midi_features) / norm(x)`. **This is the one that matters.** The gate alone is uninterpretable because the scale of `midi_encoder`'s output is unknown: a gate of 0.2 on a tiny feature is still no conditioning.

### Findings so far (5-epoch runs, 2026-09-06)

Cello and bassoon, ground truth vs basic-pitch, no fifth, ~2,500 steps each:

- **The gate moves decisively.** Bassoon reached **+0.18** (basic-pitch) and **−0.22** (ground truth), smooth and monotonic from ~step 400, neither plateauing. The ~400-step flat start is the bootstrap: while the gate is 0, `midi_encoder` receives no gradient, so the encoder has to be nudged into usefulness by a gate that is itself only reacting to a random projection. It escapes, but slowly.
- **Sign is meaningless.** Gate and encoder are learned jointly, so `(+g, f)` and `(−g, −f)` are the same function. Compare magnitudes, not signs.
- **`train_loss` was identical between the two sources.** The MIDI source changes nothing measurable in the loss.
- **The envelope hypothesis is ruled out.** Ground truth sits at 0.99 notes/frame (strictly monophonic, essentially always sounding); basic-pitch at 1.17–1.44, stacking its hallucinated octaves into the *same* time columns (frame counts differ by only 2–5%). Both mark ~95–99% of frames active, so "is a note sounding" carries almost no information — there is nothing there for the model to learn from.
- **Leading hypothesis: the Encodec latent already encodes pitch**, so the conditioning is largely redundant. The model turns the gate up because it is not harmful, but cannot reduce the loss with information it already had.

### Findings — 40-epoch runs, single-injection architecture (2026-09-07)

Four bassoon runs, 20,000 steps each: `gt`, `bp`, `scrambled`, `nomidi`.

- **The model genuinely reads the notes.** Real MIDI drove the gate to **+0.20** (basic-pitch) and **−0.26** (ground truth). **Scrambled MIDI left it at 0.** Same architecture, same audio, same note density — the only difference was whether the notes matched the audio. That rules out the model simply exploiting spare parameters.
- **Ground truth beat basic-pitch**, |0.26| vs |0.20|, and was still growing while `bp` had plateaued.
- **No loss benefit whatsoever.** `valid_loss` at step 19999: bp 0.57531, gt 0.57534, nomidi 0.57557, scrambled 0.57561 — a spread of 0.05%, well inside run-to-run noise.

So: the model reads the notes, and gains nothing measurable from them. Two explanations remain open — the Encodec latent already encodes pitch (making the conditioning redundant), or the single-injection architecture was too weak to exploit it. `midi_mode=multiscale` exists to separate those.

**Note:** loss was never the right metric here — the paper evaluates on FAD and pitch distance. `SampleLogger.sample()` and `Run_timbre_transfer.py` both omit `midi=`, so every sample generated so far has been unconditioned, even by the conditioned models. The plumbing works (`DiffusionSampler.forward` merges kwargs into the denoise call); it has simply never been used.

---

## 🚀 Training

### Step 1 — activate env
```bash
source /home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/activate
```

### Step 2 — pick a FREE GPU
```bash
nvidia-smi
```
Choose one at ~3 MiB / 0% util and set it in `train.py` via `CUDA_VISIBLE_DEVICES`.

### Step 3 — edit the `train.py` overrides
Set `exp=`, `trainer.max_epochs`, batch size, num_workers, a **unique `exp_tag`**, and your own `loggers.wandb.save_dir`.

### Step 4 — launch in the background (no screen/tmux)
```bash
nohup /home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/python \
  /home/shared_workspace/diffusion-timbre-transfer/train.py \
  > /home/benshise/train_log.txt 2>&1 &
```
`nohup` survives disconnect; `> file 2>&1` captures stdout+stderr; `&` backgrounds it and prints the PID.

### Step 5 — monitor
```bash
tail -f /home/benshise/train_log.txt   # Ctrl+C stops watching, not training
ps aux | grep train.py
ls -lt /home/shared_workspace/diffusion-timbre-transfer/our_checkpoints/<exp_tag>/ckpts/
```
> `RichProgressBar` uses terminal control characters that do not grep well from a log file — judge progress from saved checkpoints, or from W&B.

**For a MIDI run, watch `train/midi_gate` in W&B.** Flat at ~0 after a few thousand steps means the conditioning is not helping — re-check cache fps, cache paths, and alignment before burning 100 epochs.

### Logging
`loggers: wandb` is the active default in both `cello_latent.yaml` and `bassoon_latent.yaml`:
```yaml
wandb:
  _target_: pytorch_lightning.loggers.WandbLogger
  project: "diffusion-timbre-transfer"
  entity: "diffusion-timbre-transfer-Proj"
  name: ${exp_tag}
  save_dir: ${logs_dir}
```
`wandb login` once per user (API key from wandb.ai/settings). `exp/loggers/tensorboard.yaml` still exists if you want to switch back.

### Checkpoint behaviour
`ModelCheckpoint` keeps only the **best** and the **last** checkpoint — intermediates are deleted (each is ~4.9 GB). Seeing only `epoch=NN-valid_loss=...ckpt`, `last.ckpt` and maybe `last-vN.ckpt` is normal.

---

## 🎚️ Current Checkpoints

### Phase 1 — AddFifth, 100 epochs, **no MIDI conditioning**
```
CELLO  (source):
  our_checkpoints/cello_add_fifth_2_bs16_V4/ckpts/epoch=94-valid_loss=0.653.ckpt
  100 epochs, bs 16, 8 workers - loss 1.399 @ ep1 -> 0.653 best @ ep94 - ~7-8 min/epoch

BASSOON (target):
  our_checkpoints/bassoon_add_fifth_100_bs16_V1/ckpts/epoch=99-valid_loss=0.627.ckpt
  100 epochs, bs 16, 8 workers - best 0.627 @ ep99 - ~7 min/epoch
```
These are what `Run_timbre_transfer.py` points at today.

### Phase 2 — MIDI conditioned
**None yet.** Only the 1-epoch smoke test (`bassoon_midi_addfifth_wandb_test_yuval`). The first real run is still to be launched, and must be launched **after** the Bug 10 fix.

> **On loss values:** the paper reports no training-loss baseline (it uses FAD + DPD), and we train on a modified distribution, so our loss is not comparable to it. Rough feel for this setup: >1.0 barely learning; 0.6–0.8 decent; 0.3–0.5 good; <0.3 very good. The real test is listening plus DPD/JD.

### Pre-existing repo checkpoints (in `checkpoints/`, NOT ours)
`bassoon_sigmaMax_{5,100}.ckpt`, `cello_sigmaMax_5.ckpt(.1)`, `flute_sigmaMax_{5,100}.ckpt(.1)`, `violin_sigmaMax_{5,100}.ckpt`, `trumpet_sigmaMax_5.ckpt`, `pitch_flute_sigmaMax_5.ckpt`, plus the mean/std tensors below.

---

## 📐 Mean/Std Tensors (Normalization Stats)

Precomputed dataset summaries in `checkpoints/` — nothing is trained in them.

| Instrument | Mean file | Std file |
|---|---|---|
| Bassoon | `mean_tensor_enc_bassoon.pt` | `std_tensor_enc_bassoon.pt` |
| Cello | `mean_cello.pt` ⚠️ | `std_cello.pt` ⚠️ |
| Flute | `mean_tensor_enc_flute.pt` | `std_tensor_enc_flute.pt` |
| Violin | `mean_tensor_enc_violin.pt` | `std_tensor_enc_violin.pt` |

⚠️ **Cello files are named differently** (`mean_cello.pt` / `std_cello.pt`, not `mean_tensor_enc_cello.pt`). The yaml and inference script must use those exact names.

---

## 🎯 Inference — `Run_timbre_transfer.py`

The whole `inference.ipynb` flow in one script, all config at the top: load source + target models, load input audio, (optionally) AddFifth, crop/pad to 17 s, encode, reverse-diffuse to noise (SOURCE), forward-diffuse to target (TARGET), decode, save WAV + spectrogram PNGs, print DPD/JD.

### Current CONFIG defaults
```
SOURCE = cello   (mean_cello.pt / std_cello.pt / cello_add_fifth_2_bs16_V4 epoch=94)
TARGET = bassoon (mean_tensor_enc_bassoon.pt / std_tensor_enc_bassoon.pt / bassoon_add_fifth_100_bs16_V1 epoch=99)
INPUT_AUDIO_PATH = real_cello.wav
ADD_FIFTH = False, FIFTH_GAIN = 0.8
SAMPLING_RATE = 24000, CLIP_LENGTH = 409600 (17s), NUM_STEPS = 100
SIGMA_MIN = 0.001, SIGMA_MAX = 5, RHO = 9.0   <- MATCH the training yaml exactly
```

> ⚠️ **`Run_timbre_transfer.py` has no MIDI support.** It never passes `midi=`, so `midi_gate` conditioning is skipped entirely at inference. Once a MIDI-conditioned model exists, the script needs a piano roll for the input audio (run `basic-pitch` on it, or reuse a cached roll) threaded through both `pl_model` calls — otherwise you are running a conditioned model unconditioned.

> ⚠️ `ADD_FIFTH` currently defaults to **False** while the Phase-1 checkpoints were trained **with** the fifth. Set it to `True` to match those models.

### Things baked into the script
- `CUDA_VISIBLE_DEVICES` set at the very top; code references `cuda:0` (Bug 8).
- `os.chdir(REPO_PATH)` so relative checkpoint paths resolve.
- Headless matplotlib (`Agg`) — saves PNGs instead of showing them. `SHOW_PLOTS=True` only in a notebook.
- Crop-or-pad to exactly 409600 samples: input is often 32 s, and a naive `pad(0, 409600 - len)` would compute a negative pad.
- Saves the cropped INPUT too, so you can hear what the model actually received.
- Pitch metric called with `plot=True` (Bug 9).

### Run it
```bash
mkdir -p /home/shared_workspace/diffusion-timbre-transfer/Outputs/WAV_files/png_files
/home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/python \
  /home/shared_workspace/diffusion-timbre-transfer/Run_timbre_transfer.py
```

### Metrics
- **DPD** (pitch distance via DTW): lower = better melody preservation.
- **JD** (Jaccard distance): pitch-set overlap distance.
- These plus FAD are the real evaluation — not training loss.

---

## 🔍 Debugging the MIDI pipeline

```bash
# One track: spectrogram vs piano roll for the SAME random window.
python debug_midi/debug_compare_audio_midi_crop.py
```
Edit `wav_path` / `midi_path` at the top. Leave `crop_start_sec = None` for a random window (what training does), or set a number to re-inspect a specific one. It writes:
- `debug_audio_midi_crop_compare.png` — spectrogram over piano roll; the note contours should line up
- `debug_audio_training_crop.wav` — the audio the model sees
- `debug_midi_crop_synth.wav` — the piano roll rendered as sines; play it against the audio, they should agree
- `debug_audio_crop_exported.mid`

It also prints how far the aligned slice differs from the old frame-0 slice — a large percentage is just Bug 10 being visible, not a new fault.

---

## 🛠️ Environment gotchas quick list
- `screen` ❌, `tmux` ❌ → use `nohup`. `sudo` ❌.
- 8× V100-32GB, indices 0–7. Set `CUDA_VISIBLE_DEVICES` before torch import; reference `cuda:0` in code.
- `torchaudio.save` / `matplotlib.savefig` do not create dirs → `mkdir -p` first.
- Setgid bit on `our_checkpoints` → unique `exp_tag` per run.
- `RichProgressBar` does not grep from logs → use checkpoints or W&B.
- "Modules in train mode: 1270" (vs 1269) confirms AddFifth is registered.
- **MIDI cache fps must be 75** and must match `midi_fps` in `base.yaml`; the cache scripts default to 50.

### Known rough edges (not bugs, just things to know)
- `requirements.txt` is missing **`pretty_midi`** and **`basic-pitch`**, both required by `midi_preprocessing/`. Install them manually in the venv.
- `.gitignore` names only some cache dirs by path (`midi_cache_cello/`, `midi_cache_cello_fps75/`, `midi_cache_test/`, `midi_outputs/`) — `midi_cache_cello_fps75_addfifth/` and both `midi_cache_bassoon_*` are missing. In practice the blanket `*.npy` and `*.mid` rules already cover everything inside them, so nothing gets staged today. Worth adding the directory names anyway, so a stray `.txt` or `.json` in a cache dir can't slip through.
- Cache roots are **hardcoded absolute paths** in `WAVDataset.get_midi_cache_path()` and in `create_addfifth_pianoroll_cache.py` (module constants). Not configurable from Hydra yet.
- MIDI is interpolated twice (`training_step` to latent length, then `UNet1d.forward` to `x` length), both `nearest`. Works, but the first is redundant.

---

## ✅ Next Steps / Open Items
1. ~~Setup env & deps~~ ✅
2. ~~mean/std tensors for cello & bassoon~~ ✅
3. ~~Configure cello/bassoon yamls~~ ✅
4. ~~Implement & debug AddFifth (polyphony)~~ ✅
5. ~~Train cello + bassoon 100 epochs WITH AddFifth~~ ✅ (0.653 / 0.627)
6. ~~Build single-file inference script~~ ✅
7. ~~Set up W&B logging~~ ✅ (active logger in both exp yamls)
8. ~~Build MIDI piano-roll caches (cello + bassoon, fps75, +fifth)~~ ✅
9. ~~Wire MIDI conditioning into dataset + UNet~~ ✅
10. ~~Fix MIDI/audio crop misalignment~~ ✅ (Bug 10, 2026-09-06)
11. **Run a short MIDI training sanity run** — a few thousand steps, watch `train/midi_gate` move off zero. This is the go/no-go for the whole feature.
12. **Train cello + bassoon with MIDI conditioning** (100 epochs each) once 11 looks right.
13. **Add MIDI support to `Run_timbre_transfer.py`** — otherwise the conditioned models run unconditioned at inference.
14. **Evaluate**: DPD / JD against the Phase-1 AddFifth-only baseline, plus listening.
15. Housekeeping: add `pretty_midi` + `basic-pitch` to `requirements.txt`; name the remaining cache dirs in `.gitignore`; delete the old `midi_outputs/cache_temp_midis*` trees (see below — the scripts no longer create them).

---

## 🗑️ On-disk artefacts that are NOT in git

The container holds a lot more than the repo. What is actually there, and what it is for:

### Piano-roll caches — the `.npy` files training reads
| Directory | Status |
|---|---|
| `midi_cache_cello_fps75_addfifth/` | ✅ **live** — `get_midi_cache_path()` reads this |
| `midi_cache_bassoon_fps75_addfifth/` | ✅ **live** — same |
| `midi_cache_cello_fps75/` | keep — the source `create_addfifth_pianoroll_cache.py` reads |
| `midi_cache_bassoon_fps75/` | keep — same |
| `midi_cache_cello/` | ⚠️ **stale, 50 fps** — first run before the fps was raised. Nothing reads it; delete it so nobody points at it by mistake |
| `midi_cache_test/` | scratch from a `--limit` run |

### `midi_outputs/` — intermediate `.mid`, safe to delete
These are left over from runs made **before** the cache scripts learned to clean up after themselves. `create_*_midi_cache.py` used to run `basic-pitch` into `<temp_midi_root>/<track_name>/` and never delete it — one directory per track, kept forever:

| Directory | Roughly |
|---|---|
| `cache_temp_midis/` | ~11k dirs — first cello run |
| `cache_temp_midis_fps75/` | ~11k dirs — second cello run |
| `cache_temp_midis_bassoon_fps75/` | ~10k dirs — bassoon run |
| `cache_test_midis/`, `basic_pitch_test/`, `basic_pitch_fifth_test/` | one-off experiments |
| `real_flute_*pianoroll*.npy` | one-off rolls of the sample flute wav |

> **The two cello temp trees hold identical MIDI.** `fps` is applied when the roll is *rendered* (`midi_to_pianoroll`), not when basic-pitch transcribes — so re-running at a new fps re-transcribed ~11k files for nothing. This is now handled: the script reuses an existing `.mid` instead of re-transcribing, and deletes it afterwards unless `--keep_midi` is passed.

Once the `.npy` cache exists, the whole existing `midi_outputs/` tree is dead weight — tens of thousands of inodes on a shared lab filesystem. Safe to delete:

```bash
cd /home/shared_workspace/diffusion-timbre-transfer
du -sh midi_outputs/* midi_cache_*          # look before you leap
rm -rf midi_outputs/cache_temp_midis midi_outputs/cache_temp_midis_fps75
rm -rf midi_outputs/cache_temp_midis_bassoon_fps75 midi_outputs/cache_test_midis
rm -rf midi_cache_cello midi_cache_test     # stale 50 fps + scratch
```

### Environments
`venv_yuval/` is the documented env. `miniforge3/` is a separate conda install in the workspace (both gitignored). If a MIDI script and training disagree about a package version, check which interpreter is actually on `PATH` — always invoke the venv's python by full path.

---

## 🔗 Reference
- **Upstream repo:** https://github.com/sony/diffusion-timbre-transfer
- **Paper:** https://arxiv.org/abs/2409.06096
- **Demo page:** https://sony.github.io/diffusion-timbre-transfer/
- **Pretrained weights + mean/std tensors:** https://zenodo.org/records/13849169
- **basic-pitch (MIDI transcription):** https://github.com/spotify/basic-pitch
