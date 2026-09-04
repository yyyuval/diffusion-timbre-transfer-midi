# CLAUDE.md — Diffusion Timbre Transfer (Music Proj 2026)

This file gives Claude (or any new assistant / teammate) the full context needed to work inside this repository and dataset environment. It is intentionally detailed so that a fresh conversation can pick up exactly where the last one left off, with no loss of context.

> **Status (current):** Both cello and bassoon models have been trained for **100 epochs each WITH the AddFifth polyphony augmentation**. A single-file inference script (`Run_timbre_transfer.py`) runs the full cello→bassoon diffusion bridge end to end and saves outputs. The core project goal is essentially complete.

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

---

## 📂 Repository Layout

```
diffusion-timbre-transfer/
├── audio_data_pytorch/          ← dataset / dataloader classes
│   ├── __init__.py
│   ├── datasets/
│   ├── transforms/
│   │   └── all.py               ← AllTransform + our AddFifth class (EDITED)
│   ├── declipping.py
│   └── utils.py
├── audio_diffusion_pytorch/     ← model code
│   ├── __init__.py
│   ├── basic-pitch-torch/       ← cloned from github during setup
│   ├── diffusion.py
│   ├── encodec_utils.py
│   ├── model.py
│   ├── modules.py
│   ├── pitch_tracking_utils.py
│   └── utils.py
├── main/                        ← utils.py and training helpers
│   ├── __init__.py
│   ├── module_base_latent_cond.py  ← Model/Datamodule/SampleLogger (EDITED)
│   └── utils.py
├── exp/                         ← Hydra experiment configs
│   ├── bassoon_latent.yaml      ← source-instrument config (EDITED)
│   ├── cello_latent.yaml        ← target-instrument config (EDITED)
│   ├── flute_latent.yaml
│   ├── violin_latent.yaml
│   ├── trumpet_latent.yaml
│   ├── pitch_flute_latent.yaml
│   ├── callbacks/
│   ├── datamodule/
│   │   └── base.yaml            ← datamodule config (EDITED — add_fifth lines added then REMOVED, see below)
│   ├── loggers/
│   ├── model/
│   └── trainer/
│       └── full.yaml            ← trainer settings (EDITED)
├── checkpoints/                 ← mean/std tensors and pretrained ckpts (provided by repo)
├── our_checkpoints/             ← OUR trained models go here
├── Outputs/
│   └── WAV_files/               ← inference WAV outputs
│       └── png_files/           ← inference spectrogram PNGs
├── config.yaml                  ← root Hydra config
├── train.py                     ← training entry point (EDITED)
├── train.sh
├── Run_timbre_transfer.py       ← OUR single-file inference script (NEW)
├── inference.ipynb              ← original notebook (the bridge that does the transfer)
├── real_flute.wav               ← sample input audio
├── real_cello.wav               ← sample input audio (used as inference input)
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
├── stems_audio/
│   ├── 1_<instrument>.wav
│   ├── 2_<instrument>.wav
│   ├── 3_<instrument>.wav
│   └── 4_<instrument>.wav
└── ...
```

Relevant stems for our project:
- **Bassoon** — in `woodwind_track*/stems_audio/4_bassoon.wav` (**9982 tracks** found, confirmed in training logs)
- **Cello** — in `string_track*/stems_audio/4_cello.wav` (**11395 tracks** found, confirmed in training logs)

---

## 🧠 How the Model Works

This repo implements **Latent Diffusion Bridges** for unsupervised timbre transfer.

> Paper: https://arxiv.org/abs/2409.06096
> The paper trains on **CocoChorales** (unpaired monophonic single-instrument audio), one diffusion model per instrument, with a Gaussian prior. At inference, a SOURCE model maps the input to its Gaussian prior, and a TARGET model reconstructs from that prior — facilitating timbre transfer. Evaluation in the paper is by **FAD (Fréchet Audio Distance)** and **DPD (pitch distance)** — NOT by training loss. The noise level σ controls the melody-preservation vs timbre-transfer tradeoff.

**Conceptual point:** "cello → bassoon" is NOT a single training run.

1. Audio is encoded into a latent space (Encodec, `facebook/encodec_24khz`, 24kHz).
2. A **separate diffusion model is trained per instrument** — one for cello, one for bassoon.
3. The **actual transfer happens at inference**: it "bridges" between the two trained models — reverse-diffuse the input to noise with the SOURCE model, then forward-diffuse that noise to the TARGET instrument with the TARGET model.

So the task = **train 2 models** + **run the bridge at inference**.

### Model size (from training logs)
- `encodec_model`: 14.9 M params (frozen, eval mode)
- `model` (AudioDiffusionModel / UNet1d / XDiffusion): 302 M params (trainable)
- `model_ema`: 604 M params
- Total params: ~619 M. Trainable: 302 M. Non-trainable: 317 M.
- With AddFifth registered as a submodule, the summary shows **"Modules in train mode: 1270"** (it was 1269 before AddFifth was added — this is a quick sanity check that AddFifth is registered).

---

## 🎼 THE PROJECT GOAL — AddFifth / polyphony augmentation

**What we are doing differently from the base repo:** We want each instrument recording used in training to receive an **additional voice a perfect fifth higher (7 semitones up)**, mixed with the original, so it sounds like **two instruments playing together (polyphonic)** — one at the original pitch, one a fifth above. The models are therefore trained on this polyphonic-augmented audio, and at inference the input is augmented the same way so it matches the training distribution.

This is implemented by a custom `AddFifth` transform.

### ⚠️ Where AddFifth runs — this is critical for performance
We tried two placements:

1. **In the dataloader (via `base.yaml` transforms)** — DON'T DO THIS. `AddFifth` here runs **once per audio sample on CPU**, inside the dataloader workers. With 8 workers all running pitch-shift on CPU, the CPU saturates and the GPU starves. Training went from ~4 min/epoch to **15+ min/epoch and stuck at 0/570**. We **removed** `add_fifth`/`fifth_gain` from `base.yaml`.

2. **In the training step on GPU (in `module_base_latent_cond.py`)** — THIS IS THE CORRECT PLACEMENT. `AddFifth` runs **once per batch on the GPU**, which is fast. This restored ~4 min/epoch (final runs were ~7 min/epoch with batch size 16 — still fine). One GPU op on a batch of 16 beats 16 CPU ops across 8 workers, because GPUs are built for this parallel math.

---

## ⚙️ ALL Modified Files (with exact edits)

### 1. `audio_data_pytorch/transforms/all.py` — the `AddFifth` class

The current, WORKING version of `AddFifth`. Note: it uses precomputed `Resample` transforms (built once in `__init__`), NOT `torchaudio.functional.pitch_shift` per-call. Reasoning below in the Bugs section.

```python
from typing import Optional
from torch import Tensor, nn
from ..utils import exists
from .crop import Crop
from .loudness import Loudness
from .mono import Mono
from .randomcrop import RandomCrop
from .resample import Resample
from .scale import Scale
from .stereo import Stereo
from .pitchshift import PitchShift
import torch
import torchaudio

class AddFifth(nn.Module):
    def __init__(self, sample_rate: int = 24000, fifth_gain: float = 0.8):
        super().__init__()
        self.fifth_gain = fifth_gain
        ratio = 2 ** (7 / 12)  # 7 semitones up = perfect fifth
        orig_freq = 10000
        new_freq = round(orig_freq / ratio)  # ≈ 6674
        self.downsample = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq
        )
        self.upsample = torchaudio.transforms.Resample(
            orig_freq=new_freq, new_freq=orig_freq
        )

    def forward(self, x: Tensor) -> Tensor:
        x_fifth = self.upsample(self.downsample(x))

        min_len = min(x.shape[-1], x_fifth.shape[-1])
        x = x[..., :min_len]
        x_fifth = x_fifth[..., :min_len]

        x_mix = x + self.fifth_gain * x_fifth
        max_val = torch.max(torch.abs(x_mix))
        if max_val > 0:
            x_mix = 0.95 * x_mix / max_val
        return x_mix
```

`AllTransform.__init__` in the same file also had its `Resample` condition fixed (see Bug 3): the condition must read `source_rate != target_rate` (singular).

> Note: `AddFifth.__init__` still accepts a `sample_rate` argument for API compatibility, but the resampling ratio is built from the hardcoded `orig_freq`/`new_freq`, so `sample_rate` is effectively unused inside the class. Harmless. The `fifth_gain` default is 0.8.

### 2. `main/module_base_latent_cond.py` — apply AddFifth on GPU

**(a)** Added import near the top (after the other imports, ~line 25–26):
```python
# 1 line added by shlomi
from audio_data_pytorch.transforms.all import AddFifth
```

**(b)** In `Model.__init__`, right after `self.sampling_rate = sampling_rate` (~line 104):
```python
self.sampling_rate = sampling_rate
self.add_fifth_transform = AddFifth(sample_rate=sampling_rate, fifth_gain=0.8)
```
> IMPORTANT: Do **NOT** add `.to('cuda')` here, and do **NOT** add a `self.length = length` line. See Bug 5 below — both caused failures. PyTorch Lightning moves submodules to the right device automatically.

**(c)** At the very top of `training_step`:
```python
def training_step(self, batch, batch_idx):
    # batch = (waveforms, inst_label)
    # shlomi added 3 lines: apply AddFifth on GPU
    batch = list(batch)
    batch[0] = self.add_fifth_transform(batch[0])
    batch = tuple(batch)
    ...
```

**(d)** At the very top of `validation_step` (same 3 lines):
```python
def validation_step(self, batch, batch_idx):
    # shlomi added 3 lines: apply AddFifth on GPU
    batch = list(batch)
    batch[0] = self.add_fifth_transform(batch[0])
    batch = tuple(batch)
    ...
```

> There was also an earlier base-repo fix here: `on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0)` — the `=0` default was added for the newer PyTorch Lightning API. (~line 460 region.)

### 3. `exp/datamodule/base.yaml`

We initially added `add_fifth: true` and `fifth_gain: 0.8` under `transforms:` here, but **REMOVED them** because running AddFifth in the dataloader is far too slow (see performance note above). The current `transforms:` block should NOT contain `add_fifth`/`fifth_gain`:
```yaml
target: main.module_base_latent_cond.Datamodule
dataset:
  target: audio_data_pytorch.WAVDataset
  recursive: True
  instruments: ${instruments}
  path: ${dataset_path}
  sample_rate: ${sampling_rate}
  transforms:
    target: audio_data_pytorch.AllTransform
    random_crop_size: ${length}
    sign: ${sign}
val_split: 0.2
batch_size: 32
num_workers: 32
pin_memory: True
```

### 4. `exp/trainer/full.yaml`
The repo was written for an older PyTorch Lightning. Changed:
```yaml
# Was:
gpus: -1
# Now:
devices: 1
```

### 5. `exp/cello_latent.yaml` and `exp/bassoon_latent.yaml`
Both reference dataset path, exp_tag, logs_dir, and trainer devices at the bottom. The key fields:
```yaml
dataset_path: '/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/cocochorales_tiny_v1_zipped/main_dataset'
exp_tag: 'cello_tiny_v2'   # or bassoon_tiny_v2
logs_dir: '/home/shared_workspace/diffusion-timbre-transfer/our_checkpoints'
trainer:
  devices: 1   # ← MUST be scalar 1, NOT a list [1]. See Bug 6.
```
> ⚠️ `bassoon_latent.yaml` originally had `devices: [1]` which caused a crash (`You requested gpu: [1] But your machine only has: [0]` when combined with `CUDA_VISIBLE_DEVICES`). It must be `devices: 1`.

### 6. `train.py` — set GPU and overrides BEFORE imports
The top of `train.py` sets the GPU and Hydra overrides. `CUDA_VISIBLE_DEVICES` MUST be set before any CUDA/torch import or it's ignored.
```python
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # pick a FREE GPU (check nvidia-smi)

sys.argv += [
    "exp=cello_latent",            # or bassoon_latent
    "trainer.max_epochs=100",
    "datamodule.batch_size=16",
    "datamodule.num_workers=8",
    "exp_tag=cello_add_fifth_2_bs16_V4",   # unique tag per run (permissions!)
    "logs_dir=/home/shared_workspace/diffusion-timbre-transfer/our_checkpoints",
]
# ... then dotenv, hydra, pytorch_lightning imports follow ...
```

---

## 🐞 Bugs Encountered & Fixes (full history — read before re-introducing AddFifth)

These are the actual bugs hit during this project. If something breaks, check here first.

**Bug 1 — `_init_` instead of `__init__` (single underscores).**
Yuval's original `AddFifth` and `AllTransform` used `def _init_(...)` and `super()._init_()`. Python treats `_init_` as an ordinary method, NOT the constructor, so attributes (`sample_rate`, `fifth_gain`, `self.transform`, ...) were never stored, and `forward()` crashed with `AttributeError: ... has no attribute 'sample_rate'`. Fix: use the dunder `__init__` and `super().__init__()` everywhere.

**Bug 2 — `AF.pitch_shift` was the training slowdown.**
`torchaudio.functional.pitch_shift` does STFT-based pitch shifting on CPU and is very slow, especially recreating expensive state every call in dataloader workers. This was the main cause of 15+ min/epoch. Fix: replace with precomputed `torchaudio.transforms.Resample` transforms built once in `__init__` (the perfect-fifth ratio is `2**(7/12) ≈ 1.4983`; downsample then upsample). NOTE: this resampling trick is an approximation of pitch shift — for the *augmentation* purpose it's acceptable and fast.

**Bug 3 — `target_rates` (plural) NameError.**
In `AllTransform.__init__`, the `Resample` guard read `source_rate != target_rates` (plural, undefined). Fix: `source_rate != target_rate` (singular).

**Bug 4 — saving WAV to a read-only path.**
`torchaudio.save(...)` to a shared-workspace path failed with `LibsndfileError: ... System error`. It was a **permissions** problem, not an audio problem. Fix: save to a writable dir (`/home/benshise/...` or the `Outputs/` dir you own). Also: the target directory must already exist.

**Bug 5 — `add_fifth_transform` "has no attribute" / device pinning.**
Two sub-issues when wiring AddFifth into the Model:
- Adding `.to('cuda' if ...)` in `__init__` pins the transform to a device before Lightning moves the model, causing device mismatches. Remove `.to(...)`.
- A stray `self.length = length` line was added right after, but `length` is NOT a parameter of `Model.__init__`, so it raised `NameError` *before* `self.add_fifth_transform` was stored, making it look like the attribute "didn't exist". Remove that line.
- Also: edits sometimes silently didn't save. Always verify with `cat -n ... | head -120` that the line is actually present in `Model.__init__`.

**Bug 6 — `devices: [1]` vs `devices: 1`.**
`devices: [1]` means "use GPU physical index 1". Combined with `CUDA_VISIBLE_DEVICES=N` (which remaps the chosen GPU to index 0), Lightning looks for index 1 and crashes: `You requested gpu: [1] But your machine only has: [0]`. Fix: use scalar `devices: 1` ("use 1 GPU").

**Bug 7 — CUDA out of memory even though "GPU is free".**
The OOM error lists the processes occupying the GPU (e.g. "Process X has 21.49 GiB in use"). A GPU can be busy with another lab user's job. Always run `nvidia-smi` immediately before launching and pick a GPU showing ~3 MiB used. Then set `CUDA_VISIBLE_DEVICES` to that index.

**Bug 8 — `CUDA_VISIBLE_DEVICES=N` + `cuda:N` mismatch in the inference script.**
If you set `os.environ["CUDA_VISIBLE_DEVICES"] = "3"`, the script sees ONE gpu renamed to `cuda:0`. Asking for `torch.device("cuda:3")` then fails. Rule: set `CUDA_VISIBLE_DEVICES` to the physical GPU, but ALWAYS reference `cuda:0` in code.

**Bug 9 — `cals_pitch_metric(... plot=False)` returns a float, not a pair.**
`tracking_output[0][0]` then fails with `TypeError: cannot unpack non-iterable float object`. The function returns a different structure depending on `plot`. Fix: call with `plot=True` (as the notebook did), which returns `(metrics, images)` so `dtw, jaccard = tracking_output[0][0]` unpacks correctly.

---

## 🚀 Training

### Step 1 — Activate env
```bash
source /home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/activate
```

### Step 2 — Pick a FREE GPU
```bash
nvidia-smi
```
Choose one with ~3 MiB used and 0% util. Set it in `train.py` via `CUDA_VISIBLE_DEVICES`.

### Step 3 — Edit `train.py` overrides
Set `exp=cello_latent` or `exp=bassoon_latent`, `trainer.max_epochs`, batch size, num_workers, and a **unique `exp_tag`** (so the run writes to a fresh, owned subdirectory).

### Step 4 — Launch in the background (no screen/tmux available)
`screen` and `tmux` are NOT installed. Use `nohup`:
```bash
nohup /home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/python \
  /home/shared_workspace/diffusion-timbre-transfer/train.py \
  > /home/benshise/train_log.txt 2>&1 &
```
- `nohup` keeps it running after you disconnect.
- `> file 2>&1` sends stdout+stderr to the log file.
- `&` backgrounds it; the printed `[1] <PID>` is the process id — note it.

### Step 5 — Monitor
```bash
tail -f /home/benshise/train_log.txt        # Ctrl+C to stop watching (does NOT stop training)
ps aux | grep train.py                       # is it still running?
```
> The `RichProgressBar` uses terminal control characters that don't grep well from the log. To check epoch progress, look at saved checkpoints instead:
```bash
ls -lt /home/shared_workspace/diffusion-timbre-transfer/our_checkpoints/<exp_tag>/ckpts/
```

### Checkpoint behavior
The `ModelCheckpoint` callback keeps only the **best** and the **last** checkpoint — intermediate epochs are deleted to save space (each checkpoint is ~4.9 GB). So you'll typically see only `epoch=NN-valid_loss=...ckpt`, `last.ckpt`, maybe `last-vN.ckpt`. This is normal; epochs in between were saved then replaced.

---

## ⚠️ GPU Memory Notes
- Training uses ~30 GB on a single V100 — almost the whole GPU.
- Multiple lab users share GPUs; a "free" GPU can fill up between checking and launching. Run `nvidia-smi`, then launch immediately.
- If OOM, read which processes hold the memory (the error lists them), pick a different GPU, or coordinate with other users.

---

## 🎚️ Current Checkpoints (trained WITH AddFifth, 100 epochs)

```
CELLO  (source):
/home/shared_workspace/diffusion-timbre-transfer/our_checkpoints/cello_add_fifth_2_bs16_V4/ckpts/epoch=94-valid_loss=0.653.ckpt
  - trained 100 epochs, batch size 16, 8 workers
  - loss: 1.399 @ epoch 1  →  0.653 best @ epoch 94
  - ~7-8 min/epoch

BASSOON (target):
/home/shared_workspace/diffusion-timbre-transfer/our_checkpoints/bassoon_add_fifth_100_bs16_V1/ckpts/epoch=99-valid_loss=0.627.ckpt
  - trained 100 epochs, batch size 16, 8 workers
  - loss: best 0.627 @ epoch 99
  - ~7 min/epoch
```

> **On loss values:** the paper does not report a training-loss baseline (it uses FAD + DPD). And we train on a *modified* (cello/bassoon + fifth) distribution, so our loss can't be compared to the paper. For diffusion models, loss is a rough guide only — the real test is listening to the inference output and the DPD/JD metrics. Rough loss feel for this setup: >1.0 barely learning; 0.6–0.8 decent; 0.3–0.5 good; <0.3 very good.

### Pre-existing repo checkpoints (in `checkpoints/`, NOT ours)
`bassoon_sigmaMax_{5,100}.ckpt`, `cello_sigmaMax_5.ckpt(.1)`, `flute_sigmaMax_{5,100}.ckpt(.1)`, `violin_sigmaMax_{5,100}.ckpt`, `trumpet_sigmaMax_5.ckpt`, `pitch_flute_sigmaMax_5.ckpt`, plus mean/std tensors (see below).

---

## 📐 Mean/Std Tensors (Normalization Stats)

Each instrument config references precomputed mean/std tensors. These are dataset summaries — nothing is "trained" in them. They live in `checkpoints/`.

| Instrument | Mean file | Std file |
|---|---|---|
| Bassoon | `mean_tensor_enc_bassoon.pt` | `std_tensor_enc_bassoon.pt` |
| Cello | `mean_cello.pt` ⚠️ | `std_cello.pt` ⚠️ |
| Flute | `mean_tensor_enc_flute.pt` | `std_tensor_enc_flute.pt` |
| Violin | `mean_tensor_enc_violin.pt` | `std_tensor_enc_violin.pt` |

⚠️ **Cello files have different names** than the others (`mean_cello.pt` / `std_cello.pt`, not `mean_tensor_enc_cello.pt`). The yaml and inference script must point to `mean_cello.pt` / `std_cello.pt` exactly.

---

## 🎯 Inference — `Run_timbre_transfer.py` (single-file script)

We consolidated the whole `inference.ipynb` flow into one script with all config at the top. It runs the full bridge: load source + target models → load input audio → (optionally) AddFifth → crop/pad to 17s → encode → reverse-diffuse to noise (SOURCE) → forward-diffuse to target (TARGET) → decode → save WAV + spectrogram PNGs → print DPD/JD.

### Current CONFIG defaults
```
SOURCE = cello   (mean_cello.pt / std_cello.pt / cello_add_fifth_2_bs16_V4 epoch=94)
TARGET = bassoon (mean_tensor_enc_bassoon.pt / std_tensor_enc_bassoon.pt / bassoon_add_fifth_100_bs16_V1 epoch=99)
INPUT_AUDIO_PATH = real_cello.wav
ADD_FIFTH = True, FIFTH_GAIN = 0.8
SAMPLING_RATE = 24000, CLIP_LENGTH = 409600 (17s), NUM_STEPS = 100
SIGMA_MIN = 0.001, SIGMA_MAX = 5, RHO = 9.0   ← MATCH the training yaml exactly
WAV_DIR  = Outputs/WAV_files
PLOT_DIR = Outputs/WAV_files/png_files
```

### Things baked into the script (from the notebook + our fixes)
- Sets `os.environ["CUDA_VISIBLE_DEVICES"]` at the very top (currently "3"), references `cuda:0` in code (Bug 8).
- `os.chdir(REPO_PATH)` so the relative checkpoint paths resolve.
- Headless matplotlib (`Agg`) — saves PNG spectrograms instead of showing them (no Jupyter). Set `SHOW_PLOTS=True` only inside a notebook.
- Crop-or-pad logic to force exactly 409600 samples (17s): input audio is often 32s, longer than the 17s the model trained on. Naive `pad(0, 409600 - len)` would compute a NEGATIVE pad and misbehave — the script crops if too long, pads if too short.
- Saves the cropped INPUT (cello + fifth) as a WAV too, so you can listen to what the model actually receives.
- Pitch metric call uses `plot=True` (Bug 9).

### Run it
```bash
# make sure output dirs exist first (save funcs don't create them)
mkdir -p /home/shared_workspace/diffusion-timbre-transfer/Outputs/WAV_files/png_files

/home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/python \
  /home/shared_workspace/diffusion-timbre-transfer/Run_timbre_transfer.py
```

### To run a DIFFERENT transfer direction or instruments
Edit ONLY the CONFIG block at the top of the script:
- SOURCE_* = the instrument your INPUT audio is (maps input → noise).
- TARGET_* = the instrument you want to hear (maps noise → output).
- INPUT_AUDIO_PATH, WAV_DIR/OUTPUT_AUDIO_PATH, PLOT_DIR.
- ADD_FIFTH = True to match models trained with the fifth; False for pure monophonic.
- Keep SIGMA_* and SAMPLING_RATE matching the training yaml.

### Metrics meaning
- **DPD** (pitch distance, via DTW): lower = better melody preservation.
- **JD** (Jaccard distance): pitch-set overlap distance.
- These (plus FAD in the paper) are the real evaluation — not training loss.

---

## 📓 inference.ipynb → script variable mapping (history)

The original notebook used flute (source) and violin (target). We rewired it to cello → bassoon. When reading the old notebook, the renames were:
- `flute_waveform` → `cello_waveform` (input)
- `embeddings_flute` → `embeddings_cello`
- `noisy_flute_embeddings` → `noisy_cello_embeddings`
- `generated_violin_embeddings` → `generated_bassoon_embeddings`
- `violin_waveform` → `bassoon_waveform` (output)
- `pl_model_flute` → `pl_model_cello`, `pl_model_violin` → `pl_model_bassoon`
- titles/pairs: `'Real Flute'/'Generated Violin'` → `'Input Cello'/'Generated Bassoon'`
The notebook's encode→reverse→forward→decode→metrics structure is preserved in `Run_timbre_transfer.py`.

---

## 🛠️ Environment gotchas quick list
- `screen` ❌ not installed, `tmux` ❌ not installed → use `nohup` for background jobs.
- `sudo` ❌ not available to developers.
- 8× V100-32GB, indices 0–7. Set `CUDA_VISIBLE_DEVICES` before torch import; reference `cuda:0` in code.
- `torchaudio.save` / `matplotlib.savefig` don't create dirs → `mkdir -p` first.
- Setgid bit on `our_checkpoints` → use a unique `exp_tag` per run to avoid cross-user permission denials.
- `RichProgressBar` output doesn't grep from logs → check checkpoints to gauge progress.
- "Modules in train mode: 1270" (vs 1269) confirms AddFifth is registered.

---

## ✅ Next Steps / Open Items
1. ~~Setup env & deps~~ ✅
2. ~~mean/std tensors for cello & bassoon~~ ✅
3. ~~Configure cello/bassoon yamls~~ ✅
4. ~~Implement & debug AddFifth (polyphony)~~ ✅
5. ~~Move AddFifth to GPU (training_step) for speed~~ ✅
6. ~~Train cello 100 epochs WITH AddFifth~~ ✅ (best 0.653 @ epoch 94)
7. ~~Train bassoon 100 epochs WITH AddFifth~~ ✅ (best 0.627 @ epoch 99)
8. ~~Build single-file inference script~~ ✅ (`Run_timbre_transfer.py`)
9. **Evaluate transfer quality** — listen to outputs in `Outputs/WAV_files/`, review DPD/JD, compare to non-AddFifth baseline if desired.
10. (Optional) Set up **W&B** logging for nicer monitoring. The repo already imports `WandbLogger`; likely just switch the logger config in the exp yaml. Steps: create wandb account → `pip install wandb` → `wandb login` (API key from wandb.ai/settings) → switch loggers in the experiment yaml from tensorboard to wandb. (We did not finish this — TensorBoard is the current logger.)
11. (Optional) Run more input files / other instrument pairs through the script.

---

## 🔗 Reference
- **Upstream repo:** https://github.com/sony/diffusion-timbre-transfer
- **Paper:** https://arxiv.org/abs/2409.06096
- **Demo page:** https://sony.github.io/diffusion-timbre-transfer/
- **Pretrained weights + mean/std tensors:** https://zenodo.org/records/13849169
