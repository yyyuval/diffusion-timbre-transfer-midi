"""
=============================================================================
  TIMBRE TRANSFER — single-file inference script
=============================================================================
  Runs the full diffusion bridge from the notebook:
    1. load source + target models
    2. load input audio, (optionally) add a polyphonic fifth, crop/pad to 17s
    3. encode input -> reverse-diffuse to noise (SOURCE model)
    4. forward-diffuse noise -> target embeddings (TARGET model)
    5. decode to audio, save it, and print the DPD / JD pitch metrics

  TO USE: edit ONLY the CONFIG block below, then run:
      python run_timbre_transfer.py
=============================================================================
"""

# ============================================================================
#  CONFIG  — this is the only part you normally need to edit
# ============================================================================

REPO_PATH = "/home/shared_workspace/diffusion-timbre-transfer"

# ---- SOURCE model (the instrument your INPUT audio is) ---------------------
# This model maps the input audio -> noise.
SOURCE_NAME        = "cello"     # label, used only for printouts / titles
SOURCE_MEAN_PATH   = "checkpoints/mean_cello.pt"
SOURCE_STD_PATH    = "checkpoints/std_cello.pt"
SOURCE_CKPT        = "our_checkpoints/cello_add_fifth_2_bs16_V4/ckpts/epoch=94-valid_loss=0.653.ckpt"

# ---- TARGET model (the instrument you want the OUTPUT to sound like) --------
# This model reconstructs target audio from the noise.
TARGET_NAME        = "bassoon"
TARGET_MEAN_PATH   = "checkpoints/mean_tensor_enc_bassoon.pt"
TARGET_STD_PATH    = "checkpoints/std_tensor_enc_bassoon.pt"
TARGET_CKPT        = "our_checkpoints/bassoon_add_fifth_100_bs16_V1/ckpts/epoch=99-valid_loss=0.627.ckpt"

# ---- INPUT audio -----------------------------------------------------------
INPUT_AUDIO_PATH   = "/home/shared_workspace/diffusion-timbre-transfer/real_cello.wav"

# ---- OUTPUT audio (where to save the generated result) ---------------------
WAV_DIR            = "/home/shared_workspace/diffusion-timbre-transfer/Outputs/WAV_files"
OUTPUT_AUDIO_PATH  = WAV_DIR + "/generated_bassoon.wav"

# ---- AddFifth (polyphony) --------------------------------------------------
# Set ADD_FIFTH = True to match the models trained WITH the fifth added.
# Set ADD_FIFTH = False to feed pure monophonic audio.
ADD_FIFTH          = False
FIFTH_GAIN         = 0.8

# ---- Diffusion settings (normally leave as-is) -----------------------------
SAMPLING_RATE      = 24000
CLIP_LENGTH        = 409600     # 17 seconds @ 24kHz; model trained on this
NUM_STEPS          = 100        # diffusion steps for each half of the bridge
SIGMA_MIN          = 0.001
SIGMA_MAX          = 5
RHO                = 9.0

# ---- Plotting --------------------------------------------------------------
# In a plain terminal there is no audio player; spectrogram plots are saved
# to PNG instead of shown. Set SHOW_PLOTS = True only inside a notebook.
SHOW_PLOTS         = False
PLOT_DIR           = "/home/shared_workspace/diffusion-timbre-transfer/Outputs/WAV_files/png_files"

# ============================================================================
#  END OF CONFIG  — you should not need to edit below this line
# ============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(REPO_PATH)
os.chdir(REPO_PATH)   # so the relative checkpoint paths above resolve

import torch
import torchaudio
import matplotlib
if not SHOW_PLOTS:
    matplotlib.use("Agg")   # headless backend, just saves files
import matplotlib.pyplot as plt

from main.module_base_latent_cond import Model, AudioDiffusionModel
from audio_diffusion_pytorch import (
    KarrasSamplerReverse,
    KarrasSampler,
    KarrasSchedule,
    KDistribution,
    PitchTracker,
    NormalizedEncodec,
)
from audio_data_pytorch.transforms.all import AddFifth


def banner(text):
    print("\n" + "=" * 70)
    print("  " + text)
    print("=" * 70)


def save_spec(waveform_np, title):
    """Save a spectrogram PNG (and optionally show it)."""
    path = os.path.join(PLOT_DIR, title.replace(" ", "_") + ".png")
    plt.figure(figsize=(10, 4))
    spec = torchaudio.transforms.Spectrogram()(torch.tensor(waveform_np))
    plt.imshow(spec.log2()[0].numpy(), aspect="auto", origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  saved spectrogram: {path}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    banner(f"Device: {device}")

    # ---- diffusion schedule / samplers -------------------------------------
    diffusion_sigma_distribution = KDistribution(
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO
    )
    diffusion_schedule = KarrasSchedule(
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO
    )
    diffusion_sampler_reverse = KarrasSamplerReverse()
    diffusion_sampler = KarrasSampler()

    # ---- load SOURCE model -------------------------------------------------
    banner(f"Loading SOURCE model: {SOURCE_NAME}")
    source_diffusion = AudioDiffusionModel(
        diffusion_sigma_distribution=diffusion_sigma_distribution,
    )
    pl_model_source = Model(
        model=source_diffusion,
        mean_path=SOURCE_MEAN_PATH,
        std_path=SOURCE_STD_PATH,
    )
    ckpt_source = torch.load(SOURCE_CKPT, map_location=device)
    pl_model_source.load_state_dict(ckpt_source["state_dict"], strict=True)
    pl_model_source.to(device)

    # ---- load TARGET model -------------------------------------------------
    banner(f"Loading TARGET model: {TARGET_NAME}")
    target_diffusion = AudioDiffusionModel(
        diffusion_sigma_distribution=diffusion_sigma_distribution,
    )
    pl_model_target = Model(
        model=target_diffusion,
        mean_path=TARGET_MEAN_PATH,
        std_path=TARGET_STD_PATH,
    )
    ckpt_target = torch.load(TARGET_CKPT, map_location=device)
    pl_model_target.load_state_dict(ckpt_target["state_dict"], strict=True)
    pl_model_target.to(device)

    # ---- load + prepare INPUT audio ----------------------------------------
    banner(f"Loading input audio: {INPUT_AUDIO_PATH}")
    waveform_raw, orig_sr = torchaudio.load(INPUT_AUDIO_PATH)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=SAMPLING_RATE)
    waveform_raw = resampler(waveform_raw)

    if ADD_FIFTH:
        print("  applying AddFifth (polyphony) to match training distribution")
        add_fifth = AddFifth(sample_rate=SAMPLING_RATE, fifth_gain=FIFTH_GAIN)
        waveform_raw = add_fifth(waveform_raw)

    waveform = waveform_raw.unsqueeze(0).to(device)

    # crop to CLIP_LENGTH if too long, pad if too short (model trained on 17s)
    if waveform.shape[-1] >= CLIP_LENGTH:
        input_waveform = waveform[..., :CLIP_LENGTH]
    else:
        pad_size = CLIP_LENGTH - waveform.shape[-1]
        input_waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        print(f"  input waveform shape: {input_waveform.shape}")
    save_spec(
        input_waveform.squeeze(0).cpu().numpy(),
        f"Input {SOURCE_NAME}" + (" + fifth" if ADD_FIFTH else ""),
    )
    # save the (cropped) input audio so you can listen to it too
    input_audio_out = WAV_DIR + f"/input_{SOURCE_NAME}" + ("_plus_fifth" if ADD_FIFTH else "") + ".wav"
    torchaudio.save(input_audio_out, input_waveform.squeeze(0).cpu(), SAMPLING_RATE)
    print(f"  saved input audio: {input_audio_out}")



    # ---- encode input ------------------------------------------------------
    encodec = NormalizedEncodec(device=device)
    embeddings_source = encodec.encode_latent(
        input_waveform, pl_model_source.mean, pl_model_source.std
    )
    print(f"  encodec embeddings shape: {embeddings_source.shape}")

    # ---- BRIDGE step 1: input -> noise (SOURCE model, reverse) -------------
    banner("Reverse diffusion: input -> noise")
    noisy_embeddings = pl_model_source.model.sample(
        noise=embeddings_source,
        sampler=diffusion_sampler_reverse,
        sigma_schedule=diffusion_schedule,
        num_steps=NUM_STEPS,
    )
    noise_waveform = encodec.decode_latent(
        noisy_embeddings, pl_model_source.mean, pl_model_source.std
    )
    save_spec(
        noise_waveform.cpu().detach().squeeze(0).numpy(),
        f"Noisy {SOURCE_NAME}",
    )

    # ---- BRIDGE step 2: noise -> target (TARGET model, forward) ------------
    banner("Forward diffusion: noise -> target")
    generated_embeddings = pl_model_target.model.sample(
        noise=noisy_embeddings,
        sampler=diffusion_sampler,
        sigma_schedule=diffusion_schedule,
        num_steps=NUM_STEPS,
    )
    target_waveform = encodec.decode_latent(
        generated_embeddings, pl_model_target.mean, pl_model_target.std
    )
    target_waveform_np = target_waveform.cpu().detach().squeeze(0).numpy()
    save_spec(target_waveform_np, f"Generated {TARGET_NAME}")

    # ---- save the generated audio ------------------------------------------
    banner(f"Saving generated audio: {OUTPUT_AUDIO_PATH}")
    torchaudio.save(
        OUTPUT_AUDIO_PATH,
        torch.tensor(target_waveform_np).cpu(),
        SAMPLING_RATE,
    )
    print(f"  done -> {OUTPUT_AUDIO_PATH}")

    # ---- pitch metrics (DPD / JD) ------------------------------------------
    banner("Pitch metrics")
    pitch_tracker = PitchTracker()
    tracking_output = pitch_tracker.cals_pitch_metric(
        input_waveform,
        torch.tensor(target_waveform_np),
        "both",
        plot=True,
        pair=(f"Input {SOURCE_NAME}", f"Generated {TARGET_NAME}"),
    )
    dtw, jaccard = tracking_output[0][0]
    print(f"  DPD (pitch distance, lower = better melody preservation): {round(dtw, 2)}")
    print(f"  JD  (Jaccard distance):                                   {round(jaccard, 2)}")

    banner("All done")


if __name__ == "__main__":
    main()