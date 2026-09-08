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
# One wav, or SEVERAL stems of the SAME track in a list -- those are summed, the
# way the mixture models saw them during training. The order must match the
# model's mix_instruments, since that is the order their piano rolls stack in.
#
#   solo:    ".../string_track005158/stems_audio/4_cello.wav"
#   mixture: [".../stems_audio/1_violin.wav", ".../stems_audio/4_cello.wav"]
INPUT_AUDIO_PATH   = "/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/cocochorales_tiny_v1_zipped/main_dataset/string_track005158/stems_audio/4_cello.wav"

# ---- OUTPUT audio (where to save the generated result) ---------------------
WAV_DIR            = "/home/shared_workspace/diffusion-timbre-transfer/Outputs/WAV_files"
OUTPUT_AUDIO_PATH  = WAV_DIR + "/generated_bassoon.wav"

# ---- AddFifth (polyphony) --------------------------------------------------
# Set ADD_FIFTH = True to match the models trained WITH the fifth added.
# Set ADD_FIFTH = False to feed pure monophonic audio.
ADD_FIFTH          = False
FIFTH_GAIN         = 0.8

# ---- MIDI conditioning -----------------------------------------------------
# These MUST match the checkpoints being loaded. The models are constructed from
# this config and then loaded with strict=True, so a mismatch fails immediately
# with missing/unexpected keys rather than silently running unconditioned.
#
#   USE_MIDI = False              -> mode A checkpoints (no MIDI params at all)
#   USE_MIDI, MIDI_MODE="single"  -> mode B (original one-shot injection)
#   USE_MIDI, MIDI_MODE="multiscale" -> mode C (injected at every resolution)
#
# Both halves of the bridge get the same roll, so the SOURCE and TARGET models
# must have been trained with the same mode and the same MIDI_BINS.
USE_MIDI           = False
MIDI_MODE          = "multiscale"
MIDI_BINS          = 128        # 128 * number of instruments the model saw
MIDI_FPS           = 75         # must match the cache the models trained on

# Leave EMPTY to take the notes straight from the dataset: every CocoChorales
# track ships stems_midi/<stem>.mid beside stems_audio/<stem>.wav, so the roll is
# DERIVED from INPUT_AUDIO_PATH rather than chosen. That is the point -- pairing
# audio with another track's notes is worse than no conditioning at all, and it
# cannot happen if nobody picks the file.
#
# Set it only to override: a cached .npy roll, a .mid, or a list of either (one
# per stem, in INPUT_AUDIO_PATH order).
MIDI_PATH          = ""

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

import numpy as np
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


def as_list(value):
    """One path or several -- the rest of the script only handles lists."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def derive_midi_paths(audio_paths):
    """stems_audio/4_cello.wav  ->  stems_midi/4_cello.mid, for each stem.

    CocoChorales ships ground-truth MIDI beside every stem, so the notes that
    belong to this exact audio are one substitution away. Deriving them removes
    the only way the pair can go wrong.
    """
    midi_paths = []
    for audio_path in audio_paths:
        stems_dir, wav_name = os.path.split(audio_path)
        track_dir, audio_folder = os.path.split(stems_dir)
        if audio_folder != "stems_audio":
            raise ValueError(
                "cannot derive MIDI: %s is not inside a stems_audio/ folder. "
                "Point INPUT_AUDIO_PATH at a CocoChorales stem, or set "
                "MIDI_PATH yourself." % (audio_path,)
            )
        midi_path = os.path.join(
            track_dir, "stems_midi", os.path.splitext(wav_name)[0] + ".mid"
        )
        if not os.path.exists(midi_path):
            raise FileNotFoundError("no ground-truth MIDI at %s" % (midi_path,))
        midi_paths.append(midi_path)
    return midi_paths


def load_piano_roll(paths, target_frames, midi_fps, midi_bins, device):
    """Build the [128*K, T] roll from one .npy/.mid per stem, stacked.

    The input audio is cropped from the start of the file, so each roll is too --
    they have to describe the same stretch of music or the conditioning is worse
    than useless (that was Bug 10).
    """
    rolls = []
    for path in as_list(paths):
        if path.lower().endswith(".npy"):
            roll = np.load(path).astype(np.float32)
        else:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(path)
            # Same two lines the cache builders use. Threshold 0 keeps every
            # sounding note: CocoChorales writes a flat velocity of 64, so a
            # threshold of 40 filters nothing there but would silently drop
            # quiet notes in a hand-made .mid.
            roll = (pm.get_piano_roll(fs=midi_fps) > 0).astype(np.float32)

        roll = roll[:, :target_frames]
        if roll.shape[1] < target_frames:
            roll = np.pad(
                roll,
                pad_width=((0, 0), (0, target_frames - roll.shape[1])),
                mode="constant",
                constant_values=0,
            )
        print("    %-40s %s  active %d/%d"
              % (os.path.basename(path), roll.shape,
                 int((roll.sum(axis=0) > 0).sum()), roll.shape[1]))
        rolls.append(roll)

    # Stacked, not merged: rows 0-127 are the first stem, 128-255 the second.
    # Merging would throw away which instrument plays which note.
    roll = np.concatenate(rolls, axis=0) if len(rolls) > 1 else rolls[0]

    if roll.shape[0] != midi_bins:
        raise ValueError(
            "the roll has %d pitch rows but MIDI_BINS is %d. A mixture model "
            "needs one stem per 128 rows -- check that INPUT_AUDIO_PATH lists "
            "as many stems as the model was trained on."
            % (roll.shape[0], midi_bins)
        )

    print("  piano roll: %s" % (roll.shape,))
    return torch.from_numpy(roll).unsqueeze(0).to(device)


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
        use_midi=USE_MIDI,
        midi_mode=MIDI_MODE,
        midi_bins=MIDI_BINS,
    )
    pl_model_source = Model(
        model=source_diffusion,
        mean_path=SOURCE_MEAN_PATH,
        std_path=SOURCE_STD_PATH,
        use_midi=USE_MIDI,
    )
    ckpt_source = torch.load(SOURCE_CKPT, map_location=device)
    pl_model_source.load_state_dict(ckpt_source["state_dict"], strict=True)
    pl_model_source.to(device)

    # ---- load TARGET model -------------------------------------------------
    banner(f"Loading TARGET model: {TARGET_NAME}")
    target_diffusion = AudioDiffusionModel(
        diffusion_sigma_distribution=diffusion_sigma_distribution,
        use_midi=USE_MIDI,
        midi_mode=MIDI_MODE,
        midi_bins=MIDI_BINS,
    )
    pl_model_target = Model(
        model=target_diffusion,
        mean_path=TARGET_MEAN_PATH,
        std_path=TARGET_STD_PATH,
        use_midi=USE_MIDI,
    )
    ckpt_target = torch.load(TARGET_CKPT, map_location=device)
    pl_model_target.load_state_dict(ckpt_target["state_dict"], strict=True)
    pl_model_target.to(device)

    # ---- load + prepare INPUT audio ----------------------------------------
    audio_paths = as_list(INPUT_AUDIO_PATH)
    banner("Loading input audio: %d stem(s)" % len(audio_paths))

    waveform_raw = None
    for audio_path in audio_paths:
        print("  " + audio_path)
        stem, orig_sr = torchaudio.load(audio_path)
        if orig_sr != SAMPLING_RATE:
            stem = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=SAMPLING_RATE
            )(stem)
        waveform_raw = stem if waveform_raw is None else waveform_raw + stem

    # Same headroom rule as the training dataset's mixture path. A solo stem is
    # fed exactly as recorded, so only rescale when there was something to sum.
    if len(audio_paths) > 1:
        peak = waveform_raw.abs().max()
        print("  summed %d stems, peak %.3f" % (len(audio_paths), float(peak)))
        if peak > 0.95:
            waveform_raw = 0.95 * waveform_raw / peak
            print("  rescaled to 0.95")

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

    # ---- MIDI conditioning -------------------------------------------------
    midi = None
    if USE_MIDI:
        if MIDI_PATH:
            midi_paths = as_list(MIDI_PATH)
            banner("Loading MIDI (from MIDI_PATH)")
        else:
            midi_paths = derive_midi_paths(audio_paths)
            banner("Loading MIDI (ground truth, derived from the input stems)")
        target_frames = int(round(CLIP_LENGTH / SAMPLING_RATE * MIDI_FPS))
        midi = load_piano_roll(
            midi_paths, target_frames, MIDI_FPS, MIDI_BINS, device
        )
    else:
        print("  MIDI conditioning OFF (USE_MIDI=False)")

    # ---- BRIDGE step 1: input -> noise (SOURCE model, reverse) -------------
    banner("Reverse diffusion: input -> noise")
    noisy_embeddings = pl_model_source.model.sample(
        noise=embeddings_source,
        midi=midi,
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
        midi=midi,
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