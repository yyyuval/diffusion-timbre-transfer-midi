from pathlib import Path
import random

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import pretty_midi

# ===== paths =====
wav_path = Path(
    "/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/"
    "cocochorales_tiny_v1_zipped/main_dataset/string_track000534/stems_audio/4_cello.wav"
)

midi_path = Path(
    "/home/shared_workspace/diffusion-timbre-transfer/"
    "midi_cache_cello_fps75/string_track000534/4_cello_pianoroll.npy"
)

out_plot_path = Path("debug_audio_midi_crop_compare.png")
out_audio_crop_path = Path("debug_audio_training_crop.wav")
out_midi_synth_path = Path("debug_midi_crop_synth.wav")
out_midi_file_path = Path("debug_audio_crop_exported.mid")


# ===== parameters from training =====
target_sample_rate = 24000
final_audio_samples = 409600
midi_fps = 75

# Where in the track to crop, in seconds.
#   None -> pick at random, exactly like training does
#   a number -> pin the window so you can re-inspect the same one
crop_start_sec = None


def midi_note_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


# ===== load audio =====
waveform, sr = torchaudio.load(str(wav_path))

# mono
if waveform.shape[0] == 2:
    waveform = waveform.mean(dim=0, keepdim=True)

# resample like training
if sr != target_sample_rate:
    waveform = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=target_sample_rate,
    )(waveform)

# Crop like training ACTUALLY does: from a random offset, not from the start.
# WAVDataset.optimized_random_crop picks a uniform random frame_offset, so a
# debug script that only ever checks offset 0 tests the one case where the
# piano roll lines up by construction and cannot see a misalignment bug.
max_start_sample = max(waveform.shape[-1] - final_audio_samples, 0)

if crop_start_sec is None:
    start_sample = random.randint(0, max_start_sample)
else:
    start_sample = min(int(round(crop_start_sec * target_sample_rate)), max_start_sample)

waveform = waveform[:, start_sample:start_sample + final_audio_samples]

# if shorter, pad
if waveform.shape[-1] < final_audio_samples:
    pad = final_audio_samples - waveform.shape[-1]
    waveform = torch.nn.functional.pad(waveform, (0, pad))

crop_start_sec = start_sample / target_sample_rate
audio_duration_sec = waveform.shape[-1] / target_sample_rate

print("audio shape:", waveform.shape)
print("audio duration:", audio_duration_sec)
print("crop start:", round(crop_start_sec, 3), "sec")

# save audio crop that the model actually sees
torchaudio.save(
    str(out_audio_crop_path),
    waveform.cpu(),
    target_sample_rate,
)
print("saved audio crop:", out_audio_crop_path)


# ===== load MIDI piano roll =====
piano_roll = np.load(midi_path).astype(np.float32)

midi_start = int(round(crop_start_sec * midi_fps))
target_midi_frames = int(round(audio_duration_sec * midi_fps))

# Crop MIDI to the same WINDOW as the audio -- same offset, same duration.
piano_roll_crop = piano_roll[:, midi_start:midi_start + target_midi_frames]

# pad MIDI if needed
if piano_roll_crop.shape[1] < target_midi_frames:
    pad = target_midi_frames - piano_roll_crop.shape[1]
    piano_roll_crop = np.pad(
        piano_roll_crop,
        pad_width=((0, 0), (0, pad)),
        mode="constant",
        constant_values=0,
    )

print("full MIDI shape:", piano_roll.shape)
print("cropped MIDI shape:", piano_roll_crop.shape)
print("MIDI start frame:", midi_start)

# Show how far off the old frame-0 slice was for this window.
old_crop = piano_roll[:, :target_midi_frames]
if old_crop.shape == piano_roll_crop.shape:
    disagree = float(np.mean(old_crop != piano_roll_crop))
    print("frames differing from the old frame-0 slice: %.1f%%" % (100.0 * disagree))
print("target_midi_frames:", target_midi_frames)

# ===== export cropped piano roll to MIDI file =====
pm = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=42)  # General MIDI cello

num_notes = 0

for note in range(128):
    active = piano_roll_crop[note] > 0

    if not active.any():
        continue

    active_int = active.astype(np.int8)

    starts = np.where(np.diff(np.r_[0, active_int]) == 1)[0]
    ends = np.where(np.diff(np.r_[active_int, 0]) == -1)[0]

    for s, e in zip(starts, ends):
        start_time = s / midi_fps
        end_time = (e + 1) / midi_fps

        # skip extremely short glitches
        #if end_time - start_time < 0.03:
        #    continue

        midi_note = pretty_midi.Note(
            velocity=90,
            pitch=int(note),
            start=float(start_time),
            end=float(end_time),
        )

        instrument.notes.append(midi_note)
        num_notes += 1

pm.instruments.append(instrument)
pm.write(str(out_midi_file_path))

print("saved MIDI file:", out_midi_file_path)
print("num exported MIDI notes:", num_notes)


# ===== synthesize cropped MIDI as simple sine audio =====
num_samples = waveform.shape[-1]
midi_audio = np.zeros(num_samples, dtype=np.float32)

samples_per_frame = target_sample_rate / midi_fps

for frame_idx in range(piano_roll_crop.shape[1]):
    active_notes = np.where(piano_roll_crop[:, frame_idx] > 0)[0]

    start = int(round(frame_idx * samples_per_frame))
    end = int(round((frame_idx + 1) * samples_per_frame))
    end = min(end, num_samples)

    if start >= num_samples:
        break

    t = np.arange(end - start) / target_sample_rate

    for note in active_notes:
        freq = midi_note_to_hz(int(note))
        midi_audio[start:end] += 0.1 * np.sin(2 * np.pi * freq * t)

# normalize MIDI synth audio
max_val = np.max(np.abs(midi_audio))
if max_val > 0:
    midi_audio = 0.95 * midi_audio / max_val

midi_audio_tensor = torch.from_numpy(midi_audio).unsqueeze(0)

torchaudio.save(
    str(out_midi_synth_path),
    midi_audio_tensor,
    target_sample_rate,
)
print("saved MIDI synth:", out_midi_synth_path)


# ===== create spectrogram =====
spec = torchaudio.transforms.Spectrogram(
    n_fft=2048,
    hop_length=512,
    power=2.0,
)(waveform)

spec_db = torchaudio.transforms.AmplitudeToDB()(spec)[0].numpy()


# ===== plot =====
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# audio spectrogram
extent_spec = [0, audio_duration_sec, 0, target_sample_rate / 2]
axes[0].imshow(
    spec_db,
    origin="lower",
    aspect="auto",
    extent=extent_spec,
)
axes[0].set_title(
    "Audio spectrogram, crop @ %.2f-%.2f sec"
    % (crop_start_sec, crop_start_sec + audio_duration_sec)
)
axes[0].set_ylabel("Frequency [Hz]")
axes[0].set_ylim(0, 3000)

# MIDI piano roll
extent_midi = [0, audio_duration_sec, 0, 127]
axes[1].imshow(
    piano_roll_crop,
    origin="lower",
    aspect="auto",
    extent=extent_midi,
    interpolation="nearest",
)
axes[1].set_title("MIDI piano roll for the SAME window (frames %d-%d)"
                  % (midi_start, midi_start + target_midi_frames))
axes[1].set_xlabel("Time [sec]")
axes[1].set_ylabel("MIDI note")
axes[1].set_ylim(30, 90)

plt.tight_layout()
plt.savefig(out_plot_path, dpi=200)
print("saved plot:", out_plot_path)