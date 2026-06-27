from pathlib import Path
import numpy as np
import pretty_midi

midi_cache_path = Path(
    "/home/shared_workspace/diffusion-timbre-transfer/"
    "midi_cache_cello_fps75/string_track000534/4_cello_pianoroll.npy"
)

out_midi_path = Path("debug_cropped_midi_fps75.mid")

# Same parameters as training
target_sample_rate = 24000
final_audio_samples = 409600
midi_fps = 75

audio_duration_sec = final_audio_samples / target_sample_rate
target_midi_frames = int(round(audio_duration_sec * midi_fps))

piano_roll = np.load(midi_cache_path).astype(np.float32)
piano_roll_crop = piano_roll[:, :target_midi_frames]

pm = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=42)  # cello-ish GM instrument

for note in range(128):
    active = piano_roll_crop[note] > 0

    if not active.any():
        continue

    starts = np.where(np.diff(np.r_[False, active]) == 1)[0]
    ends = np.where(np.diff(np.r_[active, False]) == -1)[0]

    for s, e in zip(starts, ends):
        start_time = s / midi_fps
        end_time = (e + 1) / midi_fps

        # skip tiny glitches
        if end_time - start_time < 0.04:
            continue

        midi_note = pretty_midi.Note(
            velocity=90,
            pitch=int(note),
            start=float(start_time),
            end=float(end_time),
        )
        instrument.notes.append(midi_note)

pm.instruments.append(instrument)
pm.write(str(out_midi_path))

print("saved:", out_midi_path)
print("duration sec:", audio_duration_sec)
print("target midi frames:", target_midi_frames)
print("num notes:", len(instrument.notes))