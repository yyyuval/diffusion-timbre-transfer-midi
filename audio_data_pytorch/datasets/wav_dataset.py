import math
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from tinytag import TinyTag
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import os

from ..utils import fast_scandir, is_silence


def get_all_wav_filenames(paths: Sequence[str], recursive: bool, instruments: str = "") -> List[str]:
    extensions = [".wav", ".flac"]
    filenames = []
    for path in paths:
        _, files = fast_scandir(path, extensions, recursive=recursive, instruments=instruments)
        filenames.extend(files)

    print(f"==================== Found {len(filenames)} {instruments} tracks ====================")
    return filenames

class WAVDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Sequence[str]],
        recursive: bool = False,
        instruments: str = "",
        transforms: Optional[Callable] = None,
        sample_rate: Optional[int] = None,
        random_crop_size: int = None,
        check_silence: bool = True,
        with_ID3: bool = False,
        midi_fps: int = 75,
        midi_cache_root: Optional[str] = None,
        midi_shuffle: bool = False,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive, instruments=instruments)
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.check_silence = check_silence
        self.with_ID3 = with_ID3
        self.random_crop_size = random_crop_size
        # Frame rate the cached piano rolls were rendered at. Must match the
        # --fps the midi_cache_* dirs were built with -- they are named for it
        # (midi_cache_cello_fps75_addfifth), but the cache scripts default to 50.
        self.midi_fps = midi_fps
        # Which piano-roll cache to read. Set from exp/datamodule/base.yaml so a
        # run can be pointed at a different source (ground truth vs basic-pitch,
        # with or without the added fifth) without editing code.
        self.midi_cache_root = midi_cache_root
        # A wrong path here does not fail -- every lookup misses and the dataset
        # quietly serves all-zero rolls, which looks exactly like "MIDI does not
        # help". Fail at construction instead. None means MIDI is switched off.
        if midi_cache_root is not None and not os.path.isdir(midi_cache_root):
            raise FileNotFoundError(
                "midi_cache_root does not exist: %s" % (midi_cache_root,)
            )
        # Control condition: pair each clip with a roll from a DIFFERENT track.
        # Same format, same density, wrong notes. If the gate still climbs under
        # this, gate movement is not evidence the model reads the notes.
        self.midi_shuffle = midi_shuffle
        assert (
            not random_crop_size or sample_rate
        ), "Optimized random crop requires sample_rate to be set."

    # Instead of loading the whole file and chopping out our crop,
    # we only load what we need.
    def optimized_random_crop(self, idx: int) -> Tuple[Tensor, int, float]:
        # Get length/audio info
        info = torchaudio.info(self.wavs[idx])
        length = info.num_frames
        sample_rate = info.sample_rate

        # Calculate correct number of samples to read based on actual
        # and intended sample rate
        ratio = 1 if (self.sample_rate is None) else sample_rate / self.sample_rate
        crop_size = length if (self.random_crop_size is None) else math.ceil(self.random_crop_size * ratio)  # type: ignore
        frame_offset = random.randint(0, max(length - crop_size, 0))

        # Where in the track this crop starts. Returned so callers can align
        # other per-track data (the MIDI piano roll) to the same window.
        crop_start_sec = frame_offset / info.sample_rate

        # Load the samples
        waveform, sample_rate = torchaudio.load(
            self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size
        )
       

        # check if waveform is stereo
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad with zeroes if the sizes aren't quite right
        # (e.g., rates aren't exact multiples)
        if len(waveform[0]) < crop_size:
            waveform = torch.nn.functional.pad(
                waveform,
                pad=(0, crop_size - len(waveform[0])),
                mode="constant",
                value=0,
            )

        return waveform, sample_rate, crop_start_sec
    
    #yuval add func
    def get_midi_cache_path(self, wav_path: str) -> str:
        if self.midi_cache_root is None:
            raise ValueError(
                "midi_cache_root is not set. Point it at a piano-roll cache in "
                "exp/datamodule/base.yaml, or override midi_cache_root=... on "
                "the command line. It used to be hardcoded per instrument, which "
                "made it impossible to switch MIDI source without editing code -- "
                "and silently paired plain audio with the +fifth rolls."
            )

        track_name = os.path.basename(os.path.dirname(os.path.dirname(wav_path)))
        file_name = os.path.splitext(os.path.basename(wav_path))[0]

        return os.path.join(
            self.midi_cache_root,
            track_name,
            f"{file_name}_pianoroll.npy"
        )
          

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tensor,
        Tuple[Tensor, int],
        Tuple[Tensor, Tensor],
        Tuple[Tensor, List[str], List[str]],
        Tuple[Tensor, TinyTag],
    ]:  # type: ignore
        invalid_audio = False

        # Loop until we find a valid audio sample
        while True:
            # If last sample was invalid, use a new random one.
            if invalid_audio:
                idx = random.randrange(len(self))

            # Catch invalid audio files
            try:
                # Read ID3 tags if specified
                if self.with_ID3:
                    tag = TinyTag.get(self.wavs[idx])

                # Read with optimized crop if needed
                if hasattr(self, "random_crop_size"):
                    waveform, sample_rate, crop_start_sec = self.optimized_random_crop(int(idx))
                    
                else:
                    waveform, sample_rate = torchaudio.load(self.wavs[idx])
                    crop_start_sec = 0.0
            except Exception:
                invalid_audio = True
                continue

            # Apply sample rate transform if necessary
            if self.sample_rate and sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate
                )(waveform)

                # Downsampling can result in slightly different sizes.
                if hasattr(self, "random_crop_size"):
                    waveform = waveform[:, : self.random_crop_size]
                

            # Apply other transforms
            if self.transforms:
                waveform = self.transforms(waveform)
           

            # Check silence after transforms (useful for random crops)
            if self.check_silence and is_silence(waveform):
                invalid_audio = True
                continue

            # Return with TinyTag ID3 object if specified
            if self.with_ID3:
                return waveform, tag

            instrument_name = os.path.splitext(os.path.basename(self.wavs[idx]))[0]

            # MIDI switched off: skip the load entirely. A stub keeps the tuple
            # length constant so collate and `len(batch) > 3` still behave.
            if self.midi_cache_root is None:
                return (
                    waveform,
                    instrument_name[2:],
                    self.wavs[idx],
                    torch.zeros(128, 1),
                )

            #yuval add
            midi_idx = idx
            if self.midi_shuffle and len(self.wavs) > 1:
                midi_idx = random.randrange(len(self.wavs))
                while midi_idx == idx:
                    midi_idx = random.randrange(len(self.wavs))

            midi_path = self.get_midi_cache_path(self.wavs[midi_idx])
            

            if os.path.exists(midi_path):
                piano_roll = np.load(midi_path).astype(np.float32)
            else:
                piano_roll = np.zeros((128, 1), dtype=np.float32)

            # Yuval add: crop MIDI to match the audio crop -- the same WINDOW of
            # the track, not merely the same duration. optimized_random_crop
            # starts at a random offset, so slicing the roll from frame 0 would
            # condition the model on a different part of the track than the
            # audio it is denoising.
            if self.sample_rate is not None:
                audio_duration_sec = waveform.shape[-1] / self.sample_rate
            else:
                # fallback, but in our training self.sample_rate should be 24000
                audio_duration_sec = waveform.shape[-1] / 24000

            target_midi_frames = int(round(audio_duration_sec * self.midi_fps))

            if self.midi_shuffle:
                # Take a window from wherever this other track actually has
                # notes, rather than at this clip's offset -- a short track
                # sliced at a large offset would come back mostly zeros, which
                # is the no-MIDI condition, not the wrong-MIDI one.
                midi_start = random.randint(
                    0, max(piano_roll.shape[1] - target_midi_frames, 0)
                )
            else:
                midi_start = int(round(crop_start_sec * self.midi_fps))

            piano_roll = piano_roll[:, midi_start : midi_start + target_midi_frames]

            # If MIDI is shorter than needed, pad with zeros
            if piano_roll.shape[1] < target_midi_frames:
                pad_width = target_midi_frames - piano_roll.shape[1]
                piano_roll = np.pad(
                    piano_roll,
                    pad_width=((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=0,
                )

            piano_roll = torch.from_numpy(piano_roll)

            #yuval add self.wavs[idx],piano_roll for midi
            return waveform, instrument_name[2:], self.wavs[idx], piano_roll

    def __len__(self) -> int:
        return len(self.wavs)