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
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive, instruments=instruments)
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.check_silence = check_silence
        self.with_ID3 = with_ID3
        self.random_crop_size = random_crop_size
        assert (
            not random_crop_size or sample_rate
        ), "Optimized random crop requires sample_rate to be set."

    # Instead of loading the whole file and chopping out our crop,
    # we only load what we need.
    def optimized_random_crop(self, idx: int) -> Tuple[Tensor, int]:
        # Get length/audio info
        info = torchaudio.info(self.wavs[idx])
        length = info.num_frames
        sample_rate = info.sample_rate

        # Calculate correct number of samples to read based on actual
        # and intended sample rate
        ratio = 1 if (self.sample_rate is None) else sample_rate / self.sample_rate
        crop_size = length if (self.random_crop_size is None) else math.ceil(self.random_crop_size * ratio)  # type: ignore
        frame_offset = random.randint(0, max(length - crop_size, 0))
        #yuval debug
        print(
        "DEBUG wav:",
        self.wavs[idx],
        "frame_offset:",
        frame_offset,
        "crop_size:",
        crop_size,
        "length:",
        length,
        )
        #yuval stop
        # Load the samples
        waveform, sample_rate = torchaudio.load(
            self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size
        )
        #yuval debug
        print("DEBUG after torchaudio.load:", waveform.shape, "sample_rate:", sample_rate)

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

        return waveform, sample_rate
    
    #yuval add func
    def get_midi_cache_path(self, wav_path: str) -> str:
        """
        Temporary MIDI cache mapping for debug.
        Example:
        .../main_dataset/string_track000534/stems_audio/4_cello.wav
        -> midi_cache_test/string_track000534/4_cello_pianoroll.npy
        """
        track_name = os.path.basename(os.path.dirname(os.path.dirname(wav_path)))
        file_name = os.path.splitext(os.path.basename(wav_path))[0]

        return os.path.join(
        "/home/shared_workspace/diffusion-timbre-transfer/midi_cache_cello_fps75",
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
                    waveform, sample_rate = self.optimized_random_crop(int(idx))
                    #yuval debug
                    print("DEBUG after optimized_random_crop:", waveform.shape, "sample_rate:", sample_rate)
                else:
                    waveform, sample_rate = torchaudio.load(self.wavs[idx])
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
                #yuval debug    
                print("DEBUG after resample block:", waveform.shape)

            # Apply other transforms
            if self.transforms:
                waveform = self.transforms(waveform)
            #yuval debug
            print("DEBUG after self.transforms:", waveform.shape)

            # Check silence after transforms (useful for random crops)
            if self.check_silence and is_silence(waveform):
                invalid_audio = True
                continue

            # Return with TinyTag ID3 object if specified
            if self.with_ID3:
                return waveform, tag

            instrument_name = os.path.splitext(os.path.basename(self.wavs[idx]))[0]

            #yuval add
            midi_path = self.get_midi_cache_path(self.wavs[idx])
            print("DEBUG midi_path:", midi_path)
            print("DEBUG midi exists:", os.path.exists(midi_path))

            if os.path.exists(midi_path):
                piano_roll = np.load(midi_path).astype(np.float32)
            else:
                piano_roll = np.zeros((128, 1), dtype=np.float32)

            # Yuval add: crop MIDI to match the final audio duration
            midi_fps = 75

            if self.sample_rate is not None:
                audio_duration_sec = waveform.shape[-1] / self.sample_rate
            else:
                # fallback, but in our training self.sample_rate should be 24000
                audio_duration_sec = waveform.shape[-1] / 24000

            target_midi_frames = int(round(audio_duration_sec * midi_fps))

            piano_roll = piano_roll[:, :target_midi_frames]

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
            print("DEBUG before return waveform:", waveform.shape)
            print("DEBUG before return piano_roll:", piano_roll.shape)

            #yuval add self.wavs[idx],piano_roll for midi
            return waveform, instrument_name[2:], self.wavs[idx], piano_roll

    def __len__(self) -> int:
        return len(self.wavs)