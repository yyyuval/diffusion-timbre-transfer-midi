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


def get_paired_stem_filenames(
    paths: Sequence[str], instruments: Sequence[str]
) -> List[List[str]]:
    """Tracks that contain a stem for EVERY named instrument, one group per track.

    This is NOT what passing several names to `instruments` does. fast_scandir
    matches any file whose name contains any listed instrument, so that yields
    individual stems of each instrument as separate samples -- never mixed.
    Here a sample is a whole track, and the stems get summed in load_mixture.

    Stems are returned in the order the instruments were requested, which is the
    order the piano rolls are stacked in, so it has to stay stable.
    """
    extensions = (".wav", ".flac")
    groups: List[List[str]] = []
    incomplete = 0

    for root in paths:
        if not os.path.isdir(root):
            continue
        for track_name in sorted(os.listdir(root)):
            stems_dir = os.path.join(root, track_name, "stems_audio")
            if not os.path.isdir(stems_dir):
                continue

            names = sorted(os.listdir(stems_dir))
            picked: List[str] = []
            taken = set()

            for inst in instruments:
                hit = next(
                    (
                        n
                        for n in names
                        if n not in taken
                        and n.lower().endswith(extensions)
                        and inst.lower() in n.lower()
                    ),
                    None,
                )
                if hit is None:
                    break
                taken.add(hit)
                picked.append(os.path.join(stems_dir, hit))

            if len(picked) == len(instruments):
                groups.append(picked)
            else:
                incomplete += 1

    print(
        "==================== Found %d tracks with all of %s ===================="
        % (len(groups), list(instruments))
    )
    if incomplete:
        # A quiet shortfall here is the same class of problem as Bug 10: nothing
        # fails, the dataset is just smaller than intended and nobody notices.
        print(
            "                     (%d tracks skipped -- missing at least one stem)"
            % incomplete
        )
    return groups


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
        mix_instruments: Optional[Sequence[str]] = None,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]

        # Two-instrument (or K-instrument) mode: one sample is a whole track,
        # with the named stems summed. When None everything below behaves
        # exactly as it did for single-instrument training.
        self.mix_instruments = list(mix_instruments) if mix_instruments else None

        if self.mix_instruments:
            self.groups = get_paired_stem_filenames(self.paths, self.mix_instruments)
            # Kept so code paths that only need "some wav for this sample" (the
            # instrument label, the returned path) still work.
            self.wavs = [g[0] for g in self.groups]
        else:
            self.groups = None
            self.wavs = get_all_wav_filenames(
                self.paths, recursive=recursive, instruments=instruments
            )
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

    def load_mixture(self, group: Sequence[str]) -> Tuple[Tensor, int, float]:
        """Read the SAME window from every stem of one track and sum them.

        The offset is chosen once and reused for all stems. Stems within a
        CocoChorales track are the same performance and the same length, so a
        shared offset is what makes the mixture musically coherent -- and it is
        what lets the piano rolls line up too (see Bug 10).
        """
        info = torchaudio.info(group[0])
        length = info.num_frames
        sample_rate = info.sample_rate

        ratio = 1 if (self.sample_rate is None) else sample_rate / self.sample_rate
        crop_size = (
            length
            if (self.random_crop_size is None)
            else math.ceil(self.random_crop_size * ratio)
        )
        frame_offset = random.randint(0, max(length - crop_size, 0))
        crop_start_sec = frame_offset / info.sample_rate

        mixture = None
        for stem_path in group:
            waveform, _ = torchaudio.load(
                stem_path, frame_offset=frame_offset, num_frames=crop_size
            )

            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)

            if waveform.shape[-1] < crop_size:
                waveform = torch.nn.functional.pad(
                    waveform,
                    pad=(0, crop_size - waveform.shape[-1]),
                    mode="constant",
                    value=0,
                )

            mixture = waveform if mixture is None else mixture + waveform

        # Only rescale when the sum would actually clip. Normalising every clip
        # unconditionally would flatten the loudness distribution, which the
        # single-instrument path does not do.
        peak = mixture.abs().max()
        if peak > 0.95:
            mixture = 0.95 * mixture / peak

        return mixture, sample_rate, crop_start_sec

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
                if self.mix_instruments:
                    waveform, sample_rate, crop_start_sec = self.load_mixture(
                        self.groups[int(idx)]
                    )
                elif hasattr(self, "random_crop_size"):
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
            if self.midi_shuffle and len(self) > 1:
                midi_idx = random.randrange(len(self))
                while midi_idx == idx:
                    midi_idx = random.randrange(len(self))

            # One roll per stem. In mixture mode these are stacked along the
            # pitch axis below, so the model sees which instrument plays what
            # rather than a merged "something is sounding" roll.
            if self.mix_instruments:
                midi_sources = self.groups[midi_idx]
            else:
                midi_sources = [self.wavs[midi_idx]]

            piano_rolls = []
            for src in midi_sources:
                midi_path = self.get_midi_cache_path(src)
                if os.path.exists(midi_path):
                    piano_rolls.append(np.load(midi_path).astype(np.float32))
                else:
                    piano_rolls.append(np.zeros((128, 1), dtype=np.float32))

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
                shortest = min(r.shape[1] for r in piano_rolls)
                midi_start = random.randint(0, max(shortest - target_midi_frames, 0))
            else:
                midi_start = int(round(crop_start_sec * self.midi_fps))

            # Every stem is cropped at the SAME offset, so the stacked rolls stay
            # aligned with each other and with the audio.
            for i, roll in enumerate(piano_rolls):
                roll = roll[:, midi_start : midi_start + target_midi_frames]

                # If MIDI is shorter than needed, pad with zeros
                if roll.shape[1] < target_midi_frames:
                    pad_width = target_midi_frames - roll.shape[1]
                    roll = np.pad(
                        roll,
                        pad_width=((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=0,
                    )
                piano_rolls[i] = roll

            # [128, T] for one instrument, [128*K, T] for a mixture -- stacked in
            # mix_instruments order, so midi_bins must be 128*K in the model.
            piano_roll = (
                piano_rolls[0]
                if len(piano_rolls) == 1
                else np.concatenate(piano_rolls, axis=0)
            )
            piano_roll = torch.from_numpy(piano_roll)

            #yuval add self.wavs[idx],piano_roll for midi
            return waveform, instrument_name[2:], self.wavs[idx], piano_roll

    def __len__(self) -> int:
        return len(self.groups) if self.mix_instruments else len(self.wavs)