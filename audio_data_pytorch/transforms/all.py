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
#yuval added
import torch
import torchaudio

#shlomi added this func for polyphony
class AddFifth(nn.Module):
    def __init__(self, sample_rate: int = 24000, fifth_gain: float = 0.8):
        super().__init__()
        self.sample_rate = sample_rate
        self.fifth_gain = fifth_gain

    def forward(self, x: Tensor) -> Tensor:
        x_fifth = torchaudio.functional.pitch_shift(
            x,
            sample_rate=self.sample_rate,
            n_steps=7,
        )
        x_mix = x + self.fifth_gain * x_fifth
        max_val = torch.max(torch.abs(x_mix))
        if max_val > 0:
            x_mix = 0.95 * x_mix / max_val
        return x_mix
    
class AllTransform(nn.Module):
    def __init__(
        self,
        source_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        crop_size: Optional[int] = None,
        random_crop_size: Optional[int] = None,
        loudness: Optional[int] = None,
        scale: Optional[float] = None,
        stereo: bool = False,
        mono: bool = False,
        sign: Optional[int] = None,
        #yuval added these two
        add_fifth: bool = False,
        fifth_gain: float = 0.8,
    ):
        super().__init__()
        self.random_crop_size = random_crop_size

        message = "Loudness requires target_rate"
        assert not exists(loudness) or exists(target_rate), message
        self.transform = nn.Sequential(
            Resample(source=source_rate, target=target_rate)  # type: ignore
            if (
                exists(source_rate)
                and exists(target_rate)
                and source_rate != target_rate
            )
            else nn.Identity(),
            RandomCrop(random_crop_size) if exists(random_crop_size) else nn.Identity(),
            PitchShift(sign) if (sign is not None and sign != 'None') else nn.Identity(),
            #yuval added one line
            AddFifth(sample_rate=target_rate or source_rate or 24000, fifth_gain=fifth_gain) if add_fifth else nn.Identity(),
            Crop(crop_size) if exists(crop_size) else nn.Identity(),
            Mono() if mono else nn.Identity(),
            Stereo() if stereo else nn.Identity(),
            Loudness(sampling_rate=target_rate, target=loudness)  # type: ignore
            if exists(loudness)
            else nn.Identity(),
            Scale(scale) if exists(scale) else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)
