import torch
from ml4gw.waveforms import SineGaussian, MultiSineGaussian
from typing import Tuple, Dict
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler

from .generator import WaveformGenerator

class SGGenerator(WaveformGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sine_gaussian = SineGaussian(self.sample_rate, self.duration)

    def slice_waveforms(self, waveforms: torch.Tensor, waveform_size: int):
        # for sine gaussians, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        half = waveform_size // 2
        start = center - half
        stop = center + half
        return waveforms[..., start:stop]

    def forward(self, **parameters):
        return self.sine_gaussian(**parameters)


class MultiSGGenerator(WaveformGenerator):
    def __init__(self, *args, norm: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_sg = MultiSineGaussian(self.sample_rate, self.duration, norm)

    def slice_waveforms(self, waveforms: torch.Tensor, waveform_size: int):
        # for sine gaussians, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        half = waveform_size // 2
        start = center - half
        stop = center + half
        return waveforms[..., start:stop]

    def forward(self, **parameters):
        return self.multi_sg(**parameters)
