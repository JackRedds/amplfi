import torch
from ml4gw.waveforms import MorletGabor
from typing import Tuple, Dict
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler

from .generator import WaveformGenerator

class MGGenerator(WaveformGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.morlet_gabor = MorletGabor(self.sample_rate, self.duration)

    def slice_waveforms(self, waveforms: torch.Tensor, waveform_size: int):
        # for morlet-gabor wavelets, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        half = waveform_size // 2
        start = center - half
        stop = center + half
        return waveforms[..., start:stop]

    def forward(self, **parameters):
        hc, hp = self.morlet_gabor(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        hc, hp = waveforms.transpose(1, 0)
        return hc, hp