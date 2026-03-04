import torch
from ml4gw.waveforms import WhiteNoiseBurst
from typing import Tuple, Dict
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler

from .generator import WaveformGenerator

class WNBGenerator(WaveformGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.white_noise_burst = WhiteNoiseBurst(self.sample_rate, self.duration)

    def forward(self, **parameters):
        return self.white_noise_burst(**parameters)
