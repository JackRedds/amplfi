import torch
from ml4gw.waveforms import StringGenerator
from typing import Tuple, Dict
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler

from .generator import WaveformGenerator

class CSGenerator(WaveformGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cosmic_string = StringGenerator(self.sample_rate, self.duration)

    def forward(self, **parameters):
        return self.cosmic_string(**parameters)

