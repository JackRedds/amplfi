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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_sg = MultiSineGaussian(self.sample_rate, self.duration)

    # def get_val_waveforms(
    #     self, _, world_size
    # ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    #     num_waveforms = self.num_val_waveforms // world_size
    #     parameters = self.training_prior(num_waveforms, device="cpu")
    #     hc, hp = self(**parameters)
    #     parameters = self.multi_sg.ave_parameters(parameters)
    #     return hc, hp, parameters

    # def get_test_waveforms(
    #     self,
    # ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    #     parameters = self.testing_prior(self.num_test_waveforms)
    #     hc, hp = self(**parameters)
    #     parameters = self.multi_sg.ave_parameters(parameters)
    #     return hc, hp, parameters

    # def sample(
    #     self, X
    # ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    #     N = len(X)
    #     parameters = self.training_prior(N, device=X.device)
    #     hc, hp = self(**parameters)
    #     parameters = self.multi_sg.ave_parameters(parameters)
    #     return hc, hp, parameters

    # def get_fit_parameters(self) -> torch.Tensor:
    #     parameters = self.training_prior(self.num_fit_params)
    #     parameters = self.multi_sg.ave_parameters(parameters)
    #     return parameters

    def slice_waveforms(self, waveforms: torch.Tensor, waveform_size: int):
        # for sine gaussians, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        half = waveform_size // 2
        start = center - half
        stop = center + half
        return waveforms[..., start:stop]

    def forward(self, **parameters):
        return self.multi_sg(**parameters)
