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

    def get_val_waveforms(self, _, world_size):
        num_waveforms = self.num_val_waveforms // world_size
        parameters = self.parameter_sampler(num_waveforms, device="cpu")
        hc, hp = self(**parameters)
        parameters = self.multi_sg.ave_parameters(parameters)
        return hc, hp, parameters

    def get_test_waveforms(self):
        parameters = self.test_parameter_sampler(self.num_test_waveforms)
        hc, hp = self(**parameters)
        parameters = self.multi_sg.ave_parameters(parameters)
        return hc, hp, parameters

    def sample(self, X):
        N = len(X)
        parameters = self.parameter_sampler(N, device=X.device)
        hc, hp = self(**parameters)
        parameters = self.multi_sg.ave_parameters(parameters)
        return hc, hp, parameters

    def fit_scaler(self, scaler: "ChannelWiseScaler") -> "ChannelWiseScaler":
        parameters = self.parameter_sampler(self.num_fit_params)
        parameters = self.multi_sg.ave_parameters(parameters)

        dec, psi, phi = self.sample_extrinsic(torch.ones(self.num_fit_params))
        parameters.update({"dec": dec, "psi": psi, "phi": phi})
        transformed = self.parameter_transformer(parameters)

        fit = []
        for key in self.inference_params:
            fit.append(transformed[key])

        fit = torch.row_stack(fit)
        scaler.fit(fit)
        return scaler

    def slice_waveforms(self, waveforms: torch.Tensor, waveform_size: int):
        # for sine gaussians, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        half = waveform_size // 2
        start = center - half
        stop = center + half
        return waveforms[..., start:stop]

    def forward(self, **parameters):
        return self.multi_sg(**parameters)
