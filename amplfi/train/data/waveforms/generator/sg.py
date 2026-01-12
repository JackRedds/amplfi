import torch
from ml4gw.waveforms import SineGaussian
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
    def __init__(self, max_shift: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_sg = SineGaussian(self.sample_rate, self.duration)
        self.max_shift = max_shift

    def ave_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        averaged_params = {}
        for i, params in parameters.items():
            for k, v in params.items():
                if k not in averaged_params:
                    averaged_params[k] = []
                averaged_params[k].append(v.mean(dim=0))
        # average parameters
        for k in averaged_params:
            averaged_params[k] = torch.stack(averaged_params[k])
        return averaged_params

    def shift_waveforms(self, cross: torch.Tensor, plus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = cross.shape[0]
        shifts = (torch.rand(N, device=cross.device) - 0.5) * 2 * self.max_shift
        shift_samples = (shifts * self.sample_rate).long()
        shifted_cross = torch.zeros_like(cross)
        shifted_plus = torch.zeros_like(plus)
        for i in range(N):
            shift = shift_samples[i].item()
            if shift > 0:
                shifted_cross[i, shift:] = cross[i, :-shift]
                shifted_plus[i, shift:] = plus[i, :-shift]
            elif shift < 0:
                shifted_cross[i, :shift] = cross[i, -shift:]
                shifted_plus[i, :shift] = plus[i, -shift:]
            else:
                shifted_cross[i] = cross[i]
                shifted_plus[i] = plus[i]
        return shifted_cross, shifted_plus

    def get_val_waveforms(self, _, world_size):
        num_waveforms = self.num_val_waveforms // world_size
        parameters = self.parameter_sampler(num_waveforms, device="cpu")
        hc, hp = self(**parameters)
        parameters = self.ave_parameters(parameters)
        return hc, hp, parameters

    def get_test_waveforms(self):
        parameters = self.test_parameter_sampler(self.num_test_waveforms)
        hc, hp = self(**parameters)
        parameters = self.ave_parameters(parameters)
        return hc, hp, parameters

    def sample(self, X):
        N = len(X)
        parameters = self.parameter_sampler(N, device=X.device)
        hc, hp = self(**parameters)
        parameters = self.ave_parameters(parameters)
        return hc, hp, parameters

    def fit_scaler(self, scaler: "ChannelWiseScaler") -> "ChannelWiseScaler":
        parameters = self.parameter_sampler(self.num_fit_params)
        parameters = self.ave_parameters(parameters)

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
        cross_waveforms = []
        plus_waveforms = []
        for i, params in parameters.items():
            cross, plus = self.multi_sg(**params)
            cross, plus = self.shift_waveforms(cross, plus)
            cross = cross.mean(dim=0, keepdim=True)
            plus = plus.mean(dim=0, keepdim=True)
            cross_waveforms.append(cross)
            plus_waveforms.append(plus)
        cross = torch.vstack(cross_waveforms)
        plus = torch.vstack(plus_waveforms)

        return cross, plus
