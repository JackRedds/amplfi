import inspect

import torch
from ml4gw.waveforms import MultiWaveform
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler

from .generator import WaveformGenerator


class MultiWFGenerator(WaveformGenerator):
    def __init__(
        self,
        *args,
        generator: Callable[[float, float], torch.nn.Module],
        norm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        generator = generator(self.sample_rate, self.duration)
        self.multi_wf = MultiWaveform(generator, self.sample_rate, self.duration, norm)
        # only per-component parameters the generator accepts are
        # passed to it, so priors can carry extra keys like
        # `n_components`, `component_mask` or `hrss_frac`
        self.generator_params = set(
            inspect.signature(generator.forward).parameters
        )

    def slice_waveforms(self, waveforms: torch.Tensor, waveform_size: int):
        # for sine gaussians, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        half = waveform_size // 2
        start = center - half
        stop = center + half
        return waveforms[..., start:stop]

    def forward(self, **parameters):
        components = {
            k: v for k, v in parameters.items() if k in self.generator_params
        }
        num_components = next(iter(components.values())).shape[-1]

        # MultiWaveform computes the summed waveform's hrss in float32,
        # which underflows when every component is faint (hrss ~1e-23),
        # giving a badly wrong rescaling to hrss_tot. When rescaling,
        # only relative amplitudes matter, so bring them to order 1.
        # Sine-Gaussians are parameterized by `hrss`, Morlet-Gabor
        # wavelets by `amplitude`
        if self.multi_wf.norm:
            for key in ("hrss", "amplitude"):
                if key in components:
                    amp = components[key]
                    scale = amp.amax(dim=-1, keepdim=True).clamp_min(1e-30)
                    components[key] = amp / scale

        # MultiWaveform infers the number of components from `hrss_tot`,
        # so expand it if it was sampled once per event
        hrss_tot = parameters["hrss_tot"]
        if hrss_tot.ndim == 1:
            hrss_tot = hrss_tot[:, None].expand(-1, num_components)

        hc, hp = self.multi_wf(hrss_tot=hrss_tot, **components)
        waveforms = torch.stack([hc, hp], dim=1)
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        hc, hp = waveforms.transpose(1, 0)
        return hc, hp
