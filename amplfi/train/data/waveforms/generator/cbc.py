from typing import Callable

from numpy import cross

import torch
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

from .generator import WaveformGenerator


class CBCGenerator(WaveformGenerator):
    def __init__(
        self,
        *args,
        approximant: Callable,
        f_min: float,
        f_ref: float,
        right_pad: float,
        **kwargs,
    ):
        """
        A lightweight wrapper around
        `ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator`
        to make it compatible with
        `amplfi.train.data.waveforms.generator.WaveformGenerator`.


        Args:
            *args:
                Positional arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
            approximant:
                A callable that takes parameter tensors
                and returns the cross and plus polarizations.
                For example, `ml4gw.waveforms.IMRPhenomD()`
            f_min:
                Lowest frequency at which waveform signal content
                is generated
            f_ref:
                Reference frequency
            right_pad:
                Position in seconds where coalesence is placed
                relative to the right edge of the window
            **kwargs:
                Keyword arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
        """
        super().__init__(*args, **kwargs)
        self.right_pad = right_pad
        self.approximant = approximant
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant,
            self.sample_rate,
            self.duration,
            f_min,
            f_ref,
            right_pad + self.fduration / 2,
        )

    def center_waveforms(self, hc, hp):
        hc_centered = torch.zeros_like(hc)
        hp_centered = torch.zeros_like(hp)
        N = hc.shape[1]
        num_waveforms = hc.shape[0]
        idx_peak = torch.argmax(torch.abs(hc), dim=1)
        shift = N//2 - idx_peak
        for i in range(num_waveforms):
            hc_centered[i] = torch.roll(hc[i], shift[i].item())
            hp_centered[i] = torch.roll(hp[i], shift[i].item())
        return hc_centered, hp_centered

    def forward(self, **parameters) -> torch.Tensor:
        hc, hp = self.waveform_generator(**parameters)
        # hc, hp = self.center_waveforms(hc, hp)
        waveforms = torch.stack([hc, hp], dim=1)
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        hc, hp = waveforms.transpose(1, 0)

        return hc.float(), hp.float()
