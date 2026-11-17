import math
from typing import Callable, Optional

import torch


class AmplfiPrior:
    def __init__(
        self,
        priors: dict[str, torch.distributions.Distribution],
        conversion_function: Optional[Callable] = None,
    ):
        """
        A class for sampling parameters from a prior distribution

        Args:
            priors:
                A dictionary of parameter samplers that take an integer N
                and return a tensor of shape (N, ...) representing
                samples from the prior distribution
            conversion_function:
                A callable that takes a dictionary of sampled parameters
                and returns a dictionary of waveform generation parameters
        """
        super().__init__()
        self.priors = priors
        self.conversion_function = conversion_function or (lambda x: x)

    def __call__(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Generates random samples from the prior

        Args:
            N: Number of samples to generate
            device: Device to place the samples
        """
        # sample parameters from prior
        parameters = {
            k: v.sample((N,)).to(device) for k, v in self.priors.items()
        }
        # perform any necessary conversions
        # to from sampled parameters to
        # waveform generation parameters
        parameters = self.conversion_function(parameters)
        return parameters

    def log_prob(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the log probability of samples under the prior

        Args:
            samples:
                Dictionary where key is parameter and
                value is tensor of samples
        """

        first = samples[list(samples.keys())[0]]
        log_probs = torch.ones(len(first), device=first.device)
        for param, tensor in samples.items():
            log_probs += self.priors[param].log_prob(tensor).to(first.device)
        return log_probs


class AmplfiMultiWFPrior(AmplfiPrior):
    def __init__(
        self,
        n_wf: torch.distributions.Distribution,
        priors: dict[str, torch.distributions.Distribution],
        shared_priors: Optional[list[str]] = None,
        event_priors: Optional[dict[str, torch.distributions.Distribution]] = None,
        max_components: Optional[int] = None,
        conversion_function: Optional[Callable] = None,
    ):
        """
        Prior for waveforms made of a variable number of components.

        Args:
            n_wf:
                Distribution of the number of components. Samples are
                cast to integers and clamped to `[1, max_components]`
            priors:
                Per-component priors, sampled with shape `(N, K)`
            shared_priors:
                Keys of `priors` sampled once per event and repeated
                across all of that event's components
            event_priors:
                Event-level priors (e.g. `hrss_tot`), sampled with shape
                `(N,)`. These can be used as `inference_params`
            max_components:
                If set, components are always padded to this length,
                which is required by slot-based models. Otherwise they
                are padded to the largest component count in each draw
            conversion_function:
                Callable applied to the sampled parameter dictionary

        Returns parameters for each key above, along with
        `n_components` of shape `(N,)` and a boolean `component_mask`
        of shape `(N, K)` marking the real (un-padded) components.
        Padded entries of per-component parameters are zero.
        """
        self.n_wf = n_wf
        self.shared_priors = shared_priors or []
        self.event_priors = event_priors or {}
        self.max_components = max_components
        super().__init__(priors, conversion_function)

    def __call__(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        parameters = self.sample_components(N, device)
        parameters = self.conversion_function(parameters)
        return parameters

    def sample_components(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Sample parameters before `conversion_function` is applied
        """
        rand_int = self.n_wf.sample((N,)).to(device).long()
        # ensure at least 1 component
        n_wavelets = torch.clamp(rand_int, min=1, max=self.max_components)
        n_max = self.max_components or int(n_wavelets.max())
        mask = (
            torch.arange(n_max, device=device).expand(N, n_max)
            < n_wavelets.unsqueeze(1)
        )

        parameters = {}

        for name, prior in self.priors.items():
            if name in self.shared_priors:
                value = prior.sample((N,)).to(device)
                value = value[:, None].expand(-1, n_max)
            else:
                value = prior.sample((N, n_max)).to(device)

            parameters[name] = value * mask

        for name, prior in self.event_priors.items():
            parameters[name] = prior.sample((N,)).to(device)

        parameters["n_components"] = n_wavelets.float()
        parameters["component_mask"] = mask
        return parameters

    def log_prob(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Log prior of event-level samples. `n_components` is discrete
        and skipped, so this is only defined up to a constant in it
        """
        first = samples[list(samples.keys())[0]]
        log_probs = torch.zeros(len(first), device=first.device)
        for param, tensor in samples.items():
            if param == "n_components":
                continue
            if param in self.event_priors:
                prior = self.event_priors[param]
            else:
                prior = self.priors[param]
            log_probs += prior.log_prob(tensor).to(first.device)
        return log_probs


def _log_uniform(
    low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    # elementwise log-uniform samples between tensors of bounds
    u = torch.rand(low.shape, dtype=low.dtype, device=low.device)
    return torch.exp(low.log() + u * (high.log() - low.log()))


class DurationConstrainedMultiWFPrior(AmplfiMultiWFPrior):
    # Q = factor * frequency * tau, where tau is the 1/e envelope
    # half-width, exp(-t^2 / tau^2), of each waveform
    QUALITY_FACTORS = {
        "morlet_gabor": 2 * math.pi,
        "sine_gaussian": math.sqrt(2) * math.pi,
    }

    def __init__(
        self,
        n_wf: torch.distributions.Distribution,
        burst_duration: torch.distributions.Distribution,
        frequency: tuple[float, float],
        quality: tuple[float, float],
        priors: dict[str, torch.distributions.Distribution],
        waveform: str = "morlet_gabor",
        envelope_width: float = 3.0,
        sample_rate: Optional[float] = None,
        band_width: float = 2.0,
        shared_priors: Optional[list[str]] = None,
        event_priors: Optional[
            dict[str, torch.distributions.Distribution]
        ] = None,
        max_components: Optional[int] = None,
        conversion_function: Optional[Callable] = None,
    ):
        """
        Multi-component prior in which an event-level `burst_duration`
        bounds every component, so each component's envelope lies
        inside `[-burst_duration / 2, burst_duration / 2]` around the
        kernel center.

        Each component has envelope `exp(-(t - shifts)^2 / tau^2)`,
        with `tau = quality / (factor * frequency)` (see `waveform`).
        Per event and then per component:

            tau_max   = burst_duration / (2 * envelope_width)
            frequency ~ LogUniform(f_low, f_high), with
                f_low  = max(frequency[0], quality[0] / (factor tau_max))
                f_high = min(max_frequency,
                             f_nyq - band_width / (pi tau_max))
            quality   ~ LogUniform(
                max(quality[0], factor frequency tau_min),
                min(quality[1], factor frequency tau_max),
            )
            shifts    ~ Uniform(-(D/2 - envelope_width tau),
                                D/2 - envelope_width tau)

        `tau_min = band_width / (pi (f_nyq - frequency))` is the band
        limit: a component's spectrum, `exp(-pi^2 tau^2 (f - f0)^2)`,
        must fall to `exp(-band_width^2)` by Nyquist, otherwise it is
        truncated (or aliased) at the sample rate and leaks outside
        its envelope. It is skipped if `sample_rate` is None. Since
        `tau_min` grows towards Nyquist, it also caps frequencies at
        `max_frequency`, above which no quality up to `quality[1]` is
        band-limited.

        Args:
            n_wf:
                Distribution of the number of components,
                as in `AmplfiMultiWFPrior`
            burst_duration:
                Distribution of the event-level burst duration in
                seconds. Returned as `burst_duration`, so it can be
                used as an inference parameter. Every sample must be
                at least `min_burst_duration`
            frequency:
                `(low, high)` bounds of the component frequencies (Hz)
            quality:
                `(low, high)` bounds of the component quality factors
            priors:
                Remaining per-component priors (e.g. `amplitude`,
                `phase`, `ellipticity`). `frequency`, `quality` and
                `shifts` are sampled by this class, so can't be given
            waveform:
                `"morlet_gabor"` (`tau = Q / (2 pi f)`) or
                `"sine_gaussian"` (`tau = Q / (sqrt(2) pi f)`), matching
                the `ml4gw.waveforms` generators
            envelope_width:
                Number of `tau` on each side of a component's center
                that must fit inside the burst duration. At 3, the
                envelope has fallen to `exp(-9)`
            sample_rate:
                Sample rate (Hz) of the generated waveforms, used for
                the band limit. Must match the data config's
                `sample_rate`
            band_width:
                Number of spectral half-widths `1 / (pi tau)` that must
                fit between a component's frequency and Nyquist
            shared_priors, event_priors, max_components,
            conversion_function:
                As in `AmplfiMultiWFPrior`
        """
        if waveform not in self.QUALITY_FACTORS:
            raise ValueError(
                f"waveform must be one of {list(self.QUALITY_FACTORS)}, "
                f"got {waveform!r}"
            )
        sampled = {"frequency", "quality", "shifts"} & set(priors)
        if sampled:
            raise ValueError(
                f"{sorted(sampled)} are sampled from burst_duration, "
                "so can't be given in priors"
            )
        event_priors = dict(event_priors or {})
        if "burst_duration" in event_priors:
            raise ValueError(
                "burst_duration is its own argument, so can't be given "
                "in event_priors"
            )
        event_priors["burst_duration"] = burst_duration

        if frequency[0] >= frequency[1] or quality[0] >= quality[1]:
            raise ValueError("frequency and quality must be (low, high)")
        if sample_rate is not None and frequency[1] > sample_rate / 2:
            raise ValueError(
                f"frequency upper bound {frequency[1]} exceeds Nyquist "
                f"({sample_rate / 2})"
            )

        self.frequency = frequency
        self.quality = quality
        self.waveform = waveform
        self.quality_factor = self.QUALITY_FACTORS[waveform]
        self.envelope_width = envelope_width
        self.sample_rate = sample_rate
        self.band_width = band_width
        super().__init__(
            n_wf,
            priors,
            shared_priors=shared_priors,
            event_priors=event_priors,
            max_components=max_components,
            conversion_function=conversion_function,
        )

    @property
    def max_frequency(self) -> float:
        """
        Highest component frequency. With the band limit, a component
        at frequency f needs quality >= factor f tau_min(f), which
        reaches quality[1] at f_nyq q / (q + factor band_width / pi)
        """
        if self.sample_rate is None:
            return self.frequency[1]
        f_nyq = self.sample_rate / 2
        q_max = self.quality[1]
        band_q = self.quality_factor * self.band_width / math.pi
        return min(self.frequency[1], f_nyq * q_max / (q_max + band_q))

    @property
    def min_burst_duration(self) -> float:
        """
        Shortest burst duration for which some frequency admits a
        component of the minimum quality
        """
        # tau_max must satisfy f_low(tau_max) < f_high(tau_max), i.e.
        # quality[0] / (factor tau) < max_frequency, and with the band
        # limit, both quality[0] / (factor tau) and frequency[0] must be
        # below f_nyq - band_width / (pi tau)
        q_term = self.quality[0] / self.quality_factor
        tau = q_term / self.max_frequency
        if self.sample_rate is not None:
            f_nyq = self.sample_rate / 2
            band_term = self.band_width / math.pi
            tau = max(
                tau,
                (q_term + band_term) / f_nyq,
                band_term / (f_nyq - self.frequency[0]),
            )
        return 2 * self.envelope_width * tau

    def sample_components(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        parameters = super().sample_components(N, device)
        mask = parameters["component_mask"]
        duration = parameters["burst_duration"]

        if (duration < self.min_burst_duration).any():
            raise ValueError(
                f"Sampled burst_duration {duration.min().item():.3g} s "
                "is below the shortest duration that fits a "
                f"band-limited component, {self.min_burst_duration:.3g} s"
            )

        # sample in float64, since log-uniform bounds that depend on
        # other parameters lose precision in float32
        n_max = mask.shape[-1]
        duration = duration.double()[:, None].expand(-1, n_max)
        tau_max = duration / (2 * self.envelope_width)
        factor = self.quality_factor

        f_low = torch.clamp(
            self.quality[0] / (factor * tau_max), min=self.frequency[0]
        )
        f_high = torch.full_like(tau_max, self.max_frequency)
        if self.sample_rate is not None:
            f_nyq = self.sample_rate / 2
            f_high = torch.clamp(
                f_nyq - self.band_width / (math.pi * tau_max),
                max=self.max_frequency,
            )
        frequency = _log_uniform(f_low, f_high)

        q_low = torch.full_like(frequency, self.quality[0])
        if self.sample_rate is not None:
            tau_min = self.band_width / (math.pi * (f_nyq - frequency))
            q_low = torch.clamp(factor * frequency * tau_min, min=q_low)
        q_high = torch.clamp(factor * frequency * tau_max, max=self.quality[1])
        quality = _log_uniform(q_low, q_high)

        tau = quality / (factor * frequency)
        half_room = (duration / 2 - self.envelope_width * tau).clamp_min(0)
        shifts = (2 * torch.rand_like(half_room) - 1) * half_room

        dtype = parameters["burst_duration"].dtype
        for name, value in [
            ("frequency", frequency),
            ("quality", quality),
            ("shifts", shifts),
        ]:
            parameters[name] = value.to(dtype) * mask
        return parameters


def relative_hrss(parameters: dict[str, torch.Tensor]):
    """
    Conversion function for `AmplfiMultiWFPrior` that adds
    `hrss_frac`, each component's share of the event's hrss.

    When `MultiWaveform` rescales the summed waveform to `hrss_tot`,
    the absolute per-component `hrss` is not identifiable, but
    `hrss_frac = hrss_i / sqrt(sum_j hrss_j^2)` is.
    """
    parameters["hrss_frac"] = _hrss_fraction(parameters["hrss"])
    return parameters


def _hrss_fraction(hrss: torch.Tensor) -> torch.Tensor:
    # rescale before squaring, since hrss ~1e-23 underflows in float32
    hrss = hrss / hrss.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    norm = hrss.pow(2).sum(dim=-1, keepdim=True).sqrt()
    return hrss / norm


def morlet_gabor_hrss(
    amplitude: torch.Tensor,
    quality: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
    ellipticity: torch.Tensor,
) -> torch.Tensor:
    """
    Analytic hrss of an `ml4gw.waveforms.MorletGabor` wavelet,
    from Parseval's theorem applied to its frequency-domain form:

        hrss^2 = (1 + ellipticity^2) amplitude^2 tau sqrt(pi / 2) / 2
                 * (1 + exp(-quality^2 / 2) cos(2 phase))

    with `tau = quality / (2 pi frequency)`. This neglects truncation
    of the wavelet by the finite time window. Unlike sine-Gaussians,
    the wavelet's amplitude is not its hrss, so components with equal
    amplitude can carry very different energy. Padded components
    (zero quality and frequency) give an hrss of zero
    """
    tau = quality / (2 * torch.pi * frequency)
    interference = 1 + torch.exp(-(quality**2) / 2) * torch.cos(2 * phase)
    hrss_sq = (
        (1 + ellipticity**2)
        * tau
        * (torch.pi / 2) ** 0.5
        / 2
        * interference
    )
    hrss = amplitude * hrss_sq.sqrt()
    return torch.nan_to_num(hrss, nan=0.0)


def relative_mg_hrss(parameters: dict[str, torch.Tensor]):
    """
    Conversion function for `AmplfiMultiWFPrior` with Morlet-Gabor
    components, which adds `hrss_frac` like `relative_hrss`, but
    computes each component's hrss from its `amplitude`, `quality`,
    `frequency`, `phase` and `ellipticity` (see `morlet_gabor_hrss`)
    """
    # only relative amplitudes matter, and amplitude ~1e-21
    # underflows in float32 once squared, so bring them to order 1
    amplitude = parameters["amplitude"]
    amplitude = amplitude / amplitude.amax(dim=-1, keepdim=True).clamp_min(
        1e-30
    )
    hrss = morlet_gabor_hrss(
        amplitude,
        parameters["quality"],
        parameters["frequency"],
        parameters["phase"],
        parameters["ellipticity"],
    )
    parameters["hrss_frac"] = _hrss_fraction(hrss)
    return parameters


def phase_to_cartesian(parameters: dict[str, torch.Tensor]):
    """
    Conversion function that adds `cos_phase` and `sin_phase`.
    Unlike `phase`, these are continuous across 0 and 2 pi,
    so they make better inference targets for flows
    """
    phase = parameters["phase"]
    parameters["cos_phase"] = torch.cos(phase)
    parameters["sin_phase"] = torch.sin(phase)
    return parameters


def multi_sg_conversion(parameters: dict[str, torch.Tensor]):
    """
    Conversion function for multi-component sine-Gaussians,
    applying `relative_hrss` and `phase_to_cartesian`
    """
    return phase_to_cartesian(relative_hrss(parameters))


def multi_mg_conversion(parameters: dict[str, torch.Tensor]):
    """
    Conversion function for multi-component Morlet-Gabor wavelets,
    applying `relative_mg_hrss` and `phase_to_cartesian`
    """
    return phase_to_cartesian(relative_mg_hrss(parameters))


class ParameterTransformer(torch.nn.Module):
    """
    Helper class for applying preprocessing
    transformations to inference parameters
    """

    def __init__(self, **transforms: Callable):
        super().__init__()
        self.transforms = transforms

    def forward(
        self,
        parameters: dict[str, torch.Tensor],
    ):
        # transform parameters
        # skip any transforms whose parameter isn't present, so the same
        # transformer can be applied to subsets of parameters
        transformed = {
            k: v(parameters[k])
            for k, v in self.transforms.items()
            if k in parameters
        }
        # update parameter dict
        parameters.update(transformed)
        return parameters


def dequantize(x: torch.Tensor) -> torch.Tensor:
    """
    Add `U(0, 1)` noise to an integer-valued parameter, so that a
    continuous density model sees a piecewise-constant density rather
    than delta functions. Flooring samples (see
    `AmplfiModel.requantize`) recovers the integer value
    """
    return x + torch.rand_like(x)


# log the event and per-component amplitudes for the multi-component
# sine-Gaussian use case. Padded components have zero hrss, so their
# log is -inf; slot datasets mask these out after transforming.
# `n_components` is dequantized so it can be an inference parameter
multi_sg_transformer = ParameterTransformer(
    hrss_tot=torch.log, hrss=torch.log, n_components=dequantize
)

# the same for multi-component Morlet-Gabor wavelets, whose
# per-component strength is `amplitude` rather than `hrss`
multi_mg_transformer = ParameterTransformer(
    hrss_tot=torch.log,
    amplitude=torch.log,
    burst_duration=torch.log,
    n_components=dequantize,
)
