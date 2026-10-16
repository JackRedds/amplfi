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

        parameters = self.conversion_function(parameters)
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
    hrss_tot=torch.log, amplitude=torch.log, n_components=dequantize
)
