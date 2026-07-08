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
    def __init__(self, n_wf, priors, shared_priors = {}, conversion_function = None):
        self.n_wf = n_wf
        self.shared_priors = shared_priors
        super().__init__(priors, conversion_function)


    def __call__(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        rand_int = self.n_wf.sample((N,)).to(device).long()
        n_wavelets = torch.clamp(rand_int, min=1)  # ensure at least 1 sine gaussian
        n_max = n_wavelets.max()
        mask = (
            torch.arange(n_max).expand(N, n_max) 
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


        parameters = self.conversion_function(parameters)
        return parameters


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
        transformed = {k: v(parameters[k]) for k, v in self.transforms.items()}
        # update parameter dict
        parameters.update(transformed)
        return parameters
