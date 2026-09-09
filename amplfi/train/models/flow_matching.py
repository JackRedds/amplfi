from ..architectures.flow_matching import FlowMatchingArchitecture
from .posterior_sampler import PosteriorSamplerModel


class FlowMatchingModel(PosteriorSamplerModel):
    """
    A LightningModule for training flow matching models for
    likelihood-free parameter estimation.

    Args:
        *args, **kwargs:
            See arguments in
            `amplfi.train.models.posterior_sampler.PosteriorSamplerModel`
        arch:
            Neural network architecture to train.
            This should be a subclass of `FlowMatchingArchitecture`.
    """

    def __init__(
        self, *args, arch: FlowMatchingArchitecture, **kwargs
    ) -> None:
        super().__init__(*args, arch=arch, **kwargs)
