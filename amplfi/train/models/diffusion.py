from ..architectures.diffusion import DiffusionArchitecture
from .posterior_sampler import PosteriorSamplerModel


class DiffusionModel(PosteriorSamplerModel):
    """
    A LightningModule for training diffusion models for
    likelihood-free parameter estimation.

    Args:
        *args, **kwargs:
            See arguments in
            `amplfi.train.models.posterior_sampler.PosteriorSamplerModel`
        arch:
            Neural network architecture to train.
            This should be a subclass of `DiffusionArchitecture`.
    """

    def __init__(self, *args, arch: DiffusionArchitecture, **kwargs) -> None:
        super().__init__(*args, arch=arch, **kwargs)
