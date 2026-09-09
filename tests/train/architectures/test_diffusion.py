import pytest
import torch

from amplfi.train.architectures.diffusion import DDPM
from amplfi.train.architectures.embeddings import ResNet


@pytest.fixture(params=[2, 10])
def param_dim(request):
    return request.param


@pytest.fixture(params=[512, 1024])
def strain_dim(request):
    return request.param


@pytest.fixture(params=[10, 12])
def context_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


def test_diffusion_loss(param_dim, strain_dim, context_dim, n_ifos):
    data = torch.randn((100, param_dim))
    strain = (torch.randn((100, n_ifos, strain_dim)), None)

    embedding = ResNet(n_ifos, layers=[1, 1], context_dim=context_dim)
    model = DDPM(
        param_dim,
        embedding,
        num_timesteps=10,
        hidden_features=32,
        num_blocks=2,
        time_embed_dim=16,
    )

    loss = model.loss(data, context=strain)
    assert loss.shape == (len(data),)


def test_diffusion_sample(param_dim, context_dim, n_ifos):
    batch = 4
    n_samples = 5
    strain = (torch.randn((batch, n_ifos, 512)), None)

    embedding = ResNet(n_ifos, layers=[1, 1], context_dim=context_dim)
    model = DDPM(
        param_dim,
        embedding,
        num_timesteps=10,
        sampling_timesteps=5,
        hidden_features=32,
        num_blocks=2,
        time_embed_dim=16,
    )

    samples = model.sample(n_samples, context=strain)
    assert samples.shape == (n_samples, batch, param_dim)
    assert torch.isfinite(samples).all()
