import pytest
import torch

from amplfi.train.architectures.embeddings import (
    FrequencyPsd,
    MultiModalPsd,
    MultiModalPsdTransformer,
    MultiModalTransformer,
    TimeDomainTransformer,
)

N_IFOS = 2
LENGTH = 512
NUM_FREQS = 200

TRANSFORMER_KWARGS = dict(
    transformer_layers=1,
    transformer_heads=2,
    transformer_dim_feedforward=32,
)


def context(B=4):
    strain = torch.randn(B, N_IFOS, LENGTH)
    asds = torch.rand(B, N_IFOS, NUM_FREQS) * 1e-23 + 1e-23
    return strain, asds


def time_domain():
    return TimeDomainTransformer(
        N_IFOS, context_dim=16, layers=[8, 16], **TRANSFORMER_KWARGS
    )


def multimodal(cls):
    return cls(
        N_IFOS,
        time_context_dim=16,
        time_layers=[8, 16],
        freq_context_dim=8,
        freq_layers=[8],
        **TRANSFORMER_KWARGS,
    )


@pytest.mark.parametrize(
    "build",
    [
        time_domain,
        lambda: multimodal(MultiModalTransformer),
        lambda: multimodal(MultiModalPsdTransformer),
    ],
)
def test_forward_tokens(build):
    embedding = build().eval()
    X = context()
    with torch.no_grad():
        summary, groups = embedding.forward_tokens(X)
        expected = embedding(X)

    torch.testing.assert_close(summary, expected)
    assert summary.shape == (4, embedding.context_dim)
    assert len(groups) == len(embedding.token_dims)
    for group, dim in zip(groups, embedding.token_dims, strict=True):
        assert group.ndim == 3
        assert group.shape[0] == 4 and group.shape[-1] == dim


@pytest.mark.parametrize(
    "build",
    [
        lambda: multimodal(MultiModalPsdTransformer),
        lambda: MultiModalPsd(N_IFOS, 8, 8, [1, 1], [1, 1]),
        lambda: FrequencyPsd(N_IFOS, 8, [1, 1]),
    ],
)
def test_asds_are_not_modified(build):
    # embedding the same context twice (e.g. to sample and then
    # evaluate log probabilities) must give the same result
    embedding = build().eval()
    strain, asds = context()
    original = asds.clone()
    with torch.no_grad():
        first = embedding((strain, asds))
        second = embedding((strain, asds))
    torch.testing.assert_close(asds, original)
    torch.testing.assert_close(first, second)
