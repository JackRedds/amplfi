import torch

from amplfi.train.prior import AmplfiMultiWFPrior


def make_prior(n_max=4):
    return AmplfiMultiWFPrior(
        n_components=torch.distributions.Uniform(1, n_max + 1),
        priors={"hrss_tot": torch.distributions.Uniform(1e-21, 1e-20)},
    )


def test_multi_wf_prior_shapes_and_device():
    prior = make_prior(n_max=4)
    batch = 32
    params = prior(batch, device="cpu")

    assert "n_components" in params
    assert "hrss_tot" in params
    for v in params.values():
        assert isinstance(v, torch.Tensor)
        assert v.shape == (batch,)
        assert v.device.type == "cpu"

    n_components = params["n_components"]
    assert n_components.dtype == torch.long
    assert torch.all((n_components >= 1) & (n_components <= 4))
