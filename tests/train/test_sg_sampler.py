import torch

from amplfi.train.data.utils.utils import ParameterSamplerMultiSG


def test_parameter_sampler_multi_sg_shapes_and_device():
    ps = ParameterSamplerMultiSG(n_max=4)
    batch = 5
    params = ps(batch, device="cpu")

    # check expected keys
    assert "n_components" in params
    # check shapes and device
    for k, v in params.items():
        assert isinstance(v, torch.Tensor)
        assert v.shape == (batch,)
        assert v.device.type == "cpu"

    # n_components should be in 1..n_max
    n_comp = params["n_components"]
    assert torch.all((n_comp >= 1) & (n_comp <= 4))
