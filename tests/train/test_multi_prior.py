import pytest
import torch
from ml4gw.distributions import LogUniform
import ml4gw.waveforms
from ml4gw.waveforms import SineGaussian
from torch.distributions import Uniform

from amplfi.train.data.waveforms.generator.multi import MultiWFGenerator
from amplfi.train.models.base import AmplfiModel
from amplfi.train.prior import (
    AmplfiMultiWFPrior,
    dequantize,
    morlet_gabor_hrss,
    multi_mg_conversion,
    multi_mg_transformer,
    multi_sg_conversion,
    multi_sg_transformer,
)

# MorletGabor is only in newer versions of ml4gw
MorletGabor = getattr(ml4gw.waveforms, "MorletGabor", None)
requires_mg = pytest.mark.skipif(
    MorletGabor is None, reason="installed ml4gw has no MorletGabor"
)


def build_prior(max_components=None):
    return AmplfiMultiWFPrior(
        n_wf=Uniform(1, 5),
        priors={
            "frequency": Uniform(32, 1024),
            "quality": Uniform(2, 100),
            "hrss": LogUniform(1e-23, 5e-20),
            "phase": Uniform(0, 6.283),
            "eccentricity": Uniform(0, 1),
            "shifts": Uniform(-0.1, 0.1),
        },
        event_priors={"hrss_tot": LogUniform(1e-23, 5e-20)},
        max_components=max_components,
        conversion_function=multi_sg_conversion,
    )


def test_prior_shapes():
    N = 100
    parameters = build_prior(max_components=6)(N)
    mask = parameters["component_mask"]
    assert mask.shape == (N, 6)
    assert parameters["frequency"].shape == (N, 6)
    assert parameters["hrss_tot"].shape == (N,)
    assert (mask.sum(-1) == parameters["n_components"]).all()
    assert (parameters["frequency"][~mask] == 0).all()

    # the real components' relative amplitudes have unit norm
    norm = parameters["hrss_frac"].pow(2).sum(-1)
    torch.testing.assert_close(norm, torch.ones(N))

    # phase is also given as a point on the unit circle
    cos, sin = parameters["cos_phase"], parameters["sin_phase"]
    torch.testing.assert_close(cos**2 + sin**2, torch.ones_like(cos))
    recovered = torch.remainder(torch.atan2(sin, cos), 2 * torch.pi)
    torch.testing.assert_close(
        recovered[mask], parameters["phase"][mask], atol=1e-4, rtol=0
    )

    # without max_components, pad to the largest component count
    parameters = build_prior()(N)
    n_max = int(parameters["n_components"].max())
    assert parameters["frequency"].shape == (N, n_max)


def test_transformer_skips_missing_keys():
    parameters = {"hrss_tot": torch.ones(3)}
    transformed = multi_sg_transformer(parameters)
    assert set(transformed) == {"hrss_tot"}


def test_dequantize_round_trip():
    n_components = build_prior()(1000)["n_components"]
    dequantized = multi_sg_transformer(
        {"n_components": n_components.clone()}
    )["n_components"]
    assert (dequantized >= n_components).all()
    assert (dequantized < n_components + 1).all()
    # noise actually fills the unit interval
    assert (dequantized - n_components).std() > 0.2

    # flooring dequantized samples recovers the count, and
    # only discrete parameters are floored
    class Model:
        inference_params = ["hrss_tot", "n_components"]
        discrete_params = AmplfiModel.discrete_params

    samples = torch.column_stack([dequantized, dequantized])
    requantized = AmplfiModel.requantize(Model(), samples)
    torch.testing.assert_close(requantized[:, 1], n_components)
    torch.testing.assert_close(requantized[:, 0], dequantized)

    # samples below the support are clamped to one component
    low = torch.tensor([[0.0, 0.3], [0.0, -2.0]])
    assert (AmplfiModel.requantize(Model(), low)[:, 1] == 1).all()
    assert dequantize(torch.zeros(3)).max() < 1


def test_generator():
    N = 8
    generator = MultiWFGenerator(
        generator=SineGaussian,
        sample_rate=256,
        kernel_length=2,
        fduration=1,
        inference_params=[],
        num_val_waveforms=1,
        num_test_waveforms=1,
        num_fit_params=1,
        training_prior=build_prior(max_components=4),
    )
    parameters = build_prior(max_components=4)(N)
    hc, hp = generator(**parameters)
    assert hc.shape == hp.shape == (N, 3 * 256)
    assert torch.isfinite(hc).all() and torch.isfinite(hp).all()

    # summed waveforms are rescaled to hrss_tot
    # (in float64, since squaring float32 strain underflows),
    # including when every component is faint
    parameters["hrss"][:4] *= 1e-3
    hc, hp = generator(**parameters)
    hc, hp = hc.double(), hp.double()
    hrss = torch.sqrt((hc**2 + hp**2).sum(-1) / 256)
    torch.testing.assert_close(
        hrss, parameters["hrss_tot"].double(), rtol=1e-3, atol=0
    )


def build_mg_prior(max_components=None):
    return AmplfiMultiWFPrior(
        n_wf=Uniform(1, 5),
        priors={
            "frequency": Uniform(32, 1024),
            "quality": Uniform(2, 100),
            "amplitude": LogUniform(1e-23, 5e-20),
            "phase": Uniform(0, 6.283),
            "ellipticity": Uniform(-1, 1),
            "shifts": Uniform(-0.1, 0.1),
        },
        event_priors={"hrss_tot": LogUniform(1e-23, 5e-20)},
        max_components=max_components,
        conversion_function=multi_mg_conversion,
    )


@requires_mg
def test_morlet_gabor_hrss():
    # low qualities, so the cos(2 phase) interference term matters
    N, sample_rate = 64, 4096
    params = {
        "quality": torch.rand(N, dtype=torch.float64) * 4 + 1,
        "frequency": torch.rand(N, dtype=torch.float64) * 200 + 50,
        "amplitude": torch.ones(N, dtype=torch.float64),
        "phase": torch.rand(N, dtype=torch.float64) * 2 * torch.pi,
        "ellipticity": torch.rand(N, dtype=torch.float64) * 2 - 1,
        "shifts": torch.zeros(N, dtype=torch.float64),
    }
    cross, plus = MorletGabor(sample_rate, 4)(**params)
    numerical = ((plus**2 + cross**2).sum(-1) / sample_rate).sqrt()
    analytic = morlet_gabor_hrss(
        params["amplitude"],
        params["quality"],
        params["frequency"],
        params["phase"],
        params["ellipticity"],
    )
    torch.testing.assert_close(analytic, numerical, rtol=1e-3, atol=0)

    # padded components have zero hrss rather than nan
    zero = torch.zeros(3)
    assert (morlet_gabor_hrss(zero, zero, zero, zero, zero) == 0).all()


def test_mg_prior():
    N = 100
    parameters = build_mg_prior(max_components=6)(N)
    mask = parameters["component_mask"]
    frac = parameters["hrss_frac"]
    assert torch.isfinite(frac).all()
    assert (frac[~mask] == 0).all()
    torch.testing.assert_close(frac.pow(2).sum(-1), torch.ones(N))

    transformed = multi_mg_transformer(
        {k: v.clone() for k, v in parameters.items()}
    )
    torch.testing.assert_close(
        transformed["amplitude"][mask], parameters["amplitude"][mask].log()
    )
    torch.testing.assert_close(
        transformed["hrss_tot"], parameters["hrss_tot"].log()
    )


@requires_mg
def test_mg_generator():
    N = 8
    prior = build_mg_prior(max_components=4)
    generator = MultiWFGenerator(
        generator=MorletGabor,
        sample_rate=256,
        kernel_length=2,
        fduration=1,
        inference_params=[],
        num_val_waveforms=1,
        num_test_waveforms=1,
        num_fit_params=1,
        training_prior=prior,
    )
    parameters = prior(N)
    # keep frequencies below Nyquist
    parameters["frequency"] = parameters["frequency"] / 8
    # every component faint, which underflows without rescaling
    parameters["amplitude"][:4] *= 1e-3
    hc, hp = generator(**parameters)
    assert hc.shape == hp.shape == (N, 3 * 256)
    assert torch.isfinite(hc).all() and torch.isfinite(hp).all()

    hc, hp = hc.double(), hp.double()
    hrss = torch.sqrt((hc**2 + hp**2).sum(-1) / 256)
    torch.testing.assert_close(
        hrss, parameters["hrss_tot"].double(), rtol=1e-3, atol=0
    )
