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
    DurationConstrainedMultiWFPrior,
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


def build_duration_prior(
    waveform="morlet_gabor",
    sample_rate=2048,
    burst_duration=None,
    max_components=None,
):
    amplitude = "amplitude" if waveform == "morlet_gabor" else "hrss"
    shape = "ellipticity" if waveform == "morlet_gabor" else "eccentricity"
    return DurationConstrainedMultiWFPrior(
        n_wf=Uniform(1, 8),
        burst_duration=burst_duration or LogUniform(5e-3, 0.2),
        frequency=(10, sample_rate / 2),
        quality=(0.3, 90),
        priors={
            amplitude: LogUniform(0.07, 1),
            "phase": Uniform(0, 6.283),
            shape: Uniform(0, 1),
        },
        waveform=waveform,
        sample_rate=sample_rate,
        shared_priors=[shape],
        event_priors={"hrss_tot": LogUniform(1e-23, 5e-20)},
        max_components=max_components,
        conversion_function=(
            multi_mg_conversion
            if waveform == "morlet_gabor"
            else multi_sg_conversion
        ),
    )


@pytest.mark.parametrize("waveform", ["morlet_gabor", "sine_gaussian"])
def test_duration_prior_bounds(waveform):
    N, sample_rate = 5000, 2048
    prior = build_duration_prior(waveform, sample_rate, max_components=8)
    parameters = prior(N)
    mask = parameters["component_mask"]
    assert parameters["burst_duration"].shape == (N,)
    for key in ("frequency", "quality", "shifts"):
        assert parameters[key].shape == (N, 8)
        assert (parameters[key][~mask] == 0).all()

    duration = parameters["burst_duration"][:, None].expand_as(mask)[mask]
    f = parameters["frequency"][mask].double()
    q = parameters["quality"][mask].double()
    shifts = parameters["shifts"][mask].double()
    tau = q / (prior.quality_factor * f)

    # float32 storage, so allow for rounding
    eps = 1e-5
    assert (f >= 10 * (1 - eps)).all() and (f <= 1024 * (1 + eps)).all()
    assert (q >= 0.3 * (1 - eps)).all() and (q <= 90 * (1 + eps)).all()
    # every envelope fits in the burst duration
    edge = shifts.abs() + prior.envelope_width * tau
    assert (edge <= duration.double() / 2 * (1 + eps)).all()
    # and every spectrum is band-limited
    band = f + prior.band_width / (torch.pi * tau)
    assert (band <= sample_rate / 2 * (1 + eps)).all()

    # burst_duration is an event prior, so has a log prob
    log_prob = prior.log_prob(
        {"burst_duration": parameters["burst_duration"]}
    )
    assert torch.isfinite(log_prob).all()


def test_duration_prior_min_duration():
    prior = build_duration_prior()
    d_min = prior.min_burst_duration
    # matches the bound worked out for 2048 Hz, k = 3, c = 2
    assert 3.9e-3 < d_min < 4.1e-3

    # just above the minimum, frequency and quality ranges are valid
    prior = build_duration_prior(
        burst_duration=Uniform(d_min * 1.001, d_min * 1.01)
    )
    parameters = prior(1000)
    mask = parameters["component_mask"]
    assert torch.isfinite(parameters["quality"][mask]).all()
    assert (parameters["quality"][mask] > 0).all()

    # below it, sampling fails loudly
    prior = build_duration_prior(
        burst_duration=Uniform(d_min * 0.5, d_min * 0.9)
    )
    with pytest.raises(ValueError, match="below the shortest duration"):
        prior(10)


def test_duration_prior_validation():
    kwargs = dict(
        n_wf=Uniform(1, 3),
        burst_duration=LogUniform(5e-3, 0.2),
        frequency=(10, 1024),
        quality=(0.3, 90),
        sample_rate=2048,
    )
    with pytest.raises(ValueError, match="sampled from burst_duration"):
        DurationConstrainedMultiWFPrior(
            priors={"shifts": Uniform(-0.1, 0.1)}, **kwargs
        )
    with pytest.raises(ValueError, match="waveform must be"):
        DurationConstrainedMultiWFPrior(
            priors={}, waveform="gaussian", **kwargs
        )
    with pytest.raises(ValueError, match="exceeds Nyquist"):
        DurationConstrainedMultiWFPrior(
            priors={}, **{**kwargs, "sample_rate": 1024}
        )


@requires_mg
def test_duration_generator():
    N, sample_rate = 64, 512
    prior = build_duration_prior(
        sample_rate=sample_rate,
        burst_duration=LogUniform(0.03, 0.5),
        max_components=4,
    )
    generator = MultiWFGenerator(
        generator=MorletGabor,
        sample_rate=sample_rate,
        kernel_length=2,
        fduration=1,
        inference_params=[],
        num_val_waveforms=1,
        num_test_waveforms=1,
        num_fit_params=1,
        training_prior=prior,
    )
    parameters = prior(N)
    hc, hp = generator(**parameters)
    assert torch.isfinite(hc).all() and torch.isfinite(hp).all()

    # h_plus is confined to the burst duration (h_cross, the Hilbert
    # transform of h_plus, has slowly decaying tails at low quality)
    times = torch.arange(hp.shape[-1]) / sample_rate - generator.duration / 2
    inside = times.abs() <= parameters["burst_duration"][:, None] / 2
    energy = hp.double() ** 2
    frac = (energy * inside).sum(-1) / energy.sum(-1)
    assert (frac > 0.999).all()

    # the prior's band limit has to use the generator's sample rate
    with pytest.raises(ValueError, match="sample_rate"):
        MultiWFGenerator(
            generator=MorletGabor,
            sample_rate=2 * sample_rate,
            kernel_length=2,
            fduration=1,
            inference_params=[],
            num_val_waveforms=1,
            num_test_waveforms=1,
            num_fit_params=1,
            training_prior=prior,
        )
