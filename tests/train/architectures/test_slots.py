import pytest
import torch

from amplfi.train.architectures import DDPM, NSF, RectifiedFlow, SlotHead
from amplfi.train.architectures.embeddings import (
    MultiModalPsdTransformer,
    MultiModalTransformer,
    ResNet,
    TimeDomainTransformer,
)

N_IFOS = 2
STRAIN_DIM = 512
NUM_PARAMS = 3
SLOT_PARAMS = ["frequency", "quality", "phase"]


@pytest.fixture(params=[3, 7])
def max_components(request):
    # 7 exercises the scipy matching fallback
    return request.param


TRANSFORMER_KWARGS = dict(
    transformer_layers=1,
    transformer_heads=2,
    transformer_dim_feedforward=32,
)


@pytest.fixture(
    params=["resnet", "transformer", "multimodal", "multimodal_psd"]
)
def embedding(request):
    if request.param == "resnet":
        return ResNet(N_IFOS, layers=[1, 1], context_dim=16)
    if request.param == "transformer":
        return TimeDomainTransformer(
            N_IFOS, context_dim=16, layers=[8, 16], **TRANSFORMER_KWARGS
        )
    cls = {
        "multimodal": MultiModalTransformer,
        "multimodal_psd": MultiModalPsdTransformer,
    }[request.param]
    return cls(
        N_IFOS,
        time_context_dim=16,
        time_layers=[8, 16],
        freq_context_dim=8,
        freq_layers=[8],
        **TRANSFORMER_KWARGS,
    )


def make_context(B):
    strain = torch.randn(B, N_IFOS, STRAIN_DIM)
    asds = torch.rand(B, N_IFOS, 200) * 1e-23 + 1e-23
    return strain, asds


@pytest.fixture(params=["nsf", "ddpm", "rectified_flow"])
def host(request):
    return request.param


def slot_head(max_components):
    return SlotHead(
        slot_params=SLOT_PARAMS,
        max_components=max_components,
        slot_dim=16,
        decoder_heads=2,
        transforms=2,
        hidden_features=[16],
    )


def build(host, embedding, head):
    if host == "nsf":
        return NSF(
            NUM_PARAMS,
            embedding,
            transforms=2,
            hidden_features=[16],
            slot_head=head,
        )
    if host == "ddpm":
        return DDPM(
            NUM_PARAMS,
            embedding,
            hidden_features=16,
            num_blocks=1,
            num_timesteps=10,
            slot_head=head,
        )
    return RectifiedFlow(
        NUM_PARAMS,
        embedding,
        hidden_features=16,
        num_blocks=1,
        sampling_steps=4,
        slot_head=head,
    )


def make_targets(B, max_components):
    counts = torch.randint(1, max_components + 1, (B,))
    mask = torch.arange(max_components)[None] < counts[:, None]
    slots = torch.randn(B, max_components, len(SLOT_PARAMS)) * mask[..., None]
    return {
        "global": torch.randn(B, NUM_PARAMS),
        "slots": slots,
        "mask": mask,
    }


def test_slot_losses_and_sampling(host, max_components, embedding):
    B = 6
    head = slot_head(max_components)
    arch = build(host, embedding, head)
    # slots attend to the embedding's token groups, if it has them
    num_groups = len(getattr(embedding, "token_dims", []))
    assert head.use_tokens == (num_groups > 0)
    if head.use_tokens:
        assert len(head.token_projs) == num_groups + 1
    context = make_context(B)
    targets = make_targets(B, max_components)

    losses = arch.slot_losses(context, targets)
    for v in losses.values():
        assert v.shape == (B,)
        assert torch.isfinite(v).all()
    losses["loss"].mean().backward()
    # the slot head trains alongside the host architecture
    assert arch.slot_head.queries.grad is not None

    arch.eval()
    n = 10
    samples, slots = arch.sample_with_slots(n, context)
    assert samples.shape == (n, B, NUM_PARAMS)

    K, S = max_components, len(SLOT_PARAMS)
    assert slots["slots"].shape == (n, B, K, S)
    assert slots["mask"].shape == (n, B, K)
    assert slots["count_probs"].shape == (B, K)
    counts = slots["n_components"]
    assert ((counts >= 1) & (counts <= K)).all()
    assert (slots["mask"].sum(-1) == counts).all()
    assert (slots["slots"][~slots["mask"]] == 0).all()


def test_without_slot_head_is_unchanged(host):
    torch.manual_seed(0)
    embedding = ResNet(N_IFOS, layers=[1, 1], context_dim=16)
    arch = build(host, embedding, None)
    assert arch.slot_head is None
    assert not any("slot" in k for k in arch.state_dict())

    B = 4
    context = make_context(B)
    arch.eval()
    samples = arch.sample(5, context)
    assert samples.shape == (5, B, NUM_PARAMS)


def test_loss_is_permutation_invariant(max_components):
    torch.manual_seed(0)
    B = 4
    head = slot_head(max_components)
    arch = build("nsf", ResNet(N_IFOS, layers=[1, 1], context_dim=16), head)
    arch.eval()
    context = make_context(B)
    targets = make_targets(B, max_components)
    # use the same number of components everywhere so
    # one permutation of the real components applies to every event
    K = max_components
    targets["mask"][:] = True
    targets["slots"] = torch.randn(B, K, len(SLOT_PARAMS))

    perm = torch.randperm(K)
    permuted = dict(targets, slots=targets["slots"][:, perm])
    loss = arch.slot_losses(context, targets)["slot_loss"]
    permuted_loss = arch.slot_losses(context, permuted)["slot_loss"]
    torch.testing.assert_close(loss, permuted_loss)


def test_match_finds_optimal_assignment(max_components):
    head = slot_head(max_components)
    K = max_components
    B = 3
    # component j fits slot perm[j] best by a wide margin
    perms = torch.stack([torch.randperm(K) for _ in range(B)])
    log_probs = torch.full((B, K, K), -10.0)
    for b in range(B):
        log_probs[b, perms[b], torch.arange(K)] = 0.0
    # pad the last component of the first event, which
    # must then be assigned to a padded slot
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, -1] = False
    log_probs[0, :, -1] = 0.0
    log_probs[0, -1, :] = 0.0

    assignment = head.match(log_probs, mask)
    assert (assignment[1:] == perms[1:]).all()
    assert assignment[0, -1] == K - 1
    assert sorted(assignment[0].tolist()) == list(range(K))
