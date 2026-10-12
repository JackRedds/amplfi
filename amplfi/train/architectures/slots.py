import itertools
from typing import Optional, Sequence

import numpy as np
import torch
import zuko
from scipy.optimize import linear_sum_assignment

from .embeddings.base import Embedding

Tensor = torch.Tensor

# above this many slots, enumerating every permutation
# is too expensive and matching falls back to scipy
MAX_BRUTE_FORCE_SLOTS = 6


class SlotHead(torch.nn.Module):
    """
    Slot-based head for trans-dimensional inference on signals made
    of a variable, unordered set of components, following "SlotFlow:
    Amortized Trans-Dimensional Inference with Slot-Based Normalizing
    Flows". It is attached to an existing posterior architecture
    (`NSF`, `RectifiedFlow` or `DDPM`) via their `slot_head` argument,
    and shares that architecture's embedding network. The posterior
    then factorizes as

        p(K, theta_1..K, phi | d) = p(phi | d) p(K | d)
                                    prod_i q(theta_i | slot_i(d, K))

    where `phi` are the event-level (global) parameters modeled by
    the host architecture, `K` is the number of components and
    `theta_i` are the per-component (slot) parameters. `K` is
    predicted by a classifier, and `K` learned slot queries
    (conditioned on `K`) cross-attend to the embedded data to give
    one context per component. A single conditional NSF shared across
    slots models each component's parameters. Since the components
    are unordered, the training loss matches components to slots with
    the assignment that maximizes their likelihood.

    Embeddings defining `forward_tokens` and `token_dims` (e.g.
    `TimeDomainTransformer`, `MultiModalTransformer` and
    `MultiModalPsdTransformer`) are cross-attended to token by token.
    `forward_tokens` returns the context vector and a list of token
    groups (e.g. time and frequency tokens), with feature sizes
    `token_dims`. Each group is projected to `slot_dim` and tagged
    with a learned embedding of its group, so slots can tell, e.g.,
    time from frequency tokens. For any other embedding, its context
    vector is projected to `num_memory_tokens` tokens.

    Args:
        slot_params:
            Names of the per-component parameters
        max_components:
            Maximum number of components
        slot_dim:
            Dimension of the slot contexts
        decoder_layers:
            Number of transformer decoder layers producing the slots
        decoder_heads:
            Number of attention heads in the slot decoder
        num_memory_tokens:
            Number of tokens to project the embedding's context
            vector to, for embeddings without `forward_tokens`
        transforms:
            Number of transforms in the per-slot flow
        hidden_features:
            Hidden layer sizes of the per-slot flow hyper networks
        bins:
            Number of spline bins in the per-slot flow
        randperm:
            Whether to randomly permute features between transforms
        dropout:
            Dropout in the slot decoder
    """

    def __init__(
        self,
        slot_params: list[str],
        max_components: int,
        slot_dim: int = 128,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        num_memory_tokens: int = 8,
        transforms: int = 8,
        hidden_features: Sequence[int] = (256, 256),
        bins: int = 8,
        randperm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slot_params = slot_params
        self.num_slot_params = len(slot_params)
        self.max_components = max_components
        self.slot_dim = slot_dim
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.num_memory_tokens = num_memory_tokens
        self.transforms = transforms
        self.hidden_features = hidden_features
        self.bins = bins
        self.randperm = randperm
        self.dropout = dropout

        self.brute_force = max_components <= MAX_BRUTE_FORCE_SLOTS
        perms = []
        if self.brute_force:
            perms = list(itertools.permutations(range(max_components)))
        self.register_buffer(
            "perms", torch.tensor(perms, dtype=torch.long), persistent=False
        )

    def build(self, embedding_net: Embedding):
        """
        Build the networks, which depend on the host
        architecture's embedding. Called by the host architecture
        """
        context_dim = embedding_net.context_dim
        slot_dim = self.slot_dim
        self.use_tokens = hasattr(embedding_net, "forward_tokens")
        if self.use_tokens:
            # the context vector is attended to as its own token
            token_dims = [context_dim] + list(embedding_net.token_dims)
            self.token_projs = torch.nn.ModuleList(
                [torch.nn.Linear(dim, slot_dim) for dim in token_dims]
            )
            self.token_types = torch.nn.Parameter(
                torch.randn(len(token_dims), slot_dim) * 0.02
            )
        else:
            self.memory_proj = torch.nn.Linear(
                context_dim, self.num_memory_tokens * slot_dim
            )

        # slot queries, offset by an embedding of the number of components
        self.queries = torch.nn.Parameter(
            torch.randn(self.max_components, slot_dim) * 0.02
        )
        self.count_embedding = torch.nn.Embedding(
            self.max_components, slot_dim
        )
        layer = torch.nn.TransformerDecoderLayer(
            d_model=slot_dim,
            nhead=self.decoder_heads,
            dim_feedforward=4 * slot_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(
            layer, self.decoder_layers
        )
        self.slot_norm = torch.nn.LayerNorm(slot_dim)

        self.count_head = torch.nn.Sequential(
            torch.nn.Linear(context_dim, 2 * context_dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * context_dim, self.max_components),
        )

        self.slot_flow = zuko.flows.NSF(
            self.num_slot_params,
            context=slot_dim,
            transforms=self.transforms,
            bins=self.bins,
            randperm=self.randperm,
            hidden_features=self.hidden_features,
        )

    # ================================================ #
    # Network components
    # ================================================ #

    def memory(
        self, summary: Tensor, tokens: Optional[list[Tensor]]
    ) -> Tensor:
        """
        Tokens for the slots to attend to, `(B, T, slot_dim)`, from
        the embedding's token groups if it has them, or else its
        context vector `summary`
        """
        if self.use_tokens:
            groups = [summary[:, None]] + list(tokens)
            memory = [
                proj(group) + token_type
                for proj, group, token_type in zip(
                    self.token_projs, groups, self.token_types, strict=True
                )
            ]
            return torch.cat(memory, dim=1)
        memory = self.memory_proj(summary)
        return memory.view(-1, self.num_memory_tokens, self.slot_dim)

    def decode_slots(self, memory: Tensor, num_components: Tensor) -> Tensor:
        """
        Produce `max_components` slot contexts of shape
        `(B, K, slot_dim)` given `num_components` of shape `(B,)`.
        Only the first `num_components` slots of each event are used
        """
        B = len(memory)
        queries = self.queries.expand(B, -1, -1)
        queries = queries + self.count_embedding(num_components - 1)[:, None]

        # padded slots don't take part in slot self attention
        arange = torch.arange(self.max_components, device=memory.device)
        padding = arange[None] >= num_components[:, None]
        slots = self.decoder(queries, memory, tgt_key_padding_mask=padding)
        return self.slot_norm(slots)

    # ================================================ #
    # Training
    # ================================================ #

    def match(self, slot_log_probs: Tensor, mask: Tensor) -> Tensor:
        """
        Find the assignment of components to slots that
        maximizes the total log likelihood of the components.

        Args:
            slot_log_probs:
                Tensor of shape `(B, K, K)`, where entry
                `[b, i, j]` is the log likelihood of component `j`
                under slot `i`
            mask:
                Boolean tensor of shape `(B, K)` of real components.
                Real components are always the first in each row

        Returns:
            Tensor of shape `(B, K)` giving the slot assigned to
            each component. Padded components are assigned to
            padded slots
        """
        cost = -slot_log_probs.detach()
        # forbid assigning real components to padded slots
        # and vice versa, and make padded pairs free
        valid = mask[:, :, None] & mask[:, None, :]
        mixed = mask[:, :, None] ^ mask[:, None, :]
        cost = torch.where(valid, cost, torch.zeros_like(cost))
        cost = torch.where(mixed, torch.full_like(cost, 1e9), cost)
        cost = torch.nan_to_num(cost, nan=1e9, posinf=1e9, neginf=-1e9)

        if self.brute_force:
            # cost of each permutation, where perm[j]
            # is the slot assigned to component j
            arange = torch.arange(self.max_components, device=cost.device)
            perm_costs = cost[:, self.perms, arange].sum(-1)
            best = perm_costs.argmin(dim=-1)
            return self.perms[best]

        assignment = []
        for c in cost.cpu().numpy():
            slots, components = linear_sum_assignment(c)
            assignment.append(slots[np.argsort(components)])
        return torch.as_tensor(np.stack(assignment), device=cost.device)

    def loss(
        self,
        summary: Tensor,
        tokens: Optional[list[Tensor]],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Per-event losses for the number of components and the slots

        Args:
            summary:
                The embedding's context vector, `(B, C)`
            tokens:
                The embedding's token groups if it has `forward_tokens`
            targets:
                Dictionary with `slots` parameters `(B, K, S)` and a
                boolean `mask` of real components `(B, K)`, as returned
                by `amplfi.train.data.datasets.SlotFlowDataset`

        Returns:
            Dictionary of per-event `count_loss` and `slot_loss`,
            and `count_correct`, whether the most probable number
            of components is right
        """
        slots, mask = targets["slots"], targets["mask"]
        num_components = mask.sum(-1)

        logits = self.count_head(summary)
        count_loss = torch.nn.functional.cross_entropy(
            logits, num_components - 1, reduction="none"
        )
        count_correct = (logits.argmax(-1) + 1 == num_components).float()

        # log likelihood of every component under every slot
        memory = self.memory(summary, tokens)
        slot_context = self.decode_slots(memory, num_components)
        B, K, S = slots.shape
        pair_context = slot_context[:, :, None].expand(-1, -1, K, -1)
        pair_slots = slots[:, None].expand(-1, K, -1, -1)
        pair_log_probs = self.slot_flow(
            pair_context.reshape(-1, self.slot_dim)
        ).log_prob(pair_slots.reshape(-1, S))
        pair_log_probs = pair_log_probs.view(B, K, K)

        assignment = self.match(pair_log_probs, mask)
        matched = pair_log_probs.gather(1, assignment[:, None]).squeeze(1)
        slot_loss = -(matched * mask).sum(-1)

        return {
            "count_loss": count_loss,
            "slot_loss": slot_loss,
            "count_correct": count_correct,
        }

    # ================================================ #
    # Inference
    # ================================================ #

    @torch.no_grad()
    def sample(
        self, n: int, summary: Tensor, tokens: Optional[list[Tensor]]
    ) -> dict[str, Tensor]:
        """
        Draw `n` posterior samples for each of `B` events

        Returns:
            Dictionary with `n_components` `(n, B)`, its probabilities
            `count_probs` `(B, K)`, and `slots` `(n, B, K, S)` with a
            boolean `mask` `(n, B, K)` of the sampled components.
            Padded slots are zero
        """
        memory = self.memory(summary, tokens)
        B, K = len(summary), self.max_components

        probs = self.count_head(summary).softmax(-1)
        num_components = torch.multinomial(probs, n, replacement=True)
        num_components = num_components.T + 1

        # sample slots for every possible number of components,
        # then keep the ones matching each draw's number of components
        slots = torch.zeros(
            n, B, K, self.num_slot_params, device=summary.device
        )
        for k in range(1, K + 1):
            counts = torch.full((B,), k, device=summary.device)
            slot_context = self.decode_slots(memory, counts)
            samples = self.slot_flow(slot_context).sample((n,))
            selected = (num_components == k)[..., None, None]
            slots = torch.where(selected, samples, slots)

        arange = torch.arange(K, device=summary.device)
        mask = arange < num_components[..., None]
        slots = slots * mask[..., None]

        return {
            "n_components": num_components,
            "count_probs": probs,
            "slots": slots,
            "mask": mask,
        }


class SlotArchitectureMixin:
    """
    Mixin for posterior architectures (normalizing flows, diffusion
    and flow matching) that adds an optional `SlotHead`. Hosts must
    define `embedding_net` and `embedding_context`, call
    `init_slot_head`, and implement `global_loss` and `sample_global`,
    which act on an already embedded context
    """

    def init_slot_head(self, slot_head: Optional[SlotHead]):
        self.slot_head = slot_head
        if slot_head is not None:
            slot_head.build(self.embedding_net)

    def global_loss(self, x: Tensor, embedded: Tensor) -> Tensor:
        """Per-event training loss for the global parameters `x`"""
        raise NotImplementedError

    def sample_global(self, n: int, embedded: Tensor) -> Tensor:
        """`n` samples `(n, B, P)` of the global parameters"""
        raise NotImplementedError

    def embed(self, context) -> tuple[Tensor, Optional[list[Tensor]]]:
        """
        Embed the data once for both the global posterior and the
        slot head, returning the context vector `(B, C)` and, if the
        slot head uses them, the embedding's token groups
        """
        with self.embedding_context():
            if self.slot_head is not None and self.slot_head.use_tokens:
                return self.embedding_net.forward_tokens(context)
            return self.embedding_net(context), None

    def slot_losses(
        self, context, targets: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Per-event training losses with the slot head

        Args:
            context:
                Context passed to the embedding network
            targets:
                Dictionary with `global` parameters `(B, P)`,
                `slots` parameters `(B, K, S)` and a boolean
                `mask` of real components `(B, K)`, as returned
                by `amplfi.train.data.datasets.SlotFlowDataset`

        Returns:
            Dictionary of per-event `loss` (the sum of the
            losses), `global_loss`, `count_loss` and `slot_loss`,
            and `count_correct`
        """
        summary, tokens = self.embed(context)
        losses = self.slot_head.loss(summary, tokens, targets)
        losses["global_loss"] = self.global_loss(targets["global"], summary)
        losses["loss"] = (
            losses["global_loss"] + losses["count_loss"] + losses["slot_loss"]
        )
        return losses

    @torch.no_grad()
    def sample_with_slots(
        self, n: int, context
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Draw `n` samples of the global parameters, `(n, B, P)`,
        and of the slots (see `SlotHead.sample`) for each event
        """
        summary, tokens = self.embed(context)
        samples = self.sample_global(n, summary)
        return samples, self.slot_head.sample(n, summary, tokens)
