from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

from .components import ConditionalMLP
from .embeddings.base import Embedding, load_embedding_weights


class FlowMatchingArchitecture(torch.nn.Module):
    """
    Base class for flow matching architectures that provides the
    interface for interacting with embedding networks, mirroring
    `amplfi.train.architectures.flows.FlowArchitecture` and
    `amplfi.train.architectures.diffusion.DiffusionArchitecture`.

    Trains a velocity field `v(x_t, t, context)` to match the
    conditional flow matching objective of Lipman et al. 2023, using
    linear (optimal-transport) probability paths between a standard
    normal `x_0` and the target parameters `x_1`:

        x_t = (1 - t) * x_0 + t * x_1,   u_t = x_1 - x_0

    Posterior samples are drawn by numerically integrating the
    learned ODE `dx/dt = v(x, t, context)` from `t=0` (noise) to
    `t=1` (posterior sample) with the forward Euler method.

    Args:
        num_params:
            Dimensionality of the parameter vector being inferred.
        embedding_net:
            Network used to embed the strain context.
        sampling_steps:
            Number of Euler integration steps used to solve the
            learned ODE at sampling time.
        embedding_weights:
            Path to a checkpoint from which to load pre-trained
            embedding weights.
        freeze_embedding:
            Whether to freeze the embedding network's weights.
    """

    def __init__(
        self,
        num_params: int,
        embedding_net: Embedding,
        sampling_steps: int = 100,
        embedding_weights: Optional[Path] = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        self.num_params = num_params
        self.embedding_net = embedding_net
        self.sampling_steps = sampling_steps

        if freeze_embedding:
            self.embedding_context = torch.no_grad
        else:
            self.embedding_context = nullcontext

        if embedding_weights is not None:
            load_embedding_weights(self.embedding_net, embedding_weights)

    def build_velocity_field(self) -> torch.nn.Module:
        raise NotImplementedError

    def loss(self, x1: torch.Tensor, context) -> torch.Tensor:
        """
        Conditional flow matching loss: regress the velocity field
        along the linear path between random noise `x0` and the
        target parameters `x1`.
        """
        if not hasattr(self, "velocity_field"):
            raise RuntimeError("Velocity field is not built")

        with self.embedding_context():
            embedded_context = self.embedding_net(context)

        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t = x0 + t.unsqueeze(-1) * (x1 - x0)
        target = x1 - x0

        pred = self.velocity_field(x_t, t, embedded_context)
        return torch.nn.functional.mse_loss(
            pred, target, reduction="none"
        ).mean(dim=-1)

    @torch.no_grad()
    def sample(self, n: int, context) -> torch.Tensor:
        """
        Draw `n` samples per event in `context` by integrating the
        learned ODE from `t=0` to `t=1` with `self.sampling_steps`
        forward Euler steps.
        """
        if not hasattr(self, "velocity_field"):
            raise RuntimeError("Velocity field is not built")

        embedded_context = self.embedding_net(context)
        batch = embedded_context.shape[0]
        device = embedded_context.device

        x = torch.randn(n * batch, self.num_params, device=device)
        context_flat = (
            embedded_context.unsqueeze(0)
            .expand(n, -1, -1)
            .reshape(n * batch, -1)
        )

        dt = 1.0 / self.sampling_steps
        for i in range(self.sampling_steps):
            t = torch.full((n * batch,), i * dt, device=device)
            x = x + self.velocity_field(x, t, context_flat) * dt

        return x.reshape(n, batch, self.num_params)


class RectifiedFlow(FlowMatchingArchitecture):
    """
    Flow matching architecture using a residual-MLP velocity field,
    for compatibility with the `FlowMatchingArchitecture` interface.

    Args:
        hidden_features:
            Width of the velocity field's residual blocks.
        num_blocks:
            Number of residual blocks in the velocity field.
        time_embed_dim:
            Dimensionality of the sinusoidal time embedding.
    """

    def __init__(
        self,
        *args,
        hidden_features: int = 256,
        num_blocks: int = 4,
        time_embed_dim: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.time_embed_dim = time_embed_dim
        self.velocity_field = self.build_velocity_field()

    def build_velocity_field(self) -> torch.nn.Module:
        return ConditionalMLP(
            self.num_params,
            self.embedding_net.context_dim,
            hidden_features=self.hidden_features,
            num_blocks=self.num_blocks,
            time_embed_dim=self.time_embed_dim,
            # continuous t lives in [0, 1]; scale up so the
            # sinusoidal embedding spans a comparable numeric
            # range to diffusion's integer timesteps
            time_scale=1000.0,
        )
