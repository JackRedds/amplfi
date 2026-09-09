import math

import torch


class SinusoidalTimeEmbedding(torch.nn.Module):
    """Sinusoidal embedding of a scalar time value, as in Ho et al. 2020"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half_dim, device=t.device)
            / (half_dim - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.linear1 = torch.nn.Linear(dim, dim)
        self.cond_proj = torch.nn.Linear(cond_dim, dim)
        self.act = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        h = self.linear1(h)
        h = h + self.cond_proj(cond)
        h = self.act(h)
        h = self.linear2(h)
        return x + h


class ConditionalMLP(torch.nn.Module):
    """
    Residual MLP that maps a parameter vector `x`, conditioned on a
    scalar time value and an embedded strain context, to a vector of
    the same dimensionality as `x`. Used to predict either the noise
    added to `x` (diffusion models) or the velocity field along a
    probability path from noise to `x` (flow matching models).

    Args:
        num_params:
            Dimensionality of `x`.
        context_dim:
            Dimensionality of the embedded strain context.
        hidden_features:
            Width of the residual blocks.
        num_blocks:
            Number of residual blocks.
        time_embed_dim:
            Dimensionality of the sinusoidal time embedding.
        time_scale:
            Factor `t` is multiplied by before embedding, so that
            both integer diffusion timesteps and continuous
            flow-matching times in `[0, 1]` produce well-spread
            sinusoidal embeddings.
    """

    def __init__(
        self,
        num_params: int,
        context_dim: int,
        hidden_features: int = 256,
        num_blocks: int = 4,
        time_embed_dim: int = 128,
        time_scale: float = 1.0,
    ):
        super().__init__()
        self.time_scale = time_scale
        self.time_embed = torch.nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
        )
        cond_dim = time_embed_dim + context_dim
        self.input_proj = torch.nn.Linear(num_params, hidden_features)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualBlock(hidden_features, cond_dim)
                for _ in range(num_blocks)
            ]
        )
        self.output_proj = torch.nn.Linear(hidden_features, num_params)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_embed(t * self.time_scale)
        cond = torch.cat([t_emb, context], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)
