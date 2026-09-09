import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

from .components import ConditionalMLP
from .embeddings.base import Embedding, load_embedding_weights


def cosine_beta_schedule(
    num_timesteps: int, s: float = 0.008
) -> torch.Tensor:
    """Cosine noise schedule from Nichol & Dhariwal 2021"""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionArchitecture(torch.nn.Module):
    """
    Base class for diffusion architectures that provides the
    interface for interacting with embedding networks, mirroring
    `amplfi.train.architectures.flows.FlowArchitecture`.

    Args:
        num_params:
            Dimensionality of the parameter vector being inferred.
        embedding_net:
            Network used to embed the strain context.
        num_timesteps:
            Number of steps in the forward noising / training schedule.
        sampling_timesteps:
            Number of steps to use for sampling. Defaults to
            `num_timesteps`; a smaller value samples faster via DDIM
            (Song et al. 2020) at the cost of some sample quality.
        eta:
            DDIM stochasticity parameter. `eta=1` recovers ancestral
            DDPM sampling (Ho et al. 2020) when `sampling_timesteps
            == num_timesteps`; `eta=0` gives deterministic sampling.
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
        num_timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        eta: float = 1.0,
        embedding_weights: Optional[Path] = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        self.num_params = num_params
        self.embedding_net = embedding_net
        self.num_timesteps = num_timesteps
        self.sampling_timesteps = min(
            sampling_timesteps or num_timesteps, num_timesteps
        )
        self.eta = eta

        if freeze_embedding:
            self.embedding_context = torch.no_grad
        else:
            self.embedding_context = nullcontext

        if embedding_weights is not None:
            load_embedding_weights(self.embedding_net, embedding_weights)

        betas = cosine_beta_schedule(num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def build_denoiser(self) -> torch.nn.Module:
        raise NotImplementedError

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: sample `x_t` from `q(x_t | x_0)`"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        )
        return (
            sqrt_alphas_cumprod_t * x0
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def loss(self, x0: torch.Tensor, context) -> torch.Tensor:
        """
        Denoising score-matching loss: predict the noise used to
        corrupt `x0` at a randomly sampled timestep.
        """
        if not hasattr(self, "denoiser"):
            raise RuntimeError("Denoiser is not built")

        with self.embedding_context():
            embedded_context = self.embedding_net(context)

        noise = torch.randn_like(x0)
        t = torch.randint(
            0, self.num_timesteps, (x0.shape[0],), device=x0.device
        )
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(x_t, t, embedded_context)
        return torch.nn.functional.mse_loss(
            pred_noise, noise, reduction="none"
        ).mean(dim=-1)

    @torch.no_grad()
    def sample(self, n: int, context) -> torch.Tensor:
        """
        Draw `n` samples per event in `context` via DDIM sampling,
        using `self.sampling_timesteps` reverse diffusion steps.
        """
        if not hasattr(self, "denoiser"):
            raise RuntimeError("Denoiser is not built")

        embedded_context = self.embedding_net(context)
        batch = embedded_context.shape[0]
        device = embedded_context.device

        step = self.num_timesteps // self.sampling_timesteps
        timesteps = list(range(0, self.num_timesteps, step))[::-1]

        x = torch.randn(n * batch, self.num_params, device=device)
        context_flat = (
            embedded_context.unsqueeze(0)
            .expand(n, -1, -1)
            .reshape(n * batch, -1)
        )

        for i, t in enumerate(timesteps):
            t_batch = torch.full(
                (n * batch,), t, device=device, dtype=torch.long
            )
            pred_noise = self.denoiser(x, t_batch, context_flat)

            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.ones_like(alpha_bar_t)
            )

            x0_pred = (
                x - torch.sqrt(1 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t)

            sigma_t = self.eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )
            direction = (
                torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise
            )
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + direction
                + sigma_t * noise
            )

        return x.reshape(n, batch, self.num_params)


class DDPM(DiffusionArchitecture):
    """
    Denoising diffusion probabilistic model (Ho et al. 2020) with a
    residual-MLP denoising network, for compatibility with the
    `DiffusionArchitecture` interface.

    Args:
        hidden_features:
            Width of the denoiser's residual blocks.
        num_blocks:
            Number of residual blocks in the denoiser.
        time_embed_dim:
            Dimensionality of the sinusoidal timestep embedding.
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
        self.denoiser = self.build_denoiser()

    def build_denoiser(self) -> torch.nn.Module:
        return ConditionalMLP(
            self.num_params,
            self.embedding_net.context_dim,
            hidden_features=self.hidden_features,
            num_blocks=self.num_blocks,
            time_embed_dim=self.time_embed_dim,
            time_scale=1.0,
        )
