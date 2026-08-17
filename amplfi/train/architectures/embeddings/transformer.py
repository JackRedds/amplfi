from typing import Literal, Optional

from ml4gw.nn.norm import NormLayer

import torch
import torch.nn as nn
import math

from .base import Embedding


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding whose length is determined
    dynamically from the input.

    Parameters
    ----------
    dim:
        Embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                "Sinusoidal positional encoding requires an even "
                "embedding dimension."
            )

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape (batch, sequence_length, dim).

        Returns
        -------
        Tensor:
            Input tensor with sinusoidal positional encoding added.
        """

        batch_size, seq_len, dim = x.shape

        if dim != self.dim:
            raise ValueError(
                f"Expected embedding dimension {self.dim}, "
                f"but got {dim}."
            )

        # Position indices
        position = torch.arange(
            seq_len,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(1)

        # Frequencies for each pair of dimensions
        div_term = torch.exp(
            torch.arange(
                0,
                dim,
                2,
                device=x.device,
                dtype=x.dtype,
            )
            * (-math.log(10000.0) / dim)
        )

        # Shape: (seq_len, dim)
        pe = torch.zeros(
            seq_len,
            dim,
            device=x.device,
            dtype=x.dtype,
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        return x + pe


class Transformer(nn.Module):
    """
    Time-domain CNN + Transformer embedding for gravitational-wave data.

    Parameters
    ----------
    num_ifos:
        Number of interferometers.

    context_dim:
        Dimension of the final embedding returned to the flow.

    layers:
        Number of channels in each convolutional stage.

    kernel_size:
        Kernel size of the convolutional layers.

    zero_init_residual:
        If True, initialize the final normalization scale of
        residual blocks to zero.

    groups:
        Number of groups used in grouped convolutions.

    width_per_group:
        Base channel width used to determine the CNN width.

    stride_type:
        List specifying whether each CNN stage performs a
        stride-based downsampling or uses dilation.

    norm_layer:
        Normalization layer used by the CNN.

    transformer_layers:
        Number of Transformer encoder layers.

    transformer_heads:
        Number of attention heads.

    transformer_dim_feedforward:
        Dimension of the Transformer feed-forward network.

    dropout:
        Transformer dropout probability.

    patch_size:
        Number of CNN feature samples represented by each
        Transformer token.
    """

    def __init__(
        self,
        in_channels: int,
        layers: list[int],
        transformer_dim: int,
        kernel_size: int = 3,
        groups: int = 1,
        stride_type: Optional[
            list[Literal["stride", "dilation"]]
        ] = None,
        norm_layer: Optional[NormLayer] = None,

        # Transformer-specific arguments
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_size: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.transformer_dim = transformer_dim

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        if stride_type is None:
            stride_type = ["stride"] * len(layers)

        if len(stride_type) != len(layers):
            raise ValueError(
                "stride_type must have the same length as layers"
            )

        # ---------------------------------------------------------
        # CNN feature extractor
        # ---------------------------------------------------------

        self.cnn = self._make_cnn(
            in_channels=in_channels,
            layers=layers,
            kernel_size=kernel_size,
            groups=groups,
            norm_layer=norm_layer,
            stride_type=stride_type,
        )

        cnn_dim = layers[-1]

        # ---------------------------------------------------------
        # Convert CNN features into Transformer tokens
        # ---------------------------------------------------------

        self.patch_embedding = nn.Conv1d(
            in_channels=cnn_dim,
            out_channels=transformer_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ---------------------------------------------------------
        # CLS token
        # ---------------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, transformer_dim) * 0.02
        )

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.final_norm = nn.LayerNorm(transformer_dim)

        # ---------------------------------------------------------
        # Positional embeddings
        #
        # This is initialized lazily because the input waveform
        # length isn't necessarily known from the constructor.
        # ---------------------------------------------------------

        self.position_embedding = SinusoidalPositionalEncoding(
            transformer_dim
        )

    # =============================================================
    # CNN
    # =============================================================

    def _make_cnn(
        self,
        in_channels,
        layers,
        kernel_size,
        groups,
        norm_layer,
        stride_type,
    ):
        modules = []

        for i, out_channels in enumerate(layers):

            if stride_type[i] == "stride":
                stride = 2
                dilation = 1

            elif stride_type[i] == "dilation":
                stride = 1
                dilation = 2 ** i

            else:
                raise ValueError(
                    f"Unknown stride_type: {stride_type[i]}"
                )

            padding = (
                (kernel_size - 1) * dilation
            ) // 2

            modules.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                )
            )

            modules.append(
                norm_layer(out_channels)
            )

            modules.append(
                nn.GELU()
            )

            in_channels = out_channels

        return nn.Sequential(*modules)

    # =============================================================
    # Forward
    # =============================================================

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x:
            Tensor with shape

                [batch, in_channels, n_samples]

        Returns
        -------
        context:
            Tensor with shape

                [batch, classes]
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, in_channels, samples], "
                f"got {tuple(x.shape)}"
            )

        batch_size, num_ifos, _ = x.shape

        if num_ifos != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"got {num_ifos}"
            )

        # ---------------------------------------------------------
        # CNN
        #
        # [B, IFO, T]
        #       ↓
        # [B, C, T']
        # ---------------------------------------------------------

        x = self.cnn(x)

        # ---------------------------------------------------------
        # Patch embedding
        #
        # [B, C, T']
        #       ↓
        # [B, transformer_dim, N]
        # ---------------------------------------------------------

        x = self.patch_embedding(x)

        # ---------------------------------------------------------
        # Transformer expects
        #
        # [B, N, D]
        # ---------------------------------------------------------

        x = x.transpose(1, 2)

        # ---------------------------------------------------------
        # CLS token
        # ---------------------------------------------------------

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [cls, x],
            dim=1,
        )

        # ---------------------------------------------------------
        # Positional embedding
        # ---------------------------------------------------------

        x = self.position_embedding(x)

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        x = self.transformer(x)

        x = self.final_norm(x)

        # ---------------------------------------------------------
        # Global representation
        #
        # CLS token
        # ---------------------------------------------------------

        return x[:, 0]


class TimeDomainTransformer(Embedding):
    def __init__(
        self,
        num_ifos: int,

        # Time branch
        context_dim: int,
        layers: list[int],
        kernel_size: int = 3,
        groups: int = 1,
        stride_type: Optional[
            list[Literal["stride", "dilation"]]
        ] = None,
        norm_layer: Optional[NormLayer] = None,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_size: int = 4,
        # context_dim: int = 256,
    ):
        super().__init__()

        self.context_dim = context_dim

        self.embedding = Transformer(
            in_channels=num_ifos,
            layers=layers,
            transformer_dim=context_dim,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
            groups=groups,
            stride_type=stride_type,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
        )

    def forward(self, X):
        strain, _ = X

        context = self.embedding(strain)

        return context


class MultiModalTransformer(Embedding):
    def __init__(
        self,
        num_ifos: int,

        # Time branch
        time_context_dim: int,
        time_layers: list[int],
        freq_context_dim: int,
        freq_layers: list[int],
        time_kernel_size: int = 3,
        time_groups: int = 1,
        time_stride_type: Optional[
            list[Literal["stride", "dilation"]]
        ] = None,
        freq_kernel_size: int = 3,
        freq_groups: int = 1,
        freq_stride_type: Optional[
            list[Literal["stride", "dilation"]]
        ] = None,
        norm_layer: Optional[NormLayer] = None,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_size: int = 4,
        # context_dim: int = 256,
    ):
        super().__init__()

        self.context_dim = time_context_dim + freq_context_dim

        self.time_embedding = Transformer(
            in_channels=num_ifos,
            transformer_dim=time_context_dim,
            layers=time_layers,
            kernel_size=time_kernel_size,
            norm_layer=norm_layer,
            groups=time_groups,
            stride_type=time_stride_type,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
        )

        self.freq_embedding = Transformer(
            in_channels=int(num_ifos * 2),
            transformer_dim=freq_context_dim,
            layers=freq_layers,
            kernel_size=freq_kernel_size,
            norm_layer=norm_layer,
            groups=freq_groups,
            stride_type=freq_stride_type,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
        )

        # Combine the two embeddings
        self.fusion = nn.Sequential(
            nn.Linear(
                time_context_dim + freq_context_dim,
                self.context_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.context_dim,
                self.context_dim,
            ),
        )

    def forward(self, X):
        strain, _ = X

        time_context = self.time_embedding(strain)
        strain_fft = torch.fft.rfft(strain)
        strain_fft = torch.cat((strain_fft.real, strain_fft.imag), dim=1)

        freq_context = self.freq_embedding(strain_fft)

        context = torch.cat(
            [
                time_context,
                freq_context,
            ],
            dim=-1,
        )

        return self.fusion(context)


class MultiModalPsdTransformer(Embedding):
    def __init__(
        self,
        num_ifos: int,

        # Time branch
        time_context_dim: int,
        time_layers: list[int],
        freq_context_dim: int,
        freq_layers: list[int],
        time_kernel_size: int = 3,
        time_groups: int = 1,
        time_stride_type: Optional[
            list[Literal["stride", "dilation"]]
        ] = None,
        freq_kernel_size: int = 3,
        freq_groups: int = 1,
        freq_stride_type: Optional[
            list[Literal["stride", "dilation"]]
        ] = None,
        norm_layer: Optional[NormLayer] = None,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_size: int = 4,
        # context_dim: int = 256,
    ):
        super().__init__()

        self.context_dim = time_context_dim + freq_context_dim

        self.time_embedding = Transformer(
            in_channels=num_ifos,
            layers=time_layers,
            transformer_dim=time_context_dim,
            kernel_size=time_kernel_size,
            norm_layer=norm_layer,
            groups=time_groups,
            stride_type=time_stride_type,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
        )

        self.freq_embedding = Transformer(
            in_channels=int(num_ifos * 3),
            layers=freq_layers,
            transformer_dim=freq_context_dim,
            kernel_size=freq_kernel_size,
            norm_layer=norm_layer,
            groups=freq_groups,
            stride_type=freq_stride_type,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
        )

        # Combine the two embeddings
        self.fusion = nn.Sequential(
            nn.Linear(
                time_context_dim + freq_context_dim,
                self.context_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.context_dim,
                self.context_dim,
            ),
        )

    def forward(self, X):
        strain, asds = X

        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        time_context = self.time_embedding(strain)
        X_fft = torch.fft.rfft(strain)
        X_fft = X_fft[..., -asds.shape[-1] :]
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)

        freq_context = self.freq_embedding(X_fft)

        context = torch.cat(
            [
                time_context,
                freq_context,
            ],
            dim=-1,
        )

        return self.fusion(context)