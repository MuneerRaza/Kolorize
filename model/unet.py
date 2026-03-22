"""
Conditional UNet for diffusion-based colorization — "Optimized RADAR-Net".

Hybrid architecture:
- Levels 0-1 (256×256, 128×128): ResBlocks only (conv is sufficient)
- Levels 2-3 (64×64, 32×32): ResBlock + Lightweight Channel Attention + Gated-Dconv FFN (k=7)
- Bottleneck (16×16): ResBlock + Self-Attention + Gated-Dconv FFN (k=7)

Input: concat(L_channel, noisy_AB) = 3 channels + timestep embedding
Output: predicted noise ε (2 channels)

Design choices:
- Channel attention from Tang et al. (ACM MM 2023) — ultra lightweight
- Gated FFN from RADAR-Net (Raza et al., IACMC 2025) with 7×7 depthwise conv
- Self-attention only at bottleneck (16×16) for global color consistency
- GroupNorm(32) throughout (stable with varying noise levels)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import AttentionBlock, GatedDconvFFN, SelfAttention


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding.

    Converts integer timestep t into a continuous vector so the model
    knows "how noisy is this input?"

    t (int) → sin/cos at different frequencies → vector of dim `dim`
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with timestep conditioning.

    Architecture:
        Input (C_in)
          ├──────────────────────────┐ (skip: 1×1 conv if C_in != C_out)
          → GroupNorm → SiLU → Conv3×3
          → + timestep embedding
          → GroupNorm → SiLU → Dropout → Conv3×3
          + ◄────────────────────────┘
        Output (C_out)
    """

    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Inject timestep
        t = self.time_proj(t_emb)[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Down/Up sampling
# ---------------------------------------------------------------------------


class Downsample(nn.Module):
    """Strided convolution downsampling (2x)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv (avoids checkerboard artifacts)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """Conditional UNet for colorization diffusion.

    Args:
        in_channels: Input channels (L + noisy_AB = 3).
        out_channels: Output channels (predicted noise for AB = 2).
        base_channels: Base channel count (multiplied at each level).
        channel_mult: Channel multiplier at each level.
        num_res_blocks: Number of ResBlocks per level.
        attention_levels: Which levels get attention (0-indexed).
        time_dim: Dimension of timestep embedding.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: tuple[int, ...] = (2, 3),
        time_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = len(channel_mult)
        channels = [base_channels * m for m in channel_mult]

        # Timestep embedding: sinusoidal → MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Initial projection
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ---------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_ch = channels[0]
        # Track all channels entering skip connections for decoder
        self._encoder_channels = [channels[0]]

        for level in range(self.num_levels):
            ch = channels[level]
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(prev_ch, ch, time_dim, dropout))
                if level in attention_levels:
                    level_blocks.append(AttentionBlock(ch))
                prev_ch = ch
                self._encoder_channels.append(ch)

            self.encoder_blocks.append(level_blocks)

            if level < self.num_levels - 1:
                self.downsamples.append(Downsample(ch))
                self._encoder_channels.append(ch)
            else:
                self.downsamples.append(None)

        # ---------------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------------
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_ffn = GatedDconvFFN(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, dropout)

        # ---------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Build decoder in reverse, consuming skip channels
        encoder_channels_copy = list(self._encoder_channels)

        for level in reversed(range(self.num_levels)):
            ch = channels[level]
            level_blocks = nn.ModuleList()

            # num_res_blocks + 1 to account for skip connection at each sub-block
            for _ in range(num_res_blocks + 1):
                skip_ch = encoder_channels_copy.pop()
                level_blocks.append(
                    ResBlock(prev_ch + skip_ch, ch, time_dim, dropout)
                )
                if level in attention_levels:
                    level_blocks.append(AttentionBlock(ch))
                prev_ch = ch

            self.decoder_blocks.append(level_blocks)

            if level > 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(None)

        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — concat(L, noisy_AB)
            t: (B,) — integer timesteps

        Returns:
            (B, 2, H, W) — predicted noise for AB channels
        """
        t_emb = self.time_embed(t)

        # Initial conv
        h = self.input_conv(x)
        skips = [h]

        # ---- Encoder ----
        for level in range(self.num_levels):
            for block in self.encoder_blocks[level]:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                    skips.append(h)  # Only save skip after ResBlock
                else:
                    h = block(h)  # Attention — no skip saved

            if self.downsamples[level] is not None:
                h = self.downsamples[level](h)
                skips.append(h)

        # ---- Bottleneck ----
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = h + self.mid_ffn(h)  # FFN with residual
        h = self.mid_block2(h, t_emb)

        # ---- Decoder ----
        for level_idx in range(self.num_levels):
            for block in self.decoder_blocks[level_idx]:
                if isinstance(block, ResBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, t_emb)
                else:
                    h = block(h)

            if self.upsamples[level_idx] is not None:
                h = self.upsamples[level_idx](h)

        return self.output_conv(h)
