"""
Attention modules for Colorize UNet — "Optimized RADAR-Net" architecture.

Design rationale:
- Lightweight Channel Attention (from Tang et al., ACM MM 2023):
  GAP + 1D Conv + Sigmoid. Ultra cheap, selects important feature channels.
  No multi-branch needed — the UNet's multi-level structure already provides
  multi-scale features.

- Gated-Dconv FFN (adapted from RADAR-Net, Raza et al., IACMC 2025):
  Gating mechanism with 7×7 depthwise conv for wide receptive field.
  7×7 kernel regains the "wide context" that dropping MSCA's multi-scale
  branches would lose, at negligible cost (depthwise convs are cheap).

- Self-Attention (bottleneck only):
  Full spatial self-attention at 16×16 resolution for global reasoning.
  "Dog head is brown on the left → tail on the right should be brown too."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightChannelAttention(nn.Module):
    """Lightweight channel attention from Tang et al. (ACM MM 2023).

    Minimal channel attention: Global Average Pool → 1D Conv → Sigmoid.
    Models inter-channel relationships with almost zero overhead.

    Architecture:
        Input F (C × H × W)
          │
          Global Average Pool → (C × 1 × 1) → squeeze to (1 × C)
          │
          1D Conv (kernel=3, learns local channel relationships)
          │
          Sigmoid → (1 × C) → unsqueeze to (C × 1 × 1)
          │
          F̃ = F + F × attention_weights   (residual)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 1D conv across channels — kernel_size=3 captures local channel relationships
        # padding=1 keeps channel dim unchanged
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # Squeeze spatial dims: (B, C, H, W) → (B, C, 1, 1) → (B, 1, C)
        attn = self.gap(x).view(b, 1, c)

        # Channel-wise 1D conv: (B, 1, C) → (B, 1, C)
        attn = self.conv1d(attn)

        # Gate: (B, 1, C) → (B, C, 1, 1)
        attn = self.sigmoid(attn).view(b, c, 1, 1)

        # Residual gating
        return x + x * attn


class GatedDconvFFN(nn.Module):
    """Gated Depthwise Conv Feed-Forward Network (from RADAR-Net).

    Uses 7×7 depthwise conv for wide receptive field — compensates for
    dropping the multi-scale branches of MSCA. At 64×64, a 7×7 kernel
    covers ~11% of the spatial extent, enough to understand local context
    like "this texture is fur, not grass."

    Architecture:
        Input (C) → 1×1 Conv (C → 4C) → Split into 2C + 2C
          Path A: 7×7 DWConv → GELU (content with wide context)
          Path B: Identity (gate)
          → A ⊙ B (gating — only informative features pass)
          → 1×1 Conv (2C → C)
        Output (C)
    """

    def __init__(self, channels: int, expansion: int = 4, kernel_size: int = 7):
        super().__init__()
        hidden = channels * expansion

        self.project_in = nn.Conv2d(channels, hidden, 1)
        self.dwconv = nn.Conv2d(
            hidden // 2,
            hidden // 2,
            kernel_size,
            padding=kernel_size // 2,
            groups=hidden // 2,
        )
        self.project_out = nn.Conv2d(hidden // 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)

        # Split into gate and content
        gate, content = x.chunk(2, dim=1)

        # Content: wide depthwise conv + activation
        content = F.gelu(self.dwconv(content))

        # Gating: selective feature filtering
        x = content * gate

        return self.project_out(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for the bottleneck.

    Only used at 16×16 resolution where it's computationally cheap.
    Captures global relationships: if one part of the image is brown,
    related parts should be brown too.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(1, channels)  # LayerNorm for conv
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(1)  # each: (B, heads, head_dim, H*W)

        # Scaled dot-product attention
        attn = torch.einsum("bhdn,bhen->bhde", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhde,bhen->bhdn", attn, v)

        out = out.reshape(b, c, h, w)
        return self.proj(out) + residual


class AttentionBlock(nn.Module):
    """Combined attention block: Channel Attention + Gated-Dconv FFN.

    Used at levels 2-3 (low resolution) of the UNet.

    Architecture:
        Input
          ├── LightweightChannelAttention (select important features)
          ├── LayerNorm → GatedDconvFFN (filter + wide context) + residual
        Output
    """

    def __init__(self, channels: int, ffn_kernel_size: int = 7):
        super().__init__()
        self.channel_attn = LightweightChannelAttention(channels)
        self.norm = nn.GroupNorm(1, channels)
        self.ffn = GatedDconvFFN(channels, kernel_size=ffn_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention (has its own residual)
        x = self.channel_attn(x)

        # Gated FFN with residual
        x = x + self.ffn(self.norm(x))

        return x
