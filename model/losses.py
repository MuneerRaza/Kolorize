"""
Loss functions for colorization diffusion training.

Primary: L1 loss on noise prediction (standard DDPM objective)
Auxiliary: VGG perceptual loss on reconstructed x0 (prevents desaturation)

The perceptual loss is key for vivid colors — L1 alone tends to produce
muted/brownish outputs because the "average" of all possible colors is gray.
VGG features push toward realistic textures and color distributions.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss.

    Extracts features from relu1_2, relu2_2, relu3_3 of pretrained VGG16
    and computes L1 distance. These layers capture increasingly abstract
    features: edges → textures → semantic patterns.

    Note: VGG expects 3-channel RGB input. Since we predict AB (2 channels),
    we combine with L channel to create a pseudo-RGB for VGG.
    """

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Extract at these layers (relu after each conv block)
        self.slice1 = nn.Sequential(*features[:4])   # relu1_2
        self.slice2 = nn.Sequential(*features[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*features[9:16])  # relu3_3

        # Freeze — we never train VGG
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet range."""
        x = (x + 1) / 2  # [-1, 1] → [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predicted and target.

        Args:
            pred: Predicted image (B, 3, H, W), range [-1, 1]
            target: Target image (B, 3, H, W), range [-1, 1]

        Returns:
            Scalar perceptual loss.
        """
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)

        # Extract and compare features at each level
        p, t = pred, target
        for layer in [self.slice1, self.slice2, self.slice3]:
            p = layer(p)
            t = layer(t)
            loss = loss + nn.functional.l1_loss(p, t)

        return loss


class ColorizationLoss(nn.Module):
    """Combined loss for colorization diffusion training.

    L_total = L1(noise_pred, noise_target) + λ × VGG_perceptual(x0_pred, x0_target)

    The perceptual loss is computed on the reconstructed x0 (not the noise),
    converted to pseudo-RGB by concatenating with the L channel.

    Args:
        perceptual_weight: Weight for VGG perceptual loss (λ).
        use_perceptual: Whether to use perceptual loss at all.
    """

    def __init__(self, perceptual_weight: float = 0.1, use_perceptual: bool = True):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual

        if use_perceptual:
            self.vgg = VGGPerceptualLoss()

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        x0_pred: torch.Tensor | None = None,
        x0_target: torch.Tensor | None = None,
        L: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            noise_pred: Predicted noise (B, 2, H, W)
            noise_target: Actual noise (B, 2, H, W)
            x0_pred: Reconstructed AB from predicted noise (B, 2, H, W)
            x0_target: Ground truth AB (B, 2, H, W)
            L: L channel (B, 1, H, W) — needed to create pseudo-RGB for VGG

        Returns:
            Dict with 'total', 'noise_loss', 'perceptual_loss'
        """
        # Primary: L1 on noise prediction
        noise_loss = self.l1(noise_pred, noise_target)

        result = {"noise_loss": noise_loss}

        # Auxiliary: perceptual loss on reconstructed x0
        if (
            self.use_perceptual
            and x0_pred is not None
            and x0_target is not None
            and L is not None
        ):
            # Create pseudo-RGB: concat L + AB → 3 channels
            # This isn't true RGB, but VGG features still capture
            # texture/structure similarity effectively
            pred_rgb = torch.cat([L, x0_pred], dim=1)    # (B, 3, H, W)
            target_rgb = torch.cat([L, x0_target], dim=1)  # (B, 3, H, W)

            perceptual_loss = self.vgg(pred_rgb, target_rgb)
            result["perceptual_loss"] = perceptual_loss
            result["total"] = noise_loss + self.perceptual_weight * perceptual_loss
        else:
            result["perceptual_loss"] = torch.tensor(0.0, device=noise_loss.device)
            result["total"] = noise_loss

        return result
