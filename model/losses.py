"""
Loss functions for colorization diffusion training.

Primary: Weighted L1 loss on prediction (v or ε) with Min-SNR-γ weighting
Auxiliary: VGG perceptual loss on reconstructed x0 (prevents desaturation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss.

    Extracts features from relu1_2, relu2_2, relu3_3 of pretrained VGG16
    and computes L1 distance.
    """

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        self.slice1 = nn.Sequential(*features[:4])   # relu1_2
        self.slice2 = nn.Sequential(*features[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*features[9:16])  # relu3_3

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        p, t = pred, target
        for layer in [self.slice1, self.slice2, self.slice3]:
            p = layer(p)
            t = layer(t)
            loss = loss + F.l1_loss(p, t)

        return loss


class ColorizationLoss(nn.Module):
    """Combined loss with Min-SNR-γ weighting.

    L_total = SNR_weight × L1(pred, target) + λ × VGG_perceptual(x0_pred, x0_target)

    Args:
        perceptual_weight: Weight for VGG perceptual loss (λ).
        use_perceptual: Whether to use perceptual loss at all.
    """

    def __init__(self, perceptual_weight: float = 0.1, use_perceptual: bool = True):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual

        if use_perceptual:
            self.vgg = VGGPerceptualLoss()

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        snr_weights: torch.Tensor | None = None,
        x0_pred: torch.Tensor | None = None,
        x0_target: torch.Tensor | None = None,
        L: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            model_output: Model prediction — v or ε (B, 2, H, W)
            target: Target — v or ε (B, 2, H, W)
            snr_weights: Min-SNR-γ weights per sample (B,)
            x0_pred: Reconstructed AB (B, 2, H, W)
            x0_target: Ground truth AB (B, 2, H, W)
            L: L channel (B, 1, H, W)
        """
        # Per-sample L1 loss
        prediction_loss = F.l1_loss(model_output, target, reduction="none")
        prediction_loss = prediction_loss.mean(dim=[1, 2, 3])  # (B,)

        # Apply Min-SNR weights
        if snr_weights is not None:
            prediction_loss = prediction_loss * snr_weights

        prediction_loss = prediction_loss.mean()

        result = {"prediction_loss": prediction_loss}

        # Perceptual loss on reconstructed x0
        if (
            self.use_perceptual
            and x0_pred is not None
            and x0_target is not None
            and L is not None
        ):
            pred_rgb = torch.cat([L, x0_pred], dim=1)
            target_rgb = torch.cat([L, x0_target], dim=1)

            perceptual_loss = self.vgg(pred_rgb, target_rgb)
            result["perceptual_loss"] = perceptual_loss
            result["total"] = prediction_loss + self.perceptual_weight * perceptual_loss
        else:
            result["perceptual_loss"] = torch.tensor(0.0, device=prediction_loss.device)
            result["total"] = prediction_loss

        return result
