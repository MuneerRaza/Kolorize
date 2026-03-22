"""
Evaluation metrics for colorization quality.

Metrics:
- PSNR: Peak Signal-to-Noise Ratio (higher = better)
- SSIM: Structural Similarity Index (higher = better)
- LPIPS: Learned Perceptual Image Patch Similarity (lower = better)
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR between two RGB uint8 images."""
    return peak_signal_noise_ratio(target, pred, data_range=255)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM between two RGB uint8 images."""
    return structural_similarity(target, pred, channel_axis=2, data_range=255)


class MetricsComputer:
    """Computes PSNR, SSIM, and optionally LPIPS over a batch."""

    def __init__(self, use_lpips: bool = True):
        self.use_lpips = use_lpips
        self._lpips_model = None

    def _get_lpips(self):
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net="alex")
            self._lpips_model.eval()
        return self._lpips_model

    def compute_batch(
        self,
        pred_images: list[np.ndarray],
        gt_images: list[np.ndarray],
    ) -> dict[str, float]:
        """Compute metrics over a batch of images.

        Args:
            pred_images: List of (H, W, 3) uint8 RGB arrays.
            gt_images: List of (H, W, 3) uint8 RGB arrays.

        Returns:
            Dict with average PSNR, SSIM, and optionally LPIPS.
        """
        psnr_vals, ssim_vals = [], []

        for pred, gt in zip(pred_images, gt_images):
            psnr_vals.append(compute_psnr(pred, gt))
            ssim_vals.append(compute_ssim(pred, gt))

        result = {
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
        }

        if self.use_lpips:
            lpips_model = self._get_lpips()

            # Convert to tensors: (B, 3, H, W), range [-1, 1]
            pred_t = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
                for img in pred_images
            ])
            gt_t = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
                for img in gt_images
            ])

            with torch.no_grad():
                lpips_val = lpips_model(pred_t, gt_t).mean().item()
            result["lpips"] = lpips_val

        return result
