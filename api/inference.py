"""
Inference engine — loads models and runs colorization pipeline.

Supports both PyTorch and ONNX inference.
"""

import os
import time

import cv2
import numpy as np
import torch

from data.transforms import (
    denormalize_ab,
    denormalize_l,
    lab_to_rgb,
    normalize_l,
    rgb_to_lab,
)
from model.diffusion import GaussianDiffusion
from model.unet import UNet


class InferenceEngine:
    """Loads trained model and runs colorization.

    Args:
        checkpoint_path: Path to training checkpoint (.pt).
        device: "cuda" or "cpu".
        use_ema: Use EMA weights (recommended).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        use_ema: bool = True,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})

        # Build model
        self.model = UNet(
            in_channels=3,
            out_channels=2,
            base_channels=config.get("base_channels", 64),
            channel_mult=tuple(config.get("channel_mult", (1, 2, 4, 8))),
            num_res_blocks=config.get("num_res_blocks", 2),
            attention_levels=tuple(config.get("attention_levels", (2, 3))),
            time_dim=config.get("time_dim", 256),
        ).to(self.device)

        if use_ema and "ema" in ckpt:
            self.model.load_state_dict(ckpt["ema"])
        else:
            self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # Diffusion
        self.diffusion = GaussianDiffusion(
            timesteps=config.get("timesteps", 1000),
            schedule=config.get("beta_schedule", "linear"),
        )

        self.image_size = config.get("image_size", 256)
        print(f"Model loaded on {self.device} ({self.image_size}x{self.image_size})")

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Convert input image to normalized L channel tensor.

        Handles both grayscale and color inputs.

        Returns:
            L tensor (1, 1, H, W) and original size (h, w).
        """
        original_size = image.shape[:2]

        # Convert to grayscale if color
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = rgb_to_lab(image)
            L = lab[:, :, 0:1]  # (H, W, 1)
        elif len(image.shape) == 2:
            # Already grayscale — treat as L channel
            L = image[:, :, None].astype(np.float32)
            # Scale from [0, 255] to [0, 100]
            L = L / 255.0 * 100.0
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Resize to model input size
        L_resized = cv2.resize(
            L.squeeze(-1), (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        L_resized = L_resized[:, :, None]

        # Normalize and to tensor
        L_norm = normalize_l(L_resized)  # [-1, 1]
        L_tensor = torch.from_numpy(
            L_norm.transpose(2, 0, 1)
        ).float().unsqueeze(0).to(self.device)  # (1, 1, H, W)

        return L_tensor, original_size

    def _postprocess(
        self, L: torch.Tensor, ab: torch.Tensor, original_size: tuple[int, int]
    ) -> np.ndarray:
        """Convert L + predicted AB back to RGB image at original resolution."""
        L_np = denormalize_l(L.squeeze(0).squeeze(0).cpu().numpy())  # (H, W)
        ab_np = denormalize_ab(ab.squeeze(0).cpu().numpy())  # (2, H, W)
        ab_np = ab_np.transpose(1, 2, 0)  # (H, W, 2)

        lab = np.concatenate([L_np[:, :, None], ab_np], axis=2).astype(np.float32)
        rgb = lab_to_rgb(lab)

        # Resize back to original
        h, w = original_size
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return rgb

    @torch.no_grad()
    def colorize(
        self,
        image: np.ndarray,
        num_steps: int = 20,
        method: str = "ddim",
    ) -> dict:
        """Colorize a grayscale or faded image.

        Args:
            image: Input image (H, W, 3) uint8 RGB or (H, W) uint8 grayscale.
            num_steps: Number of sampling steps.
            method: "ddim", "piecewise", or "dpm_solver".

        Returns:
            Dict with 'colorized' (RGB uint8), 'grayscale' (RGB uint8), 'time_ms' (float).
        """
        start = time.perf_counter()

        L_tensor, original_size = self._preprocess(image)
        shape = (1, 2, self.image_size, self.image_size)

        # Sample AB channels
        if method == "piecewise":
            seq = self.diffusion.piecewise_sequence(num_steps)
            ab_pred = self.diffusion.ddim_sample(
                self.model, L_tensor, shape, timestep_sequence=seq
            )
        elif method == "dpm_solver":
            ab_pred = self.diffusion.dpm_solver_sample(
                self.model, L_tensor, shape, num_steps=num_steps
            )
        else:
            ab_pred = self.diffusion.ddim_sample(
                self.model, L_tensor, shape, num_steps=num_steps
            )

        # Postprocess
        colorized = self._postprocess(L_tensor, ab_pred, original_size)

        # Also create grayscale version for comparison
        gray = cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB
        )

        elapsed = (time.perf_counter() - start) * 1000

        return {
            "colorized": colorized,
            "grayscale": gray,
            "time_ms": elapsed,
            "method": method,
            "steps": num_steps,
        }
