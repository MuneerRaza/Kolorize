"""
LAB color space transforms and utilities.

Why LAB?
- L channel = luminance (structure/grayscale) — range [0, 100]
- A channel = green↔red color axis — range [-128, 127]
- B channel = blue↔yellow color axis — range [-128, 127]

We normalize both to [-1, 1] for the diffusion model.
OpenCV uses different ranges (L: [0,255], A/B: [0,255] for uint8),
so we convert to float32 first to get standard LAB ranges.
"""

import cv2
import numpy as np
import torch


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to LAB float32 with standard ranges.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image [0, 255]

    Returns:
        (H, W, 3) float32 LAB image. L: [0, 100], A/B: [-128, 127]
    """
    # OpenCV expects BGR, not RGB
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # Convert to float32 [0, 1] range first — gives standard LAB ranges
    bgr_float = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB)
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB float32 image back to RGB uint8.

    Args:
        lab: (H, W, 3) float32 LAB. L: [0, 100], A/B: [-128, 127]

    Returns:
        (H, W, 3) uint8 RGB image [0, 255]
    """
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = np.clip(bgr * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def normalize_l(L: np.ndarray) -> np.ndarray:
    """Normalize L channel from [0, 100] to [-1, 1]."""
    return (L - 50.0) / 50.0


def denormalize_l(L: np.ndarray) -> np.ndarray:
    """Denormalize L channel from [-1, 1] to [0, 100]."""
    return L * 50.0 + 50.0


def normalize_ab(ab: np.ndarray) -> np.ndarray:
    """Normalize AB channels from [-128, 127] to [-1, 1]."""
    return ab / 128.0


def denormalize_ab(ab: np.ndarray) -> np.ndarray:
    """Denormalize AB channels from [-1, 1] to [-128, 127]."""
    return ab * 128.0


def lab_tensors_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """Convert normalized L and AB tensors back to RGB numpy image.

    Args:
        L: (1, H, W) tensor, normalized [-1, 1]
        ab: (2, H, W) tensor, normalized [-1, 1]

    Returns:
        (H, W, 3) uint8 RGB numpy array
    """
    L_np = denormalize_l(L.squeeze(0).cpu().numpy())       # (H, W), [0, 100]
    ab_np = denormalize_ab(ab.cpu().numpy())                # (2, H, W), [-128, 127]
    ab_np = ab_np.transpose(1, 2, 0)                       # (H, W, 2)

    lab = np.concatenate([L_np[:, :, None], ab_np], axis=2)  # (H, W, 3)
    lab = lab.astype(np.float32)
    return lab_to_rgb(lab)


def lab_batch_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> list[np.ndarray]:
    """Convert a batch of normalized L and AB tensors to RGB images.

    Args:
        L: (B, 1, H, W) tensor, normalized [-1, 1]
        ab: (B, 2, H, W) tensor, normalized [-1, 1]

    Returns:
        List of (H, W, 3) uint8 RGB numpy arrays
    """
    batch_size = L.shape[0]
    images = []
    for i in range(batch_size):
        images.append(lab_tensors_to_rgb(L[i], ab[i]))
    return images
