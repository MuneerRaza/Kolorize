"""
Colorization dataset — loads images, converts to LAB, returns L and AB tensors.

Supports two modes:
1. HuggingFace dataset (e.g. nickpai/coco2017-colorization)
2. Local directory of images (any folder of JPGs/PNGs)

All paths are passed via arguments — nothing hardcoded.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import normalize_ab, normalize_l, rgb_to_lab


class ColorizationDataset(Dataset):
    """Dataset that converts RGB images to LAB and returns L (input) and AB (target).

    Args:
        image_dir: Path to directory containing RGB images.
        image_size: Size to resize/crop images to (square).
        split: "train" or "val" — controls augmentation.
        max_samples: Limit number of samples (useful for testing).
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        image_dir: str,
        image_size: int = 256,
        split: str = "train",
        max_samples: int | None = None,
    ):
        self.image_size = image_size
        self.split = split

        # Collect all image paths
        image_dir = Path(image_dir)
        self.image_paths = sorted(
            p
            for p in image_dir.rglob("*")
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load image as RGB
        img_path = str(self.image_paths[idx])
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            # Fallback to a random other image if this one is corrupted
            return self.__getitem__(random.randint(0, len(self) - 1))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Resize — maintain aspect ratio then crop, or just resize directly
        rgb = self._resize_and_crop(rgb)

        # Augmentation (train only)
        if self.split == "train":
            rgb = self._augment(rgb)

        # Convert to LAB
        lab = rgb_to_lab(rgb)  # (H, W, 3), L: [0,100], AB: [-128,127]

        # Split into L and AB
        L = lab[:, :, 0:1]  # (H, W, 1)
        ab = lab[:, :, 1:3]  # (H, W, 2)

        # Normalize to [-1, 1]
        L = normalize_l(L)
        ab = normalize_ab(ab)

        # Convert to tensors (C, H, W)
        L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).float()  # (1, H, W)
        ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).float()  # (2, H, W)

        return {"L": L_tensor, "ab": ab_tensor}

    def _resize_and_crop(self, rgb: np.ndarray) -> np.ndarray:
        """Resize shortest side to image_size, then center/random crop."""
        h, w = rgb.shape[:2]
        size = self.image_size

        # Resize shortest side to target size
        if h < w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)

        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Crop
        if self.split == "train":
            # Random crop
            top = random.randint(0, new_h - size)
            left = random.randint(0, new_w - size)
        else:
            # Center crop
            top = (new_h - size) // 2
            left = (new_w - size) // 2

        rgb = rgb[top : top + size, left : left + size]
        return rgb

    def _augment(self, rgb: np.ndarray) -> np.ndarray:
        """Training augmentations — horizontal flip."""
        if random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
        return rgb


class HuggingFaceColorizationDataset(Dataset):
    """Wraps a HuggingFace dataset for colorization.

    Expects the dataset to have an 'image' column with PIL images.

    Args:
        dataset_name: HuggingFace dataset ID (e.g. "nickpai/coco2017-colorization").
        split: "train" or "validation".
        image_size: Size to resize/crop images to.
        cache_dir: Where to cache the downloaded dataset.
        max_samples: Limit number of samples.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_size: int = 256,
        cache_dir: str | None = None,
        max_samples: int | None = None,
    ):
        from datasets import load_dataset

        self.image_size = image_size
        self.split = split

        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.ds = ds

        # Find the image column name
        self.image_column = None
        for col in ds.column_names:
            if col in ("image", "img", "input_image"):
                self.image_column = col
                break
        if self.image_column is None:
            # Try first column
            self.image_column = ds.column_names[0]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.ds[idx]
        pil_image = item[self.image_column]

        # Convert PIL to RGB numpy
        rgb = np.array(pil_image.convert("RGB"))

        # Resize and crop
        rgb = self._resize_and_crop(rgb)

        # Augmentation
        if self.split == "train":
            if random.random() > 0.5:
                rgb = np.fliplr(rgb).copy()

        # Convert to LAB
        lab = rgb_to_lab(rgb)

        L = lab[:, :, 0:1]
        ab = lab[:, :, 1:3]

        L = normalize_l(L)
        ab = normalize_ab(ab)

        L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).float()
        ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).float()

        return {"L": L_tensor, "ab": ab_tensor}

    def _resize_and_crop(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        size = self.image_size

        if h < w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)

        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.split == "train":
            top = random.randint(0, max(0, new_h - size))
            left = random.randint(0, max(0, new_w - size))
        else:
            top = (new_h - size) // 2
            left = (new_w - size) // 2

        rgb = rgb[top : top + size, left : left + size]
        return rgb
