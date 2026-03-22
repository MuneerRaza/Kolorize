"""
Super-Resolution using pretrained Real-ESRGAN (4x upscaling).

No training needed — just download weights and use.
Real-ESRGAN uses an RRDB (Residual-in-Residual Dense Block) architecture
trained with perceptual + adversarial losses for sharp, realistic upscaling.

Usage:
    sr = SuperResolution()
    enhanced = sr.enhance(rgb_image_numpy)  # (H,W,3) uint8 → (4H,4W,3) uint8
"""

import os
import urllib.request

import cv2
import numpy as np
import torch


class SuperResolution:
    """Real-ESRGAN 4x super-resolution wrapper.

    Args:
        model_name: Which Real-ESRGAN model to use.
        scale: Upscaling factor (default 4).
        weights_dir: Directory to download/store weights.
        half: Use FP16 inference (faster, less VRAM).
        device: "cuda" or "cpu".
    """

    MODELS = {
        "RealESRGAN_x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "num_block": 23,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        "RealESRGAN_x4plus_anime_6B": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "num_block": 6,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
    }

    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        scale: int = 4,
        weights_dir: str = "./weights",
        half: bool = True,
        device: str | None = None,
    ):
        self.scale = scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device == "cuda"

        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODELS.keys())}")

        model_info = self.MODELS[model_name]

        # Download weights if needed
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f"{model_name}.pth")

        if not os.path.exists(weights_path):
            print(f"Downloading {model_name} weights...")
            urllib.request.urlretrieve(model_info["url"], weights_path)
            print(f"Saved to {weights_path}")

        # Build RRDB model
        from model._rrdb import RRDBNet

        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=model_info["num_feat"],
            num_block=model_info["num_block"],
            num_grow_ch=model_info["num_grow_ch"],
            scale=scale,
        )

        # Load weights
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

        if self.half:
            self.model = self.model.half()

    @torch.no_grad()
    def enhance(self, image: np.ndarray, outscale: int | None = None) -> np.ndarray:
        """Upscale an RGB image.

        Args:
            image: (H, W, 3) uint8 RGB numpy array.
            outscale: Output scale (default: self.scale).

        Returns:
            (H*scale, W*scale, 3) uint8 RGB numpy array.
        """
        if outscale is None:
            outscale = self.scale

        # Preprocess: uint8 RGB → float32 tensor
        img = image.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        img = img.to(self.device)

        if self.half:
            img = img.half()

        # Inference
        output = self.model(img)

        # Postprocess: tensor → uint8 RGB
        output = output.squeeze(0).float().clamp(0, 1).cpu().numpy()
        output = (output.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

        # Resize if outscale differs from model scale
        if outscale != self.scale:
            h, w = image.shape[:2]
            output = cv2.resize(
                output, (int(w * outscale), int(h * outscale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        return output
