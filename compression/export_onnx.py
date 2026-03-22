"""
Export trained UNet to ONNX format for deployment.

Usage:
    python compression/export_onnx.py \
        --checkpoint /path/to/checkpoint.pt \
        --output /path/to/model.onnx \
        --image-size 256
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.unet import UNet


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    image_size: int = 256,
    use_ema: bool = True,
):
    """Export UNet checkpoint to ONNX.

    Args:
        checkpoint_path: Path to training checkpoint (.pt).
        output_path: Where to save the ONNX model.
        image_size: Image size the model was trained on.
        use_ema: Use EMA weights (recommended for best quality).
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild model from saved config
    config = ckpt.get("config", {})
    model = UNet(
        in_channels=3,
        out_channels=2,
        base_channels=config.get("base_channels", 64),
        channel_mult=tuple(config.get("channel_mult", (1, 2, 4, 8))),
        num_res_blocks=config.get("num_res_blocks", 2),
        attention_levels=tuple(config.get("attention_levels", (2, 3))),
        time_dim=config.get("time_dim", 256),
    )

    # Load weights (EMA or regular)
    if use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")

    model.eval()

    # Dummy inputs
    dummy_image = torch.randn(1, 3, image_size, image_size)
    dummy_t = torch.tensor([500])

    # Export
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_image, dummy_t),
        output_path,
        input_names=["image", "timestep"],
        output_names=["predicted_noise"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "predicted_noise": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved: {file_size:.1f} MB")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export UNet to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.image_size, not args.no_ema)


if __name__ == "__main__":
    main()
