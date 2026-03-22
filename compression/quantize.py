"""
FP16 quantization of ONNX model.

Halves model size with negligible quality loss.

Usage:
    python compression/quantize.py \
        --input /path/to/model.onnx \
        --output /path/to/model_fp16.onnx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quantize_fp16(input_path: str, output_path: str):
    """Convert ONNX model from FP32 to FP16.

    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Where to save FP16 ONNX model.
    """
    import onnx
    from onnxconverter_common import float16

    print(f"Loading ONNX model: {input_path}")
    model = onnx.load(input_path)

    print("Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(model)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model_fp16, output_path)

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"FP32: {input_size:.1f} MB → FP16: {output_size:.1f} MB ({output_size/input_size*100:.0f}%)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to FP16")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    quantize_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
