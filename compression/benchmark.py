"""
Benchmark FP32 vs FP16 models — size, latency, quality comparison.

Usage:
    python compression/benchmark.py \
        --checkpoint /path/to/checkpoint.pt \
        --onnx-fp32 /path/to/model.onnx \
        --onnx-fp16 /path/to/model_fp16.onnx \
        --data-dir /path/to/val_images \
        --output benchmark_results.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_pytorch(checkpoint_path: str, image_size: int = 256, num_runs: int = 20):
    """Benchmark PyTorch FP32 inference."""
    from model.unet import UNet

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    model = UNet(
        in_channels=3, out_channels=2,
        base_channels=config.get("base_channels", 64),
        channel_mult=tuple(config.get("channel_mult", (1, 2, 4, 8))),
        num_res_blocks=config.get("num_res_blocks", 2),
        attention_levels=tuple(config.get("attention_levels", (2, 3))),
        time_dim=config.get("time_dim", 256),
    )

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    t = torch.tensor([500])

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(dummy, t)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy, t)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "format": "PyTorch FP32",
        "file_size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
        "params": sum(p.numel() for p in model.parameters()),
    }


def benchmark_onnx(onnx_path: str, image_size: int = 256, num_runs: int = 20):
    """Benchmark ONNX Runtime inference."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    t = np.array([500], dtype=np.int64)

    # Warmup
    for _ in range(3):
        session.run(None, {"image": dummy, "timestep": t})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {"image": dummy, "timestep": t})
        times.append((time.perf_counter() - start) * 1000)

    label = "ONNX FP16" if "fp16" in onnx_path.lower() else "ONNX FP32"

    return {
        "format": label,
        "file_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark model variants")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx-fp32", type=str, default="")
    parser.add_argument("--onnx-fp16", type=str, default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    results = []

    print("Benchmarking PyTorch FP32...")
    results.append(benchmark_pytorch(args.checkpoint, args.image_size, args.num_runs))
    print(f"  {results[-1]['latency_ms_mean']:.1f}ms ± {results[-1]['latency_ms_std']:.1f}ms")

    if args.onnx_fp32 and os.path.exists(args.onnx_fp32):
        print("Benchmarking ONNX FP32...")
        results.append(benchmark_onnx(args.onnx_fp32, args.image_size, args.num_runs))
        print(f"  {results[-1]['latency_ms_mean']:.1f}ms ± {results[-1]['latency_ms_std']:.1f}ms")

    if args.onnx_fp16 and os.path.exists(args.onnx_fp16):
        print("Benchmarking ONNX FP16...")
        results.append(benchmark_onnx(args.onnx_fp16, args.image_size, args.num_runs))
        print(f"  {results[-1]['latency_ms_mean']:.1f}ms ± {results[-1]['latency_ms_std']:.1f}ms")

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Format':<20} {'Size (MB)':<12} {'Latency (ms)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['format']:<20} {r['file_size_mb']:<12.1f} {r['latency_ms_mean']:<15.1f}")
    print("=" * 60)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
