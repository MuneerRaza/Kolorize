"""
FastAPI backend for Kolorize.

Usage:
    python api/main.py --checkpoint /path/to/checkpoint.pt --port 8000

Endpoints:
    POST /api/colorize    — Upload image, get colorized result
    GET  /api/health      — Health check
    GET  /api/model-info  — Model info and stats
"""

import argparse
import base64
import io
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_app(checkpoint_path: str, device: str | None = None):
    """Create FastAPI app with loaded model."""
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from api.inference import InferenceEngine

    app = FastAPI(title="Kolorize API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model
    engine = InferenceEngine(checkpoint_path=checkpoint_path, device=device)

    def numpy_to_base64(img: np.ndarray) -> str:
        """Convert RGB numpy array to base64 PNG string."""
        pil = Image.fromarray(img)
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/model-info")
    async def model_info():
        return {
            "model": "Kolorize Diffusion",
            "parameters": "68.9M",
            "image_size": engine.image_size,
            "device": str(engine.device),
        }

    @app.post("/api/colorize")
    async def colorize(
        file: UploadFile = File(...),
        steps: int = 20,
        method: str = "ddim",
    ):
        """Colorize an uploaded image.

        Args:
            file: Image file (PNG, JPG, etc.)
            steps: Number of sampling steps (10-50).
            method: Sampling method ("ddim", "piecewise", "dpm_solver").

        Returns:
            JSON with base64 encoded colorized and grayscale images + timing.
        """
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Clamp steps
        steps = max(5, min(50, steps))

        # Colorize
        result = engine.colorize(img, num_steps=steps, method=method)

        return {
            "colorized": numpy_to_base64(result["colorized"]),
            "grayscale": numpy_to_base64(result["grayscale"]),
            "time_ms": round(result["time_ms"], 1),
            "method": result["method"],
            "steps": result["steps"],
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Kolorize API server")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.checkpoint, args.device)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
