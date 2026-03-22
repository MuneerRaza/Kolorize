"""
Gradio app for HuggingFace Spaces deployment.

This is the entry point for the live demo.

Usage (local):
    python app.py --checkpoint /path/to/checkpoint.pt

On HuggingFace Spaces:
    Automatically runs with checkpoint from the repo.
"""

import argparse
import os
import sys

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.inference import InferenceEngine

# Global engine (loaded once)
engine = None


def load_engine(checkpoint_path: str):
    global engine
    engine = InferenceEngine(checkpoint_path=checkpoint_path, device="cpu")


def colorize_image(
    image: np.ndarray,
    num_steps: int,
    method: str,
) -> tuple[np.ndarray, str]:
    """Colorize an image and return result with timing info."""
    if image is None:
        return None, "Please upload an image."

    if engine is None:
        return None, "Model not loaded."

    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    result = engine.colorize(
        image,
        num_steps=int(num_steps),
        method=method,
    )

    info = f"Method: {result['method']} | Steps: {result['steps']} | Time: {result['time_ms']:.0f}ms"

    return result["colorized"], info


def build_app():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Kolorize — AI Image Colorization",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        ),
        css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        .title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
        """,
    ) as app:
        gr.HTML("<h1 class='title'>Kolorize</h1>")
        gr.HTML("<p class='subtitle'>Diffusion-based image colorization — bring old photos to life</p>")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload grayscale or faded image",
                    type="numpy",
                )
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=5, maximum=50, value=20, step=5,
                        label="Sampling Steps (more = better quality, slower)",
                    )
                    method_dropdown = gr.Dropdown(
                        choices=["ddim", "piecewise", "dpm_solver"],
                        value="ddim",
                        label="Sampling Method",
                    )
                colorize_btn = gr.Button("Colorize", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(label="Colorized Result", type="numpy")
                info_text = gr.Textbox(label="Info", interactive=False)

        colorize_btn.click(
            fn=colorize_image,
            inputs=[input_image, steps_slider, method_dropdown],
            outputs=[output_image, info_text],
        )

        gr.Examples(
            examples=[],  # Add pre-computed examples here later
            inputs=input_image,
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    load_engine(args.checkpoint)

    app = build_app()
    app.launch(server_port=args.port)
