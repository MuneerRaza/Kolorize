"""Training configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Data
    data_dir: str = ""                  # Path to image directory (CLI override)
    dataset_name: str = ""              # HuggingFace dataset name (alternative to data_dir)
    cache_dir: str = ""                 # HF dataset cache directory
    image_size: int = 256               # Training image size
    max_samples: int | None = None      # Limit dataset size (for testing)

    # Model
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_levels: tuple[int, ...] = (2, 3)
    time_dim: int = 256
    dropout: float = 0.0

    # Diffusion
    timesteps: int = 250                # 250 is enough (was 1000)
    beta_schedule: str = "linear"       # "linear" or "cosine"
    prediction_type: str = "v"          # "v" (faster convergence) or "epsilon" (legacy)
    snr_gamma: float = 5.0              # Min-SNR-γ weighting. 0 to disable.

    # Training
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs: int = 100
    gradient_clip: float = 1.0
    ema_decay: float = 0.9999
    use_amp: bool = True                # Mixed precision (FP16)

    # Loss
    use_perceptual: bool = True
    perceptual_weight: float = 0.1

    # Logging
    wandb_project: str = "colorize"
    wandb_run_name: str = ""
    log_interval: int = 100             # Log every N steps
    sample_interval: int = 1            # Generate samples every N epochs
    save_interval: int = 5              # Save checkpoint every N epochs
    num_sample_images: int = 4          # Number of images to generate for logging

    # Sampling (for validation samples during training)
    sample_steps: int = 50
    sample_method: str = "ddim"         # "ddim", "piecewise", "dpm_solver"

    # Paths (all via CLI — nothing hardcoded)
    output_dir: str = "./checkpoints"   # Where to save checkpoints
    resume: str = ""                    # Path to checkpoint to resume from

    # Workers
    num_workers: int = 4
