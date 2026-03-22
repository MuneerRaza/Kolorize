"""
Training script for colorization diffusion model.

Usage:
    # Local testing (CPU, tiny config)
    python training/train.py --data-dir ./test_images --image-size 64 \
        --base-channels 32 --num-res-blocks 1 --batch-size 2 --epochs 2 \
        --time-dim 128 --no-wandb --output-dir ./test_checkpoints

    # Kaggle (full config)
    python training/train.py --dataset-name nickpai/coco2017-colorization \
        --cache-dir /kaggle/working/cache --output-dir /kaggle/working/checkpoints \
        --wandb-project colorize --epochs 100

    # Resume training
    python training/train.py --resume /path/to/checkpoint.pt [other args]
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ColorizationDataset, HuggingFaceColorizationDataset
from data.transforms import lab_batch_to_rgb
from model.diffusion import GaussianDiffusion
from model.losses import ColorizationLoss
from model.unet import UNet
from training.config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train colorization diffusion model")

    # Data
    parser.add_argument("--data-dir", type=str, default="", help="Path to image directory")
    parser.add_argument("--dataset-name", type=str, default="", help="HuggingFace dataset name")
    parser.add_argument("--cache-dir", type=str, default="", help="HF dataset cache dir")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")

    # Model
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    # Loss
    parser.add_argument("--no-perceptual", action="store_true")
    parser.add_argument("--perceptual-weight", type=float, default=0.1)

    # Logging
    parser.add_argument("--wandb-project", type=str, default="colorize")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--num-sample-images", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=50)

    # Paths
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")

    # Workers
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Build config from parsed arguments."""
    return TrainConfig(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        time_dim=args.time_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        gradient_clip=args.gradient_clip,
        ema_decay=args.ema_decay,
        use_amp=not args.no_amp,
        use_perceptual=not args.no_perceptual,
        perceptual_weight=args.perceptual_weight,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        save_interval=args.save_interval,
        num_sample_images=args.num_sample_images,
        sample_steps=args.sample_steps,
        output_dir=args.output_dir,
        resume=args.resume,
        num_workers=args.num_workers,
    )


class EMA:
    """Exponential Moving Average of model weights.

    Keeps a shadow copy that updates slowly:
        ema_weight = decay × ema_weight + (1 - decay) × model_weight

    EMA model produces smoother, more stable outputs. Always use for inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def build_dataloader(config: TrainConfig, split: str = "train") -> DataLoader:
    """Build dataset and dataloader from config."""
    if config.data_dir:
        dataset = ColorizationDataset(
            image_dir=config.data_dir,
            image_size=config.image_size,
            split=split,
            max_samples=config.max_samples,
        )
    elif config.dataset_name:
        hf_split = "train" if split == "train" else "validation"
        dataset = HuggingFaceColorizationDataset(
            dataset_name=config.dataset_name,
            split=hf_split,
            image_size=config.image_size,
            cache_dir=config.cache_dir or None,
            max_samples=config.max_samples,
        )
    else:
        raise ValueError("Must specify either --data-dir or --dataset-name")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    val_batch: dict[str, torch.Tensor],
    config: TrainConfig,
    device: torch.device,
) -> list:
    """Generate colorized samples for logging."""
    model.eval()

    L = val_batch["L"][:config.num_sample_images].to(device)
    ab_gt = val_batch["ab"][:config.num_sample_images].to(device)
    b = L.shape[0]

    # Sample using DDIM
    ab_pred = diffusion.ddim_sample(
        model, L, (b, 2, config.image_size, config.image_size),
        num_steps=config.sample_steps,
    )

    # Convert to RGB images
    pred_images = lab_batch_to_rgb(L, ab_pred)
    gt_images = lab_batch_to_rgb(L, ab_gt)
    gray_images = lab_batch_to_rgb(L, torch.zeros_like(ab_gt))

    model.train()
    return gray_images, pred_images, gt_images


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    config: TrainConfig,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Always save unwrapped model (works with or without DataParallel)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "model": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "config": vars(config),
    }, path)
    print(f"  Checkpoint saved: {path}")


def train(config: TrainConfig):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Build model
    model = UNet(
        in_channels=3,
        out_channels=2,
        base_channels=config.base_channels,
        channel_mult=config.channel_mult,
        num_res_blocks=config.num_res_blocks,
        attention_levels=config.attention_levels,
        time_dim=config.time_dim,
        dropout=config.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Multi-GPU support
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # EMA (always on the unwrapped model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    ema = EMA(raw_model, decay=config.ema_decay)

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=config.timesteps,
        schedule=config.beta_schedule,
    )

    # Loss
    criterion = ColorizationLoss(
        perceptual_weight=config.perceptual_weight,
        use_perceptual=config.use_perceptual,
    ).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if (config.use_amp and device.type == "cuda") else None

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if config.resume and os.path.exists(config.resume):
        ckpt = torch.load(config.resume, map_location=device, weights_only=False)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and scaler:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("step", 0)
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Data
    train_loader = build_dataloader(config, "train")
    print(f"Training samples: {len(train_loader.dataset)}")

    # Grab a fixed validation batch for sample generation
    val_batch = next(iter(train_loader))

    # W&B
    use_wandb = not hasattr(config, "_no_wandb") and config.wandb_project
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or None,
                config=vars(config),
                resume="allow",
            )
            if run is None:
                raise RuntimeError("wandb.init returned None")
            print(f"W&B run: {run.url}")
        except Exception as e:
            use_wandb = False
            print(f"W&B init failed: {e}, continuing without logging")

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    print(f"\nStarting training for {config.epochs} epochs...")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_noise_loss = 0.0
        epoch_perc_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch in pbar:
            L = batch["L"].to(device)
            ab = batch["ab"].to(device)

            optimizer.zero_grad()

            # Forward pass with optional mixed precision
            # Perceptual loss every 10 steps to save VRAM
            compute_perceptual = config.use_perceptual and (num_batches % 10 == 0)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                result = diffusion.training_step(model, L, ab)

                if compute_perceptual:
                    losses = criterion(
                        noise_pred=result["noise_pred"],
                        noise_target=result["noise_target"],
                        x0_pred=result["x0_pred"].detach(),
                        x0_target=result["x0_target"],
                        L=L,
                    )
                else:
                    losses = criterion(
                        noise_pred=result["noise_pred"],
                        noise_target=result["noise_target"],
                    )
                loss = losses["total"]

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()

            # EMA update
            ema.update(model)

            # Track losses
            epoch_loss += loss.item()
            epoch_noise_loss += losses["noise_loss"].item()
            epoch_perc_loss += losses["perceptual_loss"].item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "noise": f"{losses['noise_loss'].item():.4f}",
                "perc": f"{losses['perceptual_loss'].item():.4f}",
            })

            # Step-level logging
            if use_wandb and global_step % config.log_interval == 0:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/noise_loss": losses["noise_loss"].item(),
                    "train/perceptual_loss": losses["perceptual_loss"].item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/step": global_step,
                }, step=global_step)

        # End of epoch
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_noise = epoch_noise_loss / max(num_batches, 1)
        avg_perc = epoch_perc_loss / max(num_batches, 1)

        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f} noise={avg_noise:.4f} perc={avg_perc:.4f}")

        # Generate samples
        if (epoch + 1) % config.sample_interval == 0:
            print("  Generating samples...")
            gray, pred, gt = generate_samples(
                ema.shadow, diffusion, val_batch, config, device
            )

            if use_wandb:
                import wandb
                import numpy as np
                images = []
                for i in range(len(gray)):
                    images.append(wandb.Image(
                        np.concatenate([gray[i], pred[i], gt[i]], axis=1),
                        caption=f"Gray | Predicted | Ground Truth (sample {i})"
                    ))
                wandb.log({"samples": images, "epoch": epoch + 1}, step=global_step)

            # Save sample images locally
            samples_dir = os.path.join(config.output_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            import cv2
            import numpy as np
            for i in range(len(gray)):
                combined = np.concatenate([gray[i], pred[i], gt[i]], axis=1)
                cv2.imwrite(
                    os.path.join(samples_dir, f"epoch{epoch + 1:03d}_sample{i}.png"),
                    cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
                )

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(
                os.path.join(config.output_dir, f"checkpoint_epoch{epoch + 1:03d}.pt"),
                model, ema, optimizer, scheduler, scaler, epoch, global_step, config,
            )

    # Save final checkpoint
    save_checkpoint(
        os.path.join(config.output_dir, "checkpoint_final.pt"),
        model, ema, optimizer, scheduler, scaler, config.epochs - 1, global_step, config,
    )

    if use_wandb:
        import wandb
        wandb.finish()

    print("\nTraining complete!")


def main():
    args = parse_args()
    config = build_config(args)

    # Disable wandb if --no-wandb flag
    if args.no_wandb:
        config._no_wandb = True

    train(config)


if __name__ == "__main__":
    main()
