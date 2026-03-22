"""
Diffusion process for colorization — DDPM training + multiple sampling strategies.

Training: Standard DDPM (add noise, predict noise, L1 loss)
Sampling strategies (all use the SAME trained model):
  1. DDIM uniform — evenly spaced timesteps (baseline)
  2. DDIM piecewise — dense steps in high-noise region, sparse in low-noise
     (from Tang et al., ACM MM 2023)
  3. DPM-Solver++ — better ODE solver, converges in 12-15 steps

Forward process:  x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε
Reverse process:  predict ε, compute x_{t-1}
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    """DDPM forward/reverse process with multiple sampling strategies.

    Args:
        timesteps: Number of diffusion steps (T).
        beta_start: Starting noise level.
        beta_end: Ending noise level.
        schedule: Noise schedule type ("linear" or "cosine").
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.timesteps = timesteps

        # Beta schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Pre-compute all the constants we need
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # Forward process constants
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Reverse process constants
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule — smoother than linear, sometimes better."""
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 0.0001, 0.9999).float()

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Extract values from tensor at timestep indices, reshape for broadcasting."""
        out = tensor.to(t.device).gather(0, t)
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    # -------------------------------------------------------------------
    # Forward process (used during training)
    # -------------------------------------------------------------------

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Add noise to x0 at timestep t (forward process shortcut).

        x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε

        Args:
            x0: Clean AB channels (B, 2, H, W), normalized [-1, 1]
            t: Timesteps (B,)
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy AB channels at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x0 from x_t and predicted noise.

        x₀ = (x_t - √(1-ᾱ_t) · ε) / √(ᾱ_t)
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

    # -------------------------------------------------------------------
    # DDIM Sampling
    # -------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        L_condition: torch.Tensor,
        shape: tuple,
        num_steps: int = 50,
        eta: float = 0.0,
        timestep_sequence: list[int] | None = None,
    ) -> torch.Tensor:
        """DDIM sampling — deterministic (eta=0) or stochastic.

        Same trained model as DDPM, just a smarter sampling strategy.
        With eta=0: deterministic, skip steps freely.
        With eta=1: equivalent to DDPM.

        Args:
            model: Trained UNet.
            L_condition: L channel (B, 1, H, W), normalized [-1, 1].
            shape: Shape of AB to generate (B, 2, H, W).
            num_steps: Number of sampling steps.
            eta: Stochasticity (0 = deterministic DDIM, 1 = DDPM).
            timestep_sequence: Custom timestep sequence (for piecewise/searched).
                If None, uses uniform spacing.

        Returns:
            Predicted AB channels (B, 2, H, W), normalized [-1, 1].
        """
        device = L_condition.device
        b = shape[0]

        # Build timestep sequence
        if timestep_sequence is None:
            timestep_sequence = self._uniform_sequence(num_steps)

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        for i in range(len(timestep_sequence) - 1):
            t_curr = timestep_sequence[i]
            t_prev = timestep_sequence[i + 1]

            t_batch = torch.full((b,), t_curr, device=device, dtype=torch.long)

            # Model predicts noise
            model_input = torch.cat([L_condition, x_t], dim=1)
            predicted_noise = model(model_input, t_batch)

            # Get alpha values
            alpha_curr = self.alphas_cumprod[t_curr].to(device)
            alpha_prev = (
                self.alphas_cumprod[t_prev].to(device) if t_prev >= 0
                else torch.tensor(1.0, device=device)
            )

            # Predict x0
            x0_pred = (x_t - torch.sqrt(1 - alpha_curr) * predicted_noise) / torch.sqrt(alpha_curr)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # Direction pointing to x_t
            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_curr) * (1 - alpha_curr / alpha_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise

            # Compute x_{t-1}
            x_t = torch.sqrt(alpha_prev) * x0_pred + dir_xt

            if eta > 0 and t_prev > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        return x_t

    # -------------------------------------------------------------------
    # Timestep Sequences
    # -------------------------------------------------------------------

    def _uniform_sequence(self, num_steps: int) -> list[int]:
        """Uniform spacing: evenly distributed timesteps.

        E.g., T=1000, steps=10: [999, 899, 799, ..., 99, -1]
        """
        step_size = self.timesteps // num_steps
        seq = list(range(self.timesteps - 1, -1, -step_size))[:num_steps]
        seq.append(-1)  # Final step (fully denoised)
        return seq

    def piecewise_sequence(
        self, num_steps: int, split_ratio: float = 0.5, density_ratio: float = 2.0
    ) -> list[int]:
        """Piecewise sampling — dense in high-noise, sparse in low-noise region.

        From Tang et al. (ACM MM 2023): early denoising steps (high noise → medium)
        matter more than late steps (low noise → clean). Allocate more steps to
        the important early phase.

        Args:
            num_steps: Total number of steps.
            split_ratio: Where to split (0.5 = split at midpoint of timesteps).
            density_ratio: How much denser the first half is (2.0 = 2x more steps).

        Returns:
            Timestep sequence (descending, ends with -1).
        """
        split_t = int(self.timesteps * split_ratio)

        # Allocate steps: more to first half (high noise region)
        steps_first = int(num_steps * density_ratio / (1 + density_ratio))
        steps_second = num_steps - steps_first

        # First half: dense sampling in [T-1, split_t]
        if steps_first > 0:
            first_half = np.linspace(
                self.timesteps - 1, split_t, steps_first, dtype=int
            ).tolist()
        else:
            first_half = []

        # Second half: sparse sampling in [split_t-1, 0]
        if steps_second > 0:
            second_half = np.linspace(
                split_t - 1, 0, steps_second, dtype=int
            ).tolist()
        else:
            second_half = []

        seq = first_half + second_half
        # Remove duplicates while preserving order
        seen = set()
        seq = [x for x in seq if not (x in seen or seen.add(x))]
        seq.append(-1)
        return seq

    # -------------------------------------------------------------------
    # DPM-Solver++ (2nd order)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def dpm_solver_sample(
        self,
        model: nn.Module,
        L_condition: torch.Tensor,
        shape: tuple,
        num_steps: int = 15,
    ) -> torch.Tensor:
        """DPM-Solver++ (2nd order) — fast ODE solver for diffusion.

        Converges in 12-15 steps with quality matching 50-step DDIM.
        Uses 2nd-order multistep method for better accuracy per step.

        Args:
            model: Trained UNet.
            L_condition: L channel (B, 1, H, W).
            shape: Shape of AB to generate (B, 2, H, W).
            num_steps: Number of solver steps (12-15 recommended).

        Returns:
            Predicted AB channels (B, 2, H, W).
        """
        device = L_condition.device
        b = shape[0]

        # Compute lambda (log-SNR) schedule
        lambdas = torch.log(self.alphas_cumprod / (1 - self.alphas_cumprod)) / 2
        timestep_seq = self._uniform_sequence(num_steps)[:-1]  # Remove -1

        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        prev_noise_pred = None

        for i, t_curr in enumerate(timestep_seq):
            t_batch = torch.full((b,), t_curr, device=device, dtype=torch.long)

            # Predict noise
            model_input = torch.cat([L_condition, x_t], dim=1)
            noise_pred = model(model_input, t_batch)

            # Get alpha values
            alpha_t = self.alphas_cumprod[t_curr].to(device)
            sigma_t = torch.sqrt(1 - alpha_t)

            # Predict x0
            x0_pred = (x_t - sigma_t * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i < len(timestep_seq) - 1:
                t_next = timestep_seq[i + 1]
                alpha_next = self.alphas_cumprod[t_next].to(device)
                sigma_next = torch.sqrt(1 - alpha_next)

                # 2nd order correction using previous prediction
                if prev_noise_pred is not None and i > 0:
                    # DPM-Solver++ 2nd order
                    lambda_t = lambdas[t_curr].to(device)
                    lambda_next = lambdas[t_next].to(device)
                    lambda_prev = lambdas[timestep_seq[i - 1]].to(device)

                    h = lambda_next - lambda_t
                    h_prev = lambda_t - lambda_prev
                    r = h_prev / h

                    # 2nd order correction
                    d = (1 + 1 / (2 * r)) * noise_pred - (1 / (2 * r)) * prev_noise_pred
                    x0_corrected = (x_t - sigma_t * d) / torch.sqrt(alpha_t)
                    x0_corrected = torch.clamp(x0_corrected, -1.0, 1.0)

                    x_t = torch.sqrt(alpha_next) * x0_corrected + sigma_next * d
                else:
                    # 1st order (Euler step)
                    x_t = torch.sqrt(alpha_next) * x0_pred + sigma_next * noise_pred

            prev_noise_pred = noise_pred

        return x0_pred

    # -------------------------------------------------------------------
    # Training helper
    # -------------------------------------------------------------------

    def training_step(
        self, model: nn.Module, L: torch.Tensor, ab: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Single training step — sample noise, predict it, return losses.

        Args:
            model: UNet model.
            L: L channel (B, 1, H, W), normalized [-1, 1].
            ab: AB channels (B, 2, H, W), normalized [-1, 1].

        Returns:
            Dict with 'noise_pred', 'noise_target', 'x0_pred', 'x0_target', 't'
        """
        b = L.shape[0]
        device = L.device

        # Random timestep for each sample
        t = torch.randint(0, self.timesteps, (b,), device=device)

        # Sample noise
        noise = torch.randn_like(ab)

        # Forward process: add noise to AB
        noisy_ab = self.q_sample(ab, t, noise)

        # Model input: concat L + noisy AB
        model_input = torch.cat([L, noisy_ab], dim=1)  # (B, 3, H, W)

        # Predict noise
        noise_pred = model(model_input, t)

        # Also compute predicted x0 (for optional perceptual loss)
        x0_pred = self.predict_x0_from_noise(noisy_ab, t, noise_pred)

        return {
            "noise_pred": noise_pred,
            "noise_target": noise,
            "x0_pred": x0_pred,
            "x0_target": ab,
            "t": t,
        }
