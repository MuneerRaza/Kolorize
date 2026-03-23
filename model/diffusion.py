"""
Diffusion process for colorization — optimized for fast convergence.

Key optimizations over standard DDPM:
1. v-prediction: Predicts v = √(ᾱ_t)·ε - √(1-ᾱ_t)·x₀ instead of just ε.
   More balanced gradient signal across all timesteps → ~30% faster convergence.

2. Min-SNR-γ weighting: Weights loss by signal-to-noise ratio so the model
   focuses on informative timesteps, not trivially easy/hard ones.
   → 2-3x faster convergence. (Hang et al., ICCV 2023)

3. 250 timesteps instead of 1000: Denser noise schedule, same quality.

Sampling strategies (all use the SAME trained model):
  1. DDIM uniform — evenly spaced timesteps (baseline)
  2. DDIM piecewise — dense in high-noise, sparse in low-noise
  3. DPM-Solver++ — better ODE solver, converges in 12-15 steps

Forward process:  x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε
v-target:         v_t = √(ᾱ_t) · ε - √(1-ᾱ_t) · x₀
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    """Optimized DDPM with v-prediction and Min-SNR-γ weighting.

    Args:
        timesteps: Number of diffusion steps (T). 250 recommended.
        beta_start: Starting noise level.
        beta_end: Ending noise level.
        schedule: Noise schedule type ("linear" or "cosine").
        prediction_type: "v" (recommended) or "epsilon" (legacy).
        snr_gamma: Min-SNR-γ clipping value. 5.0 recommended. 0 to disable.
    """

    def __init__(
        self,
        timesteps: int = 250,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        prediction_type: str = "v",
        snr_gamma: float = 5.0,
    ):
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma

        # Beta schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Pre-compute constants
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # Forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # SNR (Signal-to-Noise Ratio) for Min-SNR weighting
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

        # Reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule — smoother than linear."""
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 0.0001, 0.9999).float()

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Extract values at timestep indices, reshape for broadcasting."""
        out = tensor.to(t.device).gather(0, t)
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    # -------------------------------------------------------------------
    # Forward process
    # -------------------------------------------------------------------

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Add noise to x0 at timestep t.

        x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    # -------------------------------------------------------------------
    # v-prediction helpers
    # -------------------------------------------------------------------

    def get_v_target(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute v-prediction target.

        v = √(ᾱ_t) · ε - √(1-ᾱ_t) · x₀

        v is a blend of noise and signal. Predicting v gives the model
        a balanced gradient signal at ALL timesteps, unlike ε-prediction
        which struggles at low noise levels.
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x0

    def predict_x0_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x0 from x_t and predicted v.

        x₀ = √(ᾱ_t) · x_t - √(1-ᾱ_t) · v
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    def predict_noise_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct noise from x_t and predicted v.

        ε = √(ᾱ_t) · v + √(1-ᾱ_t) · x_t
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return sqrt_alpha * v + sqrt_one_minus_alpha * x_t

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x0 from x_t and predicted noise (ε-prediction).

        x₀ = (x_t - √(1-ᾱ_t) · ε) / √(ᾱ_t)
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

    # -------------------------------------------------------------------
    # Min-SNR-γ weighting
    # -------------------------------------------------------------------

    def get_snr_weights(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR-γ loss weights for given timesteps.

        weight = min(SNR(t), γ) / SNR(t)

        This downweights:
        - Very low noise (high SNR → easy, uninformative)
        - Very high noise (low SNR → too hard)
        And focuses on the informative middle range.

        Hang et al., "Efficient Diffusion Training via Min-SNR Weighting", ICCV 2023.
        """
        if self.snr_gamma <= 0:
            return torch.ones_like(t, dtype=torch.float32)

        snr_t = self.snr.to(t.device).gather(0, t)
        weights = torch.clamp(snr_t, max=self.snr_gamma) / snr_t
        return weights

    # -------------------------------------------------------------------
    # DDIM Sampling (works with both ε and v prediction)
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
        """DDIM sampling — works with both v-prediction and ε-prediction."""
        device = L_condition.device
        b = shape[0]

        if timestep_sequence is None:
            timestep_sequence = self._uniform_sequence(num_steps)

        x_t = torch.randn(shape, device=device)

        for i in range(len(timestep_sequence) - 1):
            t_curr = timestep_sequence[i]
            t_prev = timestep_sequence[i + 1]

            t_batch = torch.full((b,), t_curr, device=device, dtype=torch.long)

            # Model forward
            model_input = torch.cat([L_condition, x_t], dim=1)
            model_output = model(model_input, t_batch)

            # Convert model output to noise prediction
            if self.prediction_type == "v":
                predicted_noise = self.predict_noise_from_v(x_t, t_batch, model_output)
            else:
                predicted_noise = model_output

            # Get alpha values
            alpha_curr = self.alphas_cumprod[t_curr].to(device)
            alpha_prev = (
                self.alphas_cumprod[t_prev].to(device) if t_prev >= 0
                else torch.tensor(1.0, device=device)
            )

            # Predict x0
            x0_pred = (x_t - torch.sqrt(1 - alpha_curr) * predicted_noise) / torch.sqrt(alpha_curr)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # DDIM step
            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_curr) * (1 - alpha_curr / alpha_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise

            x_t = torch.sqrt(alpha_prev) * x0_pred + dir_xt

            if eta > 0 and t_prev > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        return x_t

    # -------------------------------------------------------------------
    # Timestep Sequences
    # -------------------------------------------------------------------

    def _uniform_sequence(self, num_steps: int) -> list[int]:
        """Uniform spacing."""
        step_size = self.timesteps // num_steps
        seq = list(range(self.timesteps - 1, -1, -step_size))[:num_steps]
        seq.append(-1)
        return seq

    def piecewise_sequence(
        self, num_steps: int, split_ratio: float = 0.5, density_ratio: float = 2.0
    ) -> list[int]:
        """Piecewise sampling — dense in high-noise, sparse in low-noise."""
        split_t = int(self.timesteps * split_ratio)

        steps_first = int(num_steps * density_ratio / (1 + density_ratio))
        steps_second = num_steps - steps_first

        if steps_first > 0:
            first_half = np.linspace(
                self.timesteps - 1, split_t, steps_first, dtype=int
            ).tolist()
        else:
            first_half = []

        if steps_second > 0:
            second_half = np.linspace(
                split_t - 1, 0, steps_second, dtype=int
            ).tolist()
        else:
            second_half = []

        seq = first_half + second_half
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
        """DPM-Solver++ — works with both prediction types."""
        device = L_condition.device
        b = shape[0]

        lambdas = torch.log(self.alphas_cumprod / (1 - self.alphas_cumprod)) / 2
        timestep_seq = self._uniform_sequence(num_steps)[:-1]

        x_t = torch.randn(shape, device=device)
        prev_noise_pred = None

        for i, t_curr in enumerate(timestep_seq):
            t_batch = torch.full((b,), t_curr, device=device, dtype=torch.long)

            model_input = torch.cat([L_condition, x_t], dim=1)
            model_output = model(model_input, t_batch)

            # Convert to noise prediction
            if self.prediction_type == "v":
                noise_pred = self.predict_noise_from_v(x_t, t_batch, model_output)
            else:
                noise_pred = model_output

            alpha_t = self.alphas_cumprod[t_curr].to(device)
            sigma_t = torch.sqrt(1 - alpha_t)

            x0_pred = (x_t - sigma_t * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i < len(timestep_seq) - 1:
                t_next = timestep_seq[i + 1]
                alpha_next = self.alphas_cumprod[t_next].to(device)
                sigma_next = torch.sqrt(1 - alpha_next)

                if prev_noise_pred is not None and i > 0:
                    lambda_t = lambdas[t_curr].to(device)
                    lambda_next = lambdas[t_next].to(device)
                    lambda_prev = lambdas[timestep_seq[i - 1]].to(device)

                    h = lambda_next - lambda_t
                    h_prev = lambda_t - lambda_prev
                    r = h_prev / h

                    d = (1 + 1 / (2 * r)) * noise_pred - (1 / (2 * r)) * prev_noise_pred
                    x0_corrected = (x_t - sigma_t * d) / torch.sqrt(alpha_t)
                    x0_corrected = torch.clamp(x0_corrected, -1.0, 1.0)

                    x_t = torch.sqrt(alpha_next) * x0_corrected + sigma_next * d
                else:
                    x_t = torch.sqrt(alpha_next) * x0_pred + sigma_next * noise_pred

            prev_noise_pred = noise_pred

        return x0_pred

    # -------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------

    def training_step(
        self, model: nn.Module, L: torch.Tensor, ab: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Single training step with v-prediction and Min-SNR weighting.

        Args:
            model: UNet model.
            L: L channel (B, 1, H, W), normalized [-1, 1].
            ab: AB channels (B, 2, H, W), normalized [-1, 1].

        Returns:
            Dict with prediction, target, x0_pred, x0_target, t, snr_weights.
        """
        b = L.shape[0]
        device = L.device

        # Random timestep
        t = torch.randint(0, self.timesteps, (b,), device=device)

        # Sample noise
        noise = torch.randn_like(ab)

        # Forward process
        noisy_ab = self.q_sample(ab, t, noise)

        # Model input
        model_input = torch.cat([L, noisy_ab], dim=1)

        # Model predicts v or epsilon
        model_output = model(model_input, t)

        # Compute target and x0 based on prediction type
        if self.prediction_type == "v":
            target = self.get_v_target(ab, noise, t)
            x0_pred = self.predict_x0_from_v(noisy_ab, t, model_output)
        else:
            target = noise
            x0_pred = self.predict_x0_from_noise(noisy_ab, t, model_output)

        # Min-SNR weights
        snr_weights = self.get_snr_weights(t)

        return {
            "model_output": model_output,
            "target": target,
            "x0_pred": x0_pred,
            "x0_target": ab,
            "t": t,
            "snr_weights": snr_weights,
        }
