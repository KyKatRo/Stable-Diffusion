import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2

        self.alphas = 1 - self.betas
        self.alpha_comprod = torch.cumprod(self.alphas, 0)

        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_training_steps = num_training_steps

        self.timesteps = torch.from_numpy(np.arrange(0, num_training_steps)[:-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_training_steps / self.num_inference_steps
        timesteps = (np.arrange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, t: int) -> int:
        prev_t = t - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_timestep = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_comprod[timestep]
        alpha_prod_prev_t = self.alpha_comprod[prev_timestep] if prev_timestep >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_prev_t

        # Using the formula from DDPM paper (7)
        variance = (1 - alpha_prod_prev_t) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength: float = 1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        t = timestep
        prev_timestep = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_comprod[t]
        alpha_prod_prev_t = self.alpha_comprod[prev_timestep] if prev_timestep >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t

        # This equation is taken from DDPM paper formula (15)
        pred_original_sample = ((latents - beta_prod_t ** 0.5) * model_output) / alpha_prod_t ** 0.5

        # Computing coefficients for predicted original sampel and current sample x_t
        pred_original_sample_coeff = (alpha_prod_prev_t ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_prev_t) / beta_prod_t

        pred_prev_sample = pred_original_sample * pred_original_sample_coeff + latents * current_sample_coeff

        variance = 0
        if t >= 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        return pred_prev_sample + variance


    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumrod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps]**0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps])**0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod * noise)
        return noisy_samples
