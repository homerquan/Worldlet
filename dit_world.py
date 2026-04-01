import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        # t: (N,)
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.mlp(embedding)


class DiTWorldModel(nn.Module):
    """
    Diffusion Transformer (DiT) that predicts the added noise in the next frame's continuous latents.
    """

    def __init__(
        self,
        channels=64,  # Latent channels from VAE
        latent_size=8,  # 8x8 spatial size
        num_actions=5,
        d_model=512,
        nhead=8,
        num_layers=8,
    ):
        super().__init__()
        self.channels = channels
        self.latent_size = latent_size
        self.seq_len = latent_size * latent_size  # 64 tokens

        # Time embedding
        self.t_embedder = TimestepEmbedder(d_model)

        # Action embedding
        self.action_embedder = nn.Embedding(num_actions, d_model)

        # Latent projection to d_model
        self.x_proj = nn.Linear(channels, d_model)
        self.cond_proj = nn.Linear(channels, d_model)

        # Positional embedding for 64 spatial tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection back to latent channels
        self.out_proj = nn.Linear(d_model, channels)

        # Initialize weights
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x_t, t, z_cond, action):
        """
        x_t: (B, C, H, W) Noisy next-frame latents
        t: (B,) Timesteps
        z_cond: (B, C, H, W) Previous frame latents
        action: (B,) Action indices
        """
        B = x_t.size(0)

        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x_t = x_t.view(B, self.channels, -1).permute(0, 2, 1)
        z_cond = z_cond.view(B, self.channels, -1).permute(0, 2, 1)

        # Embeddings
        x_emb = self.x_proj(x_t) + self.pos_emb  # (B, 64, d_model)
        cond_emb = self.cond_proj(z_cond) + self.pos_emb  # (B, 64, d_model)

        t_emb = self.t_embedder(t).unsqueeze(1)  # (B, 1, d_model)
        a_emb = self.action_embedder(action).unsqueeze(1)  # (B, 1, d_model)

        # Sequence: [t_emb, a_emb, cond_emb, x_emb]
        # Length = 1 + 1 + 64 + 64 = 130
        seq = torch.cat([t_emb, a_emb, cond_emb, x_emb], dim=1)

        # Pass through Transformer
        out_seq = self.transformer(seq)

        # Extract only the x_emb corresponding outputs (the last 64 tokens)
        x_out = out_seq[:, -self.seq_len :]

        # Project back to channels
        noise_pred = self.out_proj(x_out)  # (B, 64, channels)

        # Reshape to (B, C, H, W)
        noise_pred = noise_pred.permute(0, 2, 1).view(
            B, self.channels, self.latent_size, self.latent_size
        )
        return noise_pred


class DDPMSampler:
    def __init__(
        self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(
            -1, 1, 1, 1
        )

        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model, z_cond, action, shape, steps=50):
        """
        shape: (B, C, H, W)
        """
        B = shape[0]
        x_t = torch.randn(shape, device=self.device)

        # DDIM sampler
        B = shape[0]
        x_t = torch.randn(shape, device=self.device)
        step_size = self.num_timesteps // steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0

            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = model(x_t, t_tensor, z_cond, action)

            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = (
                self.alphas_cumprod[t_prev]
                if t_prev > 0
                else torch.tensor(1.0, device=self.device)
            )

            # Predict x_0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(
                alpha_cumprod_t
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred

            x_t = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt

        return x_t
