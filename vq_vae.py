import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from safetensors.torch import save_file

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs shape: (B, C, H, W)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (
            quantized.permute(0, 3, 1, 2).contiguous(),
            loss,
            perplexity,
            encoding_indices,
        )


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        # Encoder: 64x64x3 -> 32x32x32 -> 16x16x64 -> 8x8x64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        # Decoder: 8x8x64 -> 16x16x64 -> 32x32x32 -> 64x64x3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perp, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perp


def train_vqvae(epochs=50, batch_size=128):
    print(f"Training VQ-VAE on {DEVICE}...")
    dataset = np.load("data/transitions_doom.npz")

    # We can use both frames and next_frames to get more data!
    frames1 = dataset["frames"]
    frames2 = dataset["next_frames"]
    all_frames = np.concatenate([frames1, frames2], axis=0)

    # Unique frames to avoid duplicates if necessary, but it's fine
    frames = torch.tensor(all_frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    dataset_size = len(frames)
    print(f"Total training frames: {dataset_size}")

    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_recon_loss = 0
        epoch_vq_loss = 0

        for i in range(0, dataset_size, batch_size):
            idx = perm[i : i + batch_size]
            b_frames = frames[idx].to(DEVICE)

            optimizer.zero_grad()
            x_recon, vq_loss, perp = model(b_frames)

            recon_loss = F.mse_loss(x_recon, b_frames)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item() * len(idx)
            epoch_vq_loss += vq_loss.item() * len(idx)

        print(
            f"Epoch {epoch + 1}/{epochs} - Recon Loss: {epoch_recon_loss / dataset_size:.5f} - VQ Loss: {epoch_vq_loss / dataset_size:.5f}"
        )

    os.makedirs("models", exist_ok=True)
    save_file(model.state_dict(), "models/vq_vae_doom.safetensors")
    print("VQ-VAE saved to models/vq_vae_doom.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train_vqvae(epochs=30)
