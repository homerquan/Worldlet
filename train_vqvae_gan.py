"""
This script trains a Vector Quantized Generative Adversarial Network 
(VQ-GAN) to improve the visual fidelity of the tokenized latents. 
It acts as an alternative/upgrade to the standard VQ-VAE experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from safetensors.torch import save_file
import lpips

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Keep the original architecture to preserve compatibility, but we will add the new training loss.
# Later we can upgrade to ResNet if needed, but let's first add perceptual loss and GAN discriminator for sharp images.
from vq_vae import VQVAE, VectorQuantizer

# Simple PatchGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        in_c = in_channels
        for feature in features:
            layers.append(
                nn.Conv2d(in_c, feature, kernel_size=4, stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = feature
        
        layers.append(nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_vqvae_gan(epochs=50, batch_size=128):
    print(f"Training VQ-VAE with Perceptual + GAN Loss on {DEVICE}...")
    dataset = np.load("data/transitions_doom.npz")

    frames1 = dataset["frames"]
    frames2 = dataset["next_frames"]
    all_frames = np.concatenate([frames1, frames2], axis=0)
    frames = torch.tensor(all_frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    dataset_size = len(frames)
    print(f"Total training frames: {dataset_size}")

    model = VQVAE().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    discriminator.apply(weights_init)

    # Losses
    perceptual_loss = lpips.LPIPS(net='vgg').to(DEVICE)
    bce_loss = nn.BCEWithLogitsLoss()

    opt_vq = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # Weight for different losses
    l1_weight = 1.0
    perceptual_weight = 1.0
    gan_weight = 0.1 # Start small for GAN loss
    disc_start_epoch = 0 # Start GAN loss from epoch 0

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        epoch_p_loss = 0
        epoch_g_loss = 0
        epoch_d_loss = 0

        for i in range(0, dataset_size, batch_size):
            idx = perm[i : i + batch_size]
            real_images = frames[idx].to(DEVICE)
            b_size = real_images.size(0)

            # ========================================
            # Train VQ-VAE (Generator)
            # ========================================
            opt_vq.zero_grad()
            fake_images, vq_loss, perp = model(real_images)

            # 1. Reconstruction Loss (L1)
            recon_loss = F.l1_loss(fake_images, real_images)
            
            # 2. Perceptual Loss (LPIPS)
            # LPIPS expects inputs in [-1, 1] range, our inputs are [0, 1]
            p_loss = perceptual_loss(fake_images * 2 - 1, real_images * 2 - 1).mean()

            # 3. GAN Loss (only after disc_start_epoch)
            if epoch >= disc_start_epoch:
                fake_preds = discriminator(fake_images)
                g_loss = bce_loss(fake_preds, torch.ones_like(fake_preds))
            else:
                g_loss = torch.tensor(0.0).to(DEVICE)

            # Total Generator Loss
            total_g_loss = (l1_weight * recon_loss) + \
                           (perceptual_weight * p_loss) + \
                           (gan_weight * g_loss) + \
                           vq_loss

            total_g_loss.backward()
            opt_vq.step()

            # ========================================
            # Train Discriminator
            # ========================================
            if epoch >= disc_start_epoch:
                opt_disc.zero_grad()
                
                # Real
                real_preds = discriminator(real_images)
                d_real_loss = bce_loss(real_preds, torch.ones_like(real_preds))
                
                # Fake (detach to avoid backprop through VQ-VAE)
                fake_preds_d = discriminator(fake_images.detach())
                d_fake_loss = bce_loss(fake_preds_d, torch.zeros_like(fake_preds_d))
                
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                d_loss.backward()
                opt_disc.step()
            else:
                d_loss = torch.tensor(0.0)

            # Logging
            epoch_recon_loss += recon_loss.item() * b_size
            epoch_vq_loss += vq_loss.item() * b_size
            epoch_p_loss += p_loss.item() * b_size
            epoch_g_loss += g_loss.item() * b_size
            epoch_d_loss += d_loss.item() * b_size

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"L1: {epoch_recon_loss / dataset_size:.4f} | "
            f"VQ: {epoch_vq_loss / dataset_size:.4f} | "
            f"LPIPS: {epoch_p_loss / dataset_size:.4f} | "
            f"G_GAN: {epoch_g_loss / dataset_size:.4f} | "
            f"D_loss: {epoch_d_loss / dataset_size:.4f}"
        )

    os.makedirs("models", exist_ok=True)
    save_file(model.state_dict(), "models/vq_vae_doom_gan.safetensors")
    print("VQ-VAE saved to models/vq_vae_doom_gan.safetensors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_vqvae_gan(epochs=args.epochs)
