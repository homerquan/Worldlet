---
license: mit
language:
- en
tags:
- reinforcement-learning
- video-prediction
- generative-world-model
- vizdoom
- transformer
- vq-vae
---

# 🌍 Worldlet: DreamDoom Video Generative Model

This repository contains the dataset and trained weights for **DreamDoom**, an experimental prototype exploring **Generative Video Models**. 

Imagine a system that doesn't just play a game, but *understands* how to generate the visual output of the game frame-by-frame entirely via a neural network. This project provides a sequence-to-sequence video prediction model trained on ViZDoom, allowing you to "play" the hallucinated game interactively.

## 📁 Repository Contents

- **`data/`**: Contains the transition datasets collected from random agent gameplay across multiple ViZDoom scenarios.
  - `transitions_doom_diverse.npz`: Latest diverse dataset (1.1GB).
- **`models/`**: Contains the trained PyTorch weights (`.safetensors`):
  - `vq_vae_doom.safetensors`: The VQ-VAE tokenizer.
  - `transformer_world_doom.safetensors`: The Autoregressive Transformer predictor.

## ✨ Model Architecture

The repository provides weights for three different generative backend architectures:

1.  **DiT (Diffusion Transformer) [Default & Best Quality]**: A continuous diffusion-based world model. It takes continuous latents from the VQ-VAE and learns to iteratively denoise the next frame's latents.
2.  **Autoregressive Transformer (Discrete)**: A sequence-to-sequence discrete token model. It receives 64 discrete visual tokens and an action token, autoregressively predicting the next 64 tokens.
3.  **VideoPredict CNN (Direct Pixel)**: A simpler direct pixel-to-pixel generative CNN baseline.

All models utilize the underlying **VQ-VAE** which encodes 64x64 visual frames into an 8x8 grid of latent representations.

## 🎮 How to Use

You can use these files directly with the [Worldlet/DreamDoom GitHub repository](https://github.com/homerquan/Worldlet). 

Download the model and run the interactive hallucination:
```bash
python3 dream_doom.py
```

### Controls in DreamDoom
- **W / S** or **Up / Down Arrow**: Move Forward / Backward
- **A / D** or **Left / Right Arrow**: Move Left / Right
- **Spacebar**: Attack

---

*This model was trained as a concept demo for exploring interactive neural-rendered environments.*
