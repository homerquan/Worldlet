import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from vq_vae import VQVAE
from safetensors.torch import load_file, save_file

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class TransformerWorldModel(nn.Module):
    """
    Autoregressive Transformer (GPT-style) that looks at a sequence of discrete
    tokens (Current Frame Tokens + Action Token) to predict Next Frame Tokens.
    """

    def __init__(
        self, vocab_size=512 + 5, d_model=512, nhead=8, num_layers=8, max_seq_len=129
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)

        emb = self.tok_emb(x) + self.pos_emb(positions)

        # Causal mask so it can't look into the future
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        out = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.head(out)
        return logits


def encode_dataset(batch_size=256):
    print(f"Loading VQ-VAE on {DEVICE} to encode dataset...")
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(
        load_file("models/vq_vae_doom.safetensors", device=str(DEVICE))
    )
    vqvae.eval()

    dataset = np.load("data/transitions_doom.npz")
    frames = (
        torch.tensor(dataset["frames"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    )
    actions = torch.tensor(dataset["actions"], dtype=torch.long)
    next_frames = (
        torch.tensor(dataset["next_frames"], dtype=torch.float32).permute(0, 3, 1, 2)
        / 255.0
    )

    encoded_frames = []
    encoded_next = []

    print(f"Encoding {len(frames)} frames into integer tokens...")
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            if i % 5000 == 0:
                print(f"Encoding {i}/{len(frames)}")
            f_batch = frames[i : i + batch_size].to(DEVICE)
            n_batch = next_frames[i : i + batch_size].to(DEVICE)

            # Frame -> Latents -> Tokens
            z_f = vqvae.encoder(f_batch)
            _, _, _, enc_f = vqvae.vq(z_f)
            encoded_frames.append(enc_f.cpu())

            z_n = vqvae.encoder(n_batch)
            _, _, _, enc_n = vqvae.vq(z_n)
            encoded_next.append(enc_n.cpu())

    encoded_frames = torch.cat(encoded_frames, dim=0).squeeze(-1).numpy()
    encoded_next = torch.cat(encoded_next, dim=0).squeeze(-1).numpy()

    # Save to disk so we don't have to re-encode every epoch
    os.makedirs("data", exist_ok=True)
    np.savez(
        "data/encoded_transitions_doom.npz",
        frames=encoded_frames,
        actions=actions.numpy(),
        next_frames=encoded_next,
    )
    print("Encoded dataset saved to data/encoded_transitions_doom.npz")


def train_transformer(epochs=40, batch_size=256):
    print(f"Training Autoregressive Transformer on {DEVICE}...")
    dataset = np.load("data/encoded_transitions_doom.npz")

    # 64 visual tokens for the past frame, reshaped
    frames = torch.tensor(dataset["frames"], dtype=torch.long).reshape(-1, 64)

    # 1 action token (shifted by +512 so it doesn't collide with visual tokens 0-511)
    actions = torch.tensor(dataset["actions"], dtype=torch.long).unsqueeze(1) + 512

    # 64 visual tokens for the next frame, reshaped
    next_frames = torch.tensor(dataset["next_frames"], dtype=torch.long).reshape(-1, 64)

    # Sequence = [Frame_t, Action_t, Frame_t+1]
    # Length = 64 + 1 + 64 = 129
    sequences = torch.cat([frames, actions, next_frames], dim=1)
    dataset_size = len(sequences)

    # Vocab size = 512 (Visual) + 5 (Actions)
    model = TransformerWorldModel(vocab_size=512 + 5, max_seq_len=129).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # ignore_index=-100 means we only train it to predict the NEXT frame,
    # not the past frame or action.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_loss = 0

        for i in range(0, dataset_size, batch_size):
            idx = perm[i : i + batch_size]
            seq_batch = sequences[idx].to(DEVICE)

            # Input: The first 128 tokens
            x = seq_batch[:, :-1]

            # Target: The next 128 tokens (shifted by 1)
            y = seq_batch[:, 1:].clone()

            # Mask out the past frame and action from the loss calculation (indices 0 to 64)
            # We only want the transformer to be penalized for mispredicting the NEXT frame
            y[:, :65] = -100

            optimizer.zero_grad()
            logits = model(x)  # (B, 128, Vocab)

            # Flatten to calculate loss
            loss = criterion(logits.reshape(-1, 512 + 5), y.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / dataset_size:.5f}")

    os.makedirs("models", exist_ok=True)
    save_file(model.state_dict(), "models/transformer_world_doom.safetensors")
    print("Transformer saved to models/transformer_world_doom.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encode", action="store_true", help="Pre-encode dataset with VQ-VAE"
    )
    parser.add_argument("--train", action="store_true", help="Train the Transformer")
    args = parser.parse_args()

    if args.encode:
        encode_dataset()
    if args.train:
        train_transformer(epochs=40)
