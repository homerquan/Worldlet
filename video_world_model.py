import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from safetensors.torch import save_file

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

RES = 64  # Tiny video prediction resolution


class VideoWorldModel(nn.Module):
    """
    Predicts the next frame given the current frame and an action.
    This acts as a pixel-to-pixel video generative predictive model.
    """

    def __init__(self, num_actions=5):
        super().__init__()
        self.num_actions = num_actions

        # Encoder (downsamples 64x64 to 8x8)
        self.enc1 = nn.Conv2d(
            3 + num_actions, 32, kernel_size=4, stride=2, padding=1
        )  # 32x32
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8

        # Decoder (upsamples 8x8 back to 64x64)
        self.dec1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 16x16
        self.dec2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # 32x32
        self.dec3 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # 64x64

    def forward(self, x, action):
        # x: (B, 3, H, W)
        B, _, H, W = x.shape

        # Spatially broadcast the action as one-hot channels
        act_one_hot = F.one_hot(action, num_classes=self.num_actions).float()  # (B, A)
        act_spatial = act_one_hot.view(B, self.num_actions, 1, 1).expand(
            B, self.num_actions, H, W
        )

        # Concat image and actions
        h = torch.cat([x, act_spatial], dim=1)

        # Encode
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))

        # Decode
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        out = torch.sigmoid(self.dec3(h))  # Pixels are 0-1

        return out


def collect_data(num_samples=10000):
    print(
        f"Collecting {num_samples} transition samples from the doom environment locally..."
    )
    from rl_doom import DoomEnv

    scenarios = [
        "basic",
        "deadly_corridor",
        "defend_the_center",
        "health_gathering",
        "my_way_home",
        "predict_position",
        "take_cover",
        "defend_the_line",
    ]
    samples_per_scenario = num_samples // len(scenarios)

    frames = []
    actions = []
    next_frames = []

    for scenario in scenarios:
        print(f"Starting scenario: {scenario}")
        env = DoomEnv(render=False, scenario=scenario)
        obs, _ = env.reset()
        for i in range(samples_per_scenario):
            if i % 1000 == 0:
                print(f"Sample {i}/{samples_per_scenario} (Scenario: {scenario})")
            action = env.action_space.sample()
            next_obs, reward, done, trunc, _ = env.step(action)

            f1 = cv2.resize(obs, (RES, RES))
            f2 = cv2.resize(next_obs, (RES, RES))

            frames.append(f1)
            actions.append(action)
            next_frames.append(f2)

            obs = next_obs
            if done or trunc:
                obs, _ = env.reset()
        env.close()

    os.makedirs("data", exist_ok=True)
    filename = "data/transitions_doom.npz"
    np.savez(
        filename,
        frames=np.array(frames),
        actions=np.array(actions),
        next_frames=np.array(next_frames),
    )
    print(f"Dataset saved to {filename}")


def train_model(epochs=100, batch_size=64):
    print(f"Training Video Generative Model on {DEVICE} for doom...")
    dataset = np.load("data/transitions_doom.npz")

    # Shape is (N, 64, 64, 3) initially, change to (N, 3, 64, 64) for PyTorch
    frames = (
        torch.tensor(dataset["frames"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    )
    actions = torch.tensor(dataset["actions"], dtype=torch.long)
    next_frames = (
        torch.tensor(dataset["next_frames"], dtype=torch.float32).permute(0, 3, 1, 2)
        / 255.0
    )

    dataset_size = len(frames)
    model = VideoWorldModel(num_actions=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_loss = 0

        for i in range(0, dataset_size, batch_size):
            idx = perm[i : i + batch_size]
            b_frames = frames[idx].to(DEVICE)
            b_actions = actions[idx].to(DEVICE)
            b_next = next_frames[idx].to(DEVICE)

            pred = model(b_frames, b_actions)
            loss = F.mse_loss(pred, b_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / dataset_size:.5f}")

    model_name = "video_predict_model_doom.safetensors"
    save_file(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Collect data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument(
        "--samples", type=int, default=40000, help="Number of samples to collect"
    )
    args = parser.parse_args()

    if args.collect:
        collect_data(num_samples=args.samples)
    if args.train:
        train_model(epochs=150)
