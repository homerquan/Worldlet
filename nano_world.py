import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import math
import os
import cv2
import gymnasium as gym
from gymnasium import spaces

# --- Settings ---
WIDTH, HEIGHT = 800, 600
OBS_WIDTH, OBS_HEIGHT = 160, 120
RESOLUTION = 64
SPLAT_SIZE_BASE = 15.0
FOV = 400.0

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# --- AI Generative World Model (Neural Field) ---
class NanoWorldGenAI(nn.Module):
    """
    A tiny Neural Radiance Field (NeRF) / MLP.
    Maps (X, Z) coordinates -> (Y-Height, R, G, B) to generate infinite terrain.
    """

    def __init__(self):
        super().__init__()
        # Input: 2 (x,z) + 4 (sin/cos encoding) = 6
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # Output: Y, R, G, B
        )

    def forward(self, xz):
        # High-frequency positional encoding for spatial awareness
        enc = torch.cat([xz, torch.sin(xz * 0.1), torch.cos(xz * 0.1)], dim=-1)
        return self.net(enc)


def create_default_image(filename):
    img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            if y < RESOLUTION // 2:
                img[y, x] = [135, max(0, 206 - y * 2), 235]
                if abs(x - RESOLUTION // 2) < (y - 10) * 1.5 and y > 10:
                    img[y, x] = [120, 120, 130]
            else:
                noise = np.random.randint(-10, 10)
                g = max(0, min(255, 180 - (y - RESOLUTION // 2) * 3 + noise))
                img[y, x] = [34 + noise, g, 34 + noise]
    Image.fromarray(img).save(filename)
    return img


def load_or_generate_image(path):
    if path and os.path.exists(path):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
            return np.array(img)
        except Exception:
            pass
    return create_default_image("default_world_seed.png")


def init_world(img):
    points, colors = [], []
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            color = img[y, x]
            world_x = (x - RESOLUTION / 2) * 2.0
            if y < RESOLUTION // 2:
                world_y = (RESOLUTION // 2 - y) * 2.0
                world_z = 200.0 + np.random.uniform(-10, 10)
            else:
                world_y = -10.0 + np.random.uniform(-0.5, 0.5)
                depth_factor = (RESOLUTION - y) / (RESOLUTION / 2)
                world_z = depth_factor * 150.0
            points.append([world_x, world_y, world_z])
            colors.append(color)

    points_t = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    colors_t = torch.tensor(colors, dtype=torch.uint8, device=DEVICE)

    # --- Train the AI Generative Model on the Seed Image ---
    print("\n[AI World] Training Nano Neural Field on Seed Image...")
    ai_model = NanoWorldGenAI().to(DEVICE)
    optimizer = torch.optim.Adam(ai_model.parameters(), lr=0.01)

    # Inputs: (X, Z). Targets: (Y, R/255, G/255, B/255)
    xz_in = points_t[:, [0, 2]]
    y_out = points_t[:, 1:2]
    c_out = colors_t.float() / 255.0
    targets = torch.cat([y_out, c_out], dim=1)

    # Fast overfit (hallucination seed)
    for epoch in range(150):
        pred = ai_model(xz_in)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[AI World] Training Complete. Initial Loss: {loss.item():.4f}")
    return points_t, colors_t, ai_model


def generate_new_chunk(z_offset, ai_model):
    """Uses the trained Neural Network to predict new unseen terrain!"""
    num_splats = 400
    xs = np.random.uniform(-RESOLUTION * 1.5, RESOLUTION * 1.5, num_splats)
    zs = z_offset + np.random.uniform(0, 40, num_splats)

    # Ask the Neural Net what is at these coordinates
    xz_new = torch.tensor(np.column_stack((xs, zs)), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        preds = ai_model(xz_new)

    y_new = preds[:, 0].cpu().numpy()

    # Denormalize colors back to 0-255
    c_new = (preds[:, 1:4] * 255.0).clamp(0, 255).byte()

    new_points = np.column_stack((xs, y_new, zs))
    return torch.tensor(new_points, dtype=torch.float32, device=DEVICE), c_new


class NanoWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, seed_path=None, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.seed_path = seed_path
        self.seed_img = load_or_generate_image(seed_path)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8
        )

        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Nano World Model (AI Generated Neural Field)")
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # We now get the AI model back from the init step
        self.points, self.colors, self.ai_model = init_world(self.seed_img)
        self.cam_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.yaw = 0.0
        self.steps = 0
        self.max_z_reached = 0.0

        self._render_frame()
        return self._get_obs(), {}

    def step(self, action):
        speed = 5.0
        rot_speed = 0.1

        if action == 0:  # Forward
            self.cam_pos[2] += speed * math.cos(self.yaw)
            self.cam_pos[0] += speed * math.sin(self.yaw)
        elif action == 1:  # Left
            self.yaw -= rot_speed
        elif action == 2:  # Right
            self.yaw += rot_speed

        max_z = self.points[:, 2].max().item()
        cam_z = self.cam_pos[2].item()

        if max_z - cam_z < 150:
            # Query the Generative Neural Network for the new chunk!
            new_p, new_c = generate_new_chunk(max_z, self.ai_model)
            self.points = torch.cat([self.points, new_p], dim=0)
            self.colors = torch.cat([self.colors, new_c], dim=0)

            mask_keep = self.points[:, 2] > cam_z - 50
            self.points = self.points[mask_keep]
            self.colors = self.colors[mask_keep]

        reward = 0.0
        if cam_z > self.max_z_reached:
            reward = (cam_z - self.max_z_reached) * 0.1
            self.max_z_reached = cam_z
        else:
            reward = -0.01

        self.steps += 1
        done = False
        truncated = self.steps >= 500

        self._render_frame()
        return self._get_obs(), reward, done, truncated, {}

    def _render_frame(self):
        self.screen.fill((135, 206, 235))

        with torch.no_grad():
            translated = self.points - self.cam_pos
            cos_y = math.cos(-self.yaw)
            sin_y = math.sin(-self.yaw)
            R_y = torch.tensor(
                [[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]],
                dtype=torch.float32,
                device=DEVICE,
            )

            rotated = torch.matmul(translated, R_y)
            rot_x, rot_y, rot_z = rotated[:, 0], rotated[:, 1], rotated[:, 2]

            mask = rot_z > 1.0
            p_x, p_y, p_z, p_c = (
                rot_x[mask],
                rot_y[mask],
                rot_z[mask],
                self.colors[mask],
            )

            if len(p_z) > 0:
                screen_x = (p_x * FOV / p_z) + WIDTH / 2
                screen_y = -(p_y * FOV / p_z) + HEIGHT / 2

                sort_idx = torch.argsort(p_z, descending=True)
                screen_x, screen_y = screen_x[sort_idx], screen_y[sort_idx]
                p_z, p_c = p_z[sort_idx], p_c[sort_idx]

                sizes = torch.clamp((SPLAT_SIZE_BASE * FOV) / p_z, min=2.0)

                screen_x_cpu = screen_x.cpu().numpy()
                screen_y_cpu = screen_y.cpu().numpy()
                sizes_cpu = sizes.cpu().numpy()
                colors_cpu = p_c.cpu().numpy()

                for i in range(len(screen_x_cpu)):
                    sz = int(sizes_cpu[i])
                    sx = int(screen_x_cpu[i] - sz / 2)
                    sy = int(screen_y_cpu[i] - sz / 2)
                    color = (colors_cpu[i][0], colors_cpu[i][1], colors_cpu[i][2])
                    rect = pygame.Rect(sx, sy, sz, sz)
                    pygame.draw.rect(self.screen, color, rect)

        if self.render_mode == "human":
            instructions = [
                f"Nano World AI Neural Field - Device: {DEVICE.type.upper()}",
                f"Active Voxels: {len(self.points)}",
                f"Pos: X:{int(self.cam_pos[0].item())} Z:{int(self.cam_pos[2].item())}",
                f"Steps: {self.steps}/500",
            ]
            for i, text in enumerate(instructions):
                img = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(
                    self.font.render(text, True, (0, 0, 0)), (11, 11 + i * 25)
                )
                self.screen.blit(img, (10, 10 + i * 25))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _get_obs(self):
        arr = pygame.surfarray.array3d(self.screen)
        arr = np.transpose(arr, (1, 0, 2))
        resized = cv2.resize(arr, (OBS_WIDTH, OBS_HEIGHT))
        return resized

    def close(self):
        pygame.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None, help="Path to seed image")
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Path to a trained PPO model (.zip) to auto-play",
    )
    args = parser.parse_args()

    model = None
    if args.agent:
        from stable_baselines3 import PPO

        if os.path.exists(args.agent):
            print(f"Loading RL Agent from {args.agent}...")
            model = PPO.load(args.agent, device=DEVICE)
        else:
            print(f"Agent model {args.agent} not found!")
            sys.exit(1)

    env = NanoWorldEnv(seed_path=args.image, render_mode="human")
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if model is not None:
            # AI CONTROL
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
        else:
            # MANUAL CONTROL
            action = -1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                action = 0
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action = 1
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action = 2
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                env.cam_pos[2] -= 5.0 * math.cos(env.yaw)
                env.cam_pos[0] -= 5.0 * math.sin(env.yaw)

            if action != -1:
                obs, _, _, _, _ = env.step(action)
            else:
                env._render_frame()

    env.close()


if __name__ == "__main__":
    main()
