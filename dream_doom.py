import os
import cv2
import pygame
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import the tokenizer and the three different world models
from vq_vae import VQVAE
from transformer_world import TransformerWorldModel
from dit_world import DiTWorldModel, DDPMSampler
from video_world_model import VideoWorldModel
from safetensors.torch import load_file

RES = 64

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class DreamDoomEnv(gym.Env):
    """
    A Gym Environment entirely driven by the Video Generative Predictive Model!
    Supports multiple generative backends (Transformer, DiT, CNN).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_type="dit",
        vqvae_path="models/vq_vae_doom.safetensors",
        transformer_path="models/transformer_world_doom.safetensors",
        dit_path="models/dit_world_doom.safetensors",
        cnn_path="models/video_predict_model_doom.safetensors",
        render_mode="human",
        scenario="basic",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.scenario = scenario
        self.model_type = model_type

        # 5 actions (standard mapping)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(RES, RES, 3), dtype=np.uint8
        )

        # Load models based on selection
        if self.model_type in ["transformer", "dit"]:
            self.vqvae = VQVAE().to(DEVICE)
            if os.path.exists(vqvae_path):
                print(f"  \033[32m✔\033[0m Loaded VQ-VAE weights from {vqvae_path}")
                self.vqvae.load_state_dict(load_file(vqvae_path, device=str(DEVICE)))
            else:
                print(f"  \033[33m⚠\033[0m WARNING: Could not find {vqvae_path}!")
            self.vqvae.eval()

        if self.model_type == "transformer":
            self.model = TransformerWorldModel(
                vocab_size=512 + 5, d_model=512, nhead=8, num_layers=8, max_seq_len=129
            ).to(DEVICE)
            if os.path.exists(transformer_path):
                print(
                    f"  \033[32m✔\033[0m Loaded Transformer weights from {transformer_path}"
                )
                self.model.load_state_dict(
                    load_file(transformer_path, device=str(DEVICE))
                )
            else:
                print(f"  \033[33m⚠\033[0m WARNING: Could not find {transformer_path}!")
            self.model.eval()

        elif self.model_type == "dit":
            self.model = DiTWorldModel(
                channels=64,
                latent_size=8,
                num_actions=5,
                d_model=512,
                nhead=8,
                num_layers=8,
            ).to(DEVICE)
            if os.path.exists(dit_path):
                print(f"  \033[32m✔\033[0m Loaded DiT weights from {dit_path}")
                self.model.load_state_dict(load_file(dit_path, device=str(DEVICE)))
            else:
                print(f"  \033[33m⚠\033[0m WARNING: Could not find {dit_path}!")
            self.model.eval()
            self.sampler = DDPMSampler(num_timesteps=1000, device=str(DEVICE))

        elif self.model_type == "cnn":
            self.model = VideoWorldModel(num_actions=5).to(DEVICE)
            if os.path.exists(cnn_path):
                print(f"  \033[32m✔\033[0m Loaded CNN weights from {cnn_path}")
                self.model.load_state_dict(load_file(cnn_path, device=str(DEVICE)))
            else:
                print(f"  \033[33m⚠\033[0m WARNING: Could not find {cnn_path}!")
            self.model.eval()

        self.WIDTH, self.HEIGHT = 640, 480  # Doom 4:3 aspect ratio
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 28)
        self.steps = 0
        self.current_frame = None
        self.screen = None
        self.window_open = False

    def open_window(self):
        if self.render_mode == "human" and not self.window_open:
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption(
                f"DreamDoom ({self.model_type.upper()}) - {self.scenario}"
            )
            self.window_open = True
        elif self.render_mode != "human" and self.screen is None:
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))

    def close_window(self):
        if self.window_open:
            pygame.display.quit()
            self.window_open = False

    def _get_initial_frame(self):
        """Use the actual Doom engine for just 1 frame to seed the hallucination"""
        try:
            from rl_doom import DoomEnv

            os.environ["SDL_VIDEODRIVER"] = "dummy"
            true_env = DoomEnv(render=False, scenario=self.scenario)
            obs, _ = true_env.reset()
            true_env.close()
            return cv2.resize(obs, (RES, RES))
        except Exception as e:
            print(f"Could not get real Doom frame: {e}")
            return np.zeros((RES, RES, 3), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if options and "scenario" in options:
            self.scenario = options["scenario"]

        self.open_window()
        self.steps = 0
        self.current_frame = self._get_initial_frame()
        self._render_frame()
        return self.current_frame, {}

    def step(self, action):
        self.steps += 1

        frame_t = (
            torch.tensor(self.current_frame, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        ).to(DEVICE)

        with torch.no_grad():
            next_frame_t = frame_t  # fallback
            if self.model_type == "transformer":
                # Encode to discrete tokens
                z = self.vqvae.encoder(frame_t)
                _, _, _, enc_indices = self.vqvae.vq(z)
                current_tokens = enc_indices.view(1, 64)

                action_token = torch.tensor(
                    [[action + 512]], dtype=torch.long, device=DEVICE
                )
                context = torch.cat([current_tokens, action_token], dim=1)

                for _ in range(64):
                    logits = self.model(context)
                    next_token_logits = logits[:, -1, :512]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    context = torch.cat([context, next_token], dim=1)

                next_frame_tokens = context[:, 65:]
                flat_tokens = next_frame_tokens.view(-1, 1)
                encodings = torch.zeros(
                    flat_tokens.shape[0], self.vqvae.vq._num_embeddings, device=DEVICE
                )
                encodings.scatter_(1, flat_tokens, 1)
                quantized = torch.matmul(encodings, self.vqvae.vq._embedding.weight)
                quantized = quantized.view(1, 8, 8, 64).permute(0, 3, 1, 2).contiguous()
                next_frame_t = self.vqvae.decoder(quantized)

            elif self.model_type == "dit":
                # Encode to continuous latents
                z = self.vqvae.encoder(frame_t)
                z_cond, _, _, _ = self.vqvae.vq(z)

                action_tensor = torch.tensor([action], dtype=torch.long, device=DEVICE)

                # Sample with DDIM (10 steps)
                x_0 = self.sampler.sample(
                    self.model, z_cond, action_tensor, shape=(1, 64, 8, 8), steps=10
                )
                x_0_quantized, _, _, _ = self.vqvae.vq(x_0)
                next_frame_t = self.vqvae.decoder(x_0_quantized)

            elif self.model_type == "cnn":
                # Direct frame prediction
                action_tensor = torch.tensor([action], dtype=torch.long, device=DEVICE)
                next_frame_t = self.model(frame_t, action_tensor)
                # Clamp to prevent artifacts
                next_frame_t = torch.clamp(next_frame_t, 0.0, 1.0)

        next_frame = (
            next_frame_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)
        self.current_frame = next_frame

        reward = 0.0
        done = False
        truncated = self.steps >= 1000

        if self.render_mode == "human":
            self._render_frame()

        return self.current_frame, reward, done, truncated, {}

    def _render_frame(self):
        if self.current_frame is None or self.screen is None:
            return

        surface = pygame.surfarray.make_surface(
            np.transpose(self.current_frame, (1, 0, 2))
        )
        surface = pygame.transform.scale(surface, (self.WIDTH, self.HEIGHT))
        self.screen.blit(surface, (0, 0))

        if self.render_mode == "human":
            model_names = {
                "transformer": "Autoregressive Transformer",
                "dit": "Diffusion Transformer (DiT)",
                "cnn": "VideoPredict CNN",
            }
            instructions = [
                f"DreamDoom ({model_names[self.model_type]})",
                f"Resolution: {RES}x{RES} (Upscaled)",
                f"Steps: {self.steps}/1000",
                "Controls: W/S (Fwd/Bck), A/D (Left/Right), SPACE (Attack)",
            ]
            for i, text in enumerate(instructions):
                img = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(
                    self.font.render(text, True, (0, 0, 0)), (11, 11 + i * 25)
                )
                self.screen.blit(img, (10, 10 + i * 25))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    import time
    import sys

    def print_welcome():
        os.system("cls" if os.name == "nt" else "clear")
        print("\033[1;35m" + "=" * 50)
        print("🌌  W E L C O M E   T O   D R E A M D O O M  🌌")
        print("=" * 50 + "\033[0m\n")
        print("A generative world model predicting ViZDoom on-the-fly.")
        print("No rendering engine. Just raw neural hallucination.\n")

    MODELS = {
        "1": ("💨", "dit", "Diffusion Transformer (Continuous - Best Quality)"),
        "2": ("🧠", "transformer", "Autoregressive Transformer (Discrete)"),
        "3": ("👁️", "cnn", "VideoPredict CNN (Direct Pixel)"),
    }

    SCENES = {
        "1": ("🧟", "basic"),
        "2": ("🚪", "deadly_corridor"),
        "3": ("🎯", "defend_the_center"),
        "4": ("🛡️", "defend_the_line"),
        "5": ("🩸", "health_gathering"),
    }

    def print_model_menu():
        print("\033[1;36mSelect Architecture:\033[0m")
        for key, (emoji, name, desc) in MODELS.items():
            print(f"  \033[1m{key}.\033[0m {emoji} {desc}")
        print("  \033[1mq.\033[0m ❌ Quit\n")

    def print_scene_menu():
        print("\033[1;36mAvailable Scenes:\033[0m")
        for key, (emoji, name) in SCENES.items():
            print(f"  \033[1m{key}.\033[0m {emoji} {name}")
        print("  \033[1mb.\033[0m 🔙 Back to Models\n")

    def animated_loading(text):
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        sys.stdout.write(f"\033[36m{text}\033[0m ")
        sys.stdout.flush()
        for i in range(15):
            sys.stdout.write(f"\b{chars[i % len(chars)]}")
            sys.stdout.flush()
            time.sleep(0.05)
        sys.stdout.write("\b\033[32m✔\033[0m\n")

    print_welcome()

    while True:
        print_model_menu()
        m_choice = (
            input("\033[1;33mSelect a model (1-3) [Default: 1] or 'q':\033[0m ")
            .strip()
            .lower()
        )
        if m_choice == "q":
            print("Shutting down... Goodbye! 👋")
            break

        if m_choice not in MODELS:
            m_choice = "1"  # default to DiT

        model_key, model_id, model_desc = MODELS[m_choice]
        print(f"\nLoading \033[1;35m{model_desc}\033[0m ...")
        env = DreamDoomEnv(render_mode="human", model_type=model_id)
        print("\033[32m✨ Model Loaded!\033[0m\n")

        while True:
            print_scene_menu()
            s_choice = (
                input("\033[1;33mSelect a scene (1-5) or 'b':\033[0m ").strip().lower()
            )

            if s_choice == "b":
                env.close()
                print("\n")
                break

            if not s_choice:
                s_choice = "1"

            if s_choice in SCENES:
                scene_name = SCENES[s_choice][1]
            else:
                import vizdoom as vzd

                scenario_path = os.path.join(
                    os.path.dirname(str(vzd.__file__)), "scenarios", f"{s_choice}.cfg"
                )
                if os.path.exists(scenario_path):
                    scene_name = s_choice
                else:
                    print(
                        f"\033[31m⚠ Scene '{s_choice}' not found in ViZDoom. Try again.\033[0m\n"
                    )
                    continue

            emoji_list = [v[0] for v in SCENES.values() if v[1] == scene_name]
            emoji = emoji_list[0] if emoji_list else "🕹️"

            animated_loading(f"Generating starting frame for {emoji} {scene_name}...")

            obs, _ = env.reset(options={"scenario": scene_name})
            print(
                f"\n\033[1;32m▶ Playing '{scene_name}' on {model_desc}! Close window to return.\033[0m\n"
            )

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                action = -1
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    action = 0
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    action = 1
                elif keys[pygame.K_SPACE]:
                    action = 2
                elif keys[pygame.K_UP] or keys[pygame.K_w]:
                    action = 3
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    action = 4

                if action != -1:
                    obs, _, _, _, _ = env.step(action)
                else:
                    env._render_frame()

            env.close_window()
