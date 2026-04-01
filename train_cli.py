import argparse
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import vizdoom as vzd
except ImportError:
    print("Error: vizdoom is not installed.")
    print("Please install it using: pip install vizdoom")
    sys.exit(1)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Hardware Device ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def check_tensorboard():
    """Check if TensorBoard is installed to avoid SB3 ImportError."""
    try:
        import tensorboard

        return "./models/tensorboard/"
    except ImportError:
        print("Warning: TensorBoard not installed. Logging will be disabled.")
        return None


class DoomEnv(gym.Env):
    """
    A simple wrapper for ViZDoom to make it compatible with Gymnasium
    and Stable-Baselines3.
    """

    def __init__(self, scenario_path, render=False):
        super(DoomEnv, self).__init__()

        self.game = vzd.DoomGame()

        if not os.path.exists(scenario_path):
            print(f"Error: Could not find scenario file at {scenario_path}")
            sys.exit(1)

        self.game.load_config(scenario_path)
        self.game.set_window_visible(render)

        # Set screen format to RGB and resolution to 160x120 for faster training
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)

        self.game.init()

        # The basic scenario has 3 actions: MOVE_LEFT, MOVE_RIGHT, ATTACK
        self.action_space = spaces.Discrete(3)
        self.actions = [
            [True, False, False],  # MOVE_LEFT
            [False, True, False],  # MOVE_RIGHT
            [False, False, True],  # ATTACK
        ]

        # Observation space is the RGB screen buffer (Height, Width, Channels)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(120, 160, 3), dtype=np.uint8
        )

    def step(self, action):
        # We skip 4 frames (frame_skip=4) for faster training
        reward = self.game.make_action(self.actions[action], 4)
        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            img = state.screen_buffer
        else:
            img = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return img, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        img = state.screen_buffer
        return img, {}

    def close(self):
        self.game.close()


def train(args):
    print(f"Starting Training on device: {DEVICE}")

    # Path to the scenario
    default_scenario = os.path.join(
        os.path.dirname(str(vzd.__file__)), "scenarios", args.scenario + ".cfg"
    )
    scenario_path = args.scenario_path if args.scenario_path else default_scenario

    env = DoomEnv(scenario_path, render=False)

    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path="./models/logs/", name_prefix=args.model_name
    )

    print(f"Initializing PPO Model (CNN Policy)...")

    if args.resume and os.path.exists(args.resume):
        print(f"Loading existing model from {args.resume}")
        model = PPO.load(args.resume, env=env, device=DEVICE)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            device=DEVICE,
            tensorboard_log=check_tensorboard(),
        )

    print(f"Training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            tb_log_name=args.model_name,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")

    save_path = f"./models/{args.model_name}.zip"
    model.save(save_path)
    print(f"Model saved to {save_path}")
    env.close()


def evaluate(args):
    print(f"Starting Evaluation for {args.episodes} episodes...")

    default_scenario = os.path.join(
        os.path.dirname(str(vzd.__file__)), "scenarios", args.scenario + ".cfg"
    )
    scenario_path = args.scenario_path if args.scenario_path else default_scenario

    env = DoomEnv(scenario_path, render=args.render)

    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path, device=DEVICE)

    total_rewards = []

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Prevent infinite loops in evaluation
            if step_count > 1000:
                print("Warning: Episode terminated early (hit 1000 steps)")
                break

        print(f"Episode {episode + 1} finished with Total Reward: {total_reward}")
        total_rewards.append(total_reward)

    print(
        f"\nAverage Reward over {args.episodes} episodes: {np.mean(total_rewards):.2f}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or Evaluate a PPO agent in ViZDoom"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train Command ---
    parser_train = subparsers.add_parser("train", help="Train a new model")
    parser_train.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train (default: 100000)",
    )
    parser_train.add_argument(
        "--model-name", type=str, default="ppo_doom", help="Name to save the model as"
    )
    parser_train.add_argument(
        "--scenario",
        type=str,
        default="basic",
        help="Name of default ViZDoom scenario (default: basic)",
    )
    parser_train.add_argument(
        "--scenario-path",
        type=str,
        default=None,
        help="Custom path to a .cfg scenario file",
    )
    parser_train.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001)"
    )
    parser_train.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing .zip model to resume training",
    )

    # --- Eval Command ---
    parser_eval = subparsers.add_parser("eval", help="Evaluate an existing model")
    parser_eval.add_argument(
        "model_path", type=str, help="Path to the .zip model to evaluate"
    )
    parser_eval.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate (default: 5)",
    )
    parser_eval.add_argument(
        "--scenario",
        type=str,
        default="basic",
        help="Name of default ViZDoom scenario (default: basic)",
    )
    parser_eval.add_argument(
        "--scenario-path",
        type=str,
        default=None,
        help="Custom path to a .cfg scenario file",
    )
    parser_eval.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable rendering the game window",
    )
    parser_eval.set_defaults(render=True)

    args = parser.parse_args()

    os.makedirs("./models", exist_ok=True)

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()
