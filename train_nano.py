"""
This script trains a native Reinforcement Learning agent (PPO) 
on the NanoWorld environment. It is used as a baseline experiment 
to verify the environment mechanics before generative modeling.
"""

import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

try:
    from nano_world import NanoWorldEnv, DEVICE
except ImportError:
    print("Error: Could not import NanoWorldEnv from nano_world.py")
    sys.exit(1)


def check_tensorboard():
    """Check if TensorBoard is installed to avoid SB3 ImportError."""
    try:
        import tensorboard

        return "./models/tensorboard/"
    except ImportError:
        print("Warning: TensorBoard not installed. Logging will be disabled.")
        return None


def train(args):
    print(f"Setting up Nano World RL Environment on {DEVICE}...")
    env = NanoWorldEnv(render_mode="rgb_array")  # Headless mode for faster training

    try:
        check_env(env)
    except Exception as e:
        print(f"Warning: Gym check failed: {e}")

    print("Initializing PPO Model (CNN Policy)...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,  # Slightly faster learning rate
        n_steps=1024,
        batch_size=64,
        device=DEVICE,
        tensorboard_log=check_tensorboard(),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000, save_path="./models/logs/", name_prefix=args.model_name
    )

    print(f"Training native agent on NanoWorld for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current progress...")

    save_path = f"./models/{args.model_name}.zip"
    model.save(save_path)
    print(f"Native model saved to {save_path}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps", type=int, default=5000, help="Timesteps to train"
    )
    parser.add_argument(
        "--model-name", type=str, default="nano_agent_snapshot", help="Model name"
    )
    args = parser.parse_args()

    os.makedirs("./models", exist_ok=True)
    train(args)
