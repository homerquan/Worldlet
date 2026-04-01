import os
import sys
import argparse
import subprocess
import time
import numpy as np
import gymnasium as gym

try:
    from train_cli import DoomEnv, DEVICE
    import vizdoom as vzd
    from stable_baselines3 import PPO
except ImportError:
    print(
        "Please install requirements and ensure train_cli.py is in the same directory."
    )
    sys.exit(1)


def evaluate_model(model_path, scenario="basic", episodes=10):
    """
    Benchmarks the model on a specific scenario and returns the average reward.
    This acts as our evaluation step in the data flywheel.
    """
    print(f"\n--- Benchmarking {model_path} ---")
    default_scenario = os.path.join(
        os.path.dirname(str(vzd.__file__)), "scenarios", scenario + ".cfg"
    )

    env = DoomEnv(default_scenario, render=False)

    try:
        model = PPO.load(model_path, device=DEVICE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return float("-inf")

    total_rewards = []
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        reward_sum = 0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            reward_sum += reward
            step_count += 1
            if step_count > 500:  # Timeout
                break
        total_rewards.append(reward_sum)

    avg_reward = np.mean(total_rewards)
    print(f"Benchmark Score: {avg_reward:.2f} (Average over {episodes} episodes)")
    env.close()
    return avg_reward


def flywheel(args):
    """
    Data Flywheel loop:
    1. Train a model for N steps.
    2. Evaluate it (Benchmark).
    3. If it improves, save it as the new "best" model.
    4. Repeat, using the new best model as a starting point.
    """
    current_model_path = args.initial_model
    best_score = float("-inf")

    if current_model_path and os.path.exists(current_model_path):
        print(f"Starting flywheel with existing model: {current_model_path}")
        best_score = evaluate_model(
            current_model_path, args.scenario, args.eval_episodes
        )
    else:
        print("Starting flywheel from scratch.")
        current_model_path = None

    os.makedirs("./models/flywheel", exist_ok=True)

    iteration = 1
    while True:
        if args.max_iterations and iteration > args.max_iterations:
            print(f"Reached max iterations ({args.max_iterations}). Stopping flywheel.")
            break

        print(f"\n========== Flywheel Iteration {iteration} ==========")
        new_model_name = f"flywheel_model_iter_{iteration}"
        new_model_path = f"./models/{new_model_name}.zip"

        # 1. Train
        train_cmd = [
            sys.executable,
            "train_cli.py",
            "train",
            "--timesteps",
            str(args.train_steps),
            "--model-name",
            new_model_name,
            "--scenario",
            args.scenario,
        ]

        if current_model_path:
            train_cmd.extend(["--resume", current_model_path])

        print(f"Training: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)

        # 2. Benchmark/Evaluate
        score = evaluate_model(new_model_path, args.scenario, args.eval_episodes)

        # 3. Improvement Check
        if score > best_score:
            print(
                f"*** New Best Model Found! Score: {score:.2f} (Previous best: {best_score:.2f}) ***"
            )
            best_score = score
            current_model_path = new_model_path

            # Keep a copy of the very best model in a specific location
            subprocess.run(["cp", new_model_path, "./models/flywheel/best_model.zip"])
            print("Saved as ./models/flywheel/best_model.zip")
        else:
            print(
                f"Model did not improve (Score: {score:.2f} vs Best: {best_score:.2f})."
            )
            if args.discard_failures:
                print(
                    "Discarding failed model. Will resume from previous best next iteration."
                )
            else:
                print("Keeping model anyway to explore new states (exploration).")
                current_model_path = new_model_path

        iteration += 1
        time.sleep(1)  # Brief pause


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Data Flywheel for RL Model Improvement"
    )
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Path to initial model to start from",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=20000,
        help="Timesteps to train per iteration",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Episodes to evaluate during benchmark",
    )
    parser.add_argument(
        "--scenario", type=str, default="basic", help="ViZDoom scenario to use"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations to run (default: infinite)",
    )
    parser.add_argument(
        "--discard-failures",
        action="store_true",
        help="If set, discards models that score worse than previous best",
    )

    args = parser.parse_args()
    flywheel(args)
