# Running Worldlet (Inference)

This guide covers how to run the Worldlet prototype for generating and navigating 3D environments from images.

## Basic Usage

Run the simple splatting world model. It will use a default seed image if you don't provide one.

```bash
python3 nano_world.py [path_to_image.png]
```

## Controls

Once the environment is running, use the following keys to move through the generated voxel world:

- **WASD** or **Arrow Keys**: Move/Rotate.
- **ESC**: Exit the application.

## AI Agent Auto-Play

If you have a trained PPO model (e.g., from the training flywheel), you can run it in auto-play mode:

```bash
python3 nano_world.py --agent ./models/flywheel/best_model.zip
```

## Benchmarking Models

To get a standardized performance report for any model, use the benchmarking suite:

```bash
# Benchmark a ViZDoom model
python3 benchmark.py ./models/ppo_doom_basic.zip --env doom --episodes 10

# Benchmark a native Worldlet model
python3 benchmark.py ./models/nano_agent_snapshot.zip --env nano --episodes 5
```

The benchmark provides average rewards, step counts, and processing throughput (FPS). You can also save results to JSON:

```bash
python3 benchmark.py ./models/flywheel/best_model.zip --output results/benchmark.json
```
