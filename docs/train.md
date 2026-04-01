# Training Worldlet (using RL)

This guide covers the reinforcement learning training pipeline for Worldlet, including manual training, the automated flywheel, and publishing models.

## RL Training CLI (`train_cli.py`)

Train a PPO model on the ViZDoom "basic" scenario (which involves moving left/right and shooting a monster).

### Training a New Model

To start a new training run:

```bash
python3 train_cli.py train --timesteps 50000 --model-name ppo_doom_basic
```

### Evaluating a Model

To evaluate a previously trained model with visual feedback:

```bash
python3 train_cli.py eval ./models/ppo_doom_basic.zip --episodes 10
```

### Benchmarking (Standardized)

For a more rigorous, non-visual performance check, use the benchmark tool:

```bash
python3 benchmark.py ./models/ppo_doom_basic.zip --episodes 20 --output models/logs/bench.json
```

---

## The Data Flywheel (`flywheel.py`)

The flywheel automates the training-evaluation-improvement loop. It sets a benchmark score, trains the model for a chunk of timesteps, re-evaluates, and repeats.

### Running the Flywheel

Start the flywheel, training in chunks of 20,000 timesteps:

```bash
# It will run indefinitely until interrupted (Ctrl+C).
python3 flywheel.py --train-steps 20000 --eval-episodes 5
```

### Advanced Flywheel Options

Start from an existing model and discard any updates that decrease the score:

```bash
python3 flywheel.py --initial-model ./models/ppo_doom_basic.zip --discard-failures
```

The best models will be automatically saved in `./models/flywheel/best_model.zip`.

## Publishing to HuggingFace

Once your flywheel has generated a strong model, you can publish it to the HuggingFace Hub using `huggingface_sb3`.

1. **Authenticate your CLI**:
   ```bash
   huggingface-cli login
   ```
2. **Follow the publishing instructions** in the main `README.md` for the script template.
