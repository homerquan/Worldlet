# AI Agent Instructions for Worldlet / DreamDoom

Welcome to the Worldlet repository. This document provides core instructions, conventions, and operational guidelines for autonomous AI agents (like yourself) working within this codebase. Please review these rules carefully before reading or modifying code.

## 1. Project Overview

**DreamDoom** is an experimental project exploring Generative Video Models. It trains neural networks to predict and hallucinate the visual output of a game (ViZDoom) frame-by-frame based on user actions. 

Instead of traditional rendering, the game environment is entirely simulated by an AI model evaluating state transitions. The latest architecture utilizes a **VQ-VAE** for discrete tokenization of visual frames and an **Autoregressive Transformer** to predict the next sequence of visual tokens based on the current frame tokens and an action token.

**Core Technologies**:
- **Python 3.9+**
- **PyTorch** (Deep Learning, Neural Networks)
- **Gymnasium** (Environment interfaces)
- **PyGame / OpenCV** (Rendering and image processing)
- **ViZDoom** (Underlying data generator)

---

## 2. Build, Lint, and Test Commands

When making changes, always ensure you don't break the environment or introduce syntax/linting errors. You are expected to run verification commands autonomously after editing code.

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Linting & Formatting
The project uses `ruff` for fast linting and formatting. Always run these checks after modifying Python files to ensure standard formatting is maintained:
```bash
# Check for linting errors
ruff check .

# Automatically fix fixable linting errors
ruff check . --fix

# Format code (Black-compatible standard)
ruff format .
```

### Testing
There is currently no formal test suite (e.g., `pytest` is not in `requirements.txt`), but if you add tests or need to verify a script, use standard Python testing conventions. If adding tests:
- Place them in a `tests/` directory at the root.
- Ensure test files are named `test_*.py`.
- Run all tests: `pytest`
- Run a single test: `pytest tests/test_model.py::test_specific_function`

### Running the Application
To verify that the main scripts still execute without syntax or runtime errors:
```bash
# Collect a small amount of data (fast sanity check)
python3 video_world_model.py --collect --samples 100

# Train the continuous model for 1 epoch (fast sanity check)
python3 video_world_model.py --train --epochs 1

# Run the interactive environment (requires a display/GUI)
python3 dream_doom.py
```

---

## 3. Code Style and Conventions

Rigorously adhere to the following style guidelines. Mimic the existing codebase's structure and formatting.

### 3.1. Formatting
- Use **Black's standard formatting rules** (enforced via `ruff format`).
- Maximum line length is **88 characters**.
- Use 4 spaces for indentation (no tabs).
- Ensure files end with a single newline character.

### 3.2. Naming Conventions
- **Classes**: `PascalCase` (e.g., `VideoWorldModel`, `TransformerWorldModel`).
- **Functions & Variables**: `snake_case` (e.g., `train_model`, `next_frame`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEVICE`, `RES`).
- **Private Methods/Variables**: Prefix with a single underscore (e.g., `_render_frame`, `_get_initial_frame`).

### 3.3. Imports
Group imports cleanly at the top of the file in this exact order, separated by a blank line:
1. Standard library imports (`os`, `argparse`)
2. Third-party imports (`torch`, `numpy`, `pygame`, `cv2`, `gymnasium`)
3. Local application imports (`from vq_vae import VQVAE`)

Avoid `*` imports (e.g., `from module import *`) as they obscure namespace origins.

### 3.4. Typing
- While the legacy codebase is lightweight on type hints, **you should proactively add Python type hints (`typing` module) to new function signatures and complex variables** to improve code readability and static analysis.
- Example: `def train_model(epochs: int = 100, batch_size: int = 64) -> None:`

### 3.5. PyTorch Conventions
- **Device Placement**: Always support dynamic device placement (CUDA, MPS, CPU). Use the established pattern at the top of executable scripts:
  ```python
  if torch.cuda.is_available():
      DEVICE = torch.device("cuda")
  elif torch.backends.mps.is_available():
      DEVICE = torch.device("mps")
  else:
      DEVICE = torch.device("cpu")
  ```
- Always move models and tensors to `DEVICE` (e.g., `model.to(DEVICE)`, `tensor.to(DEVICE)`).
- Ensure tensors match the expected `dtype` (e.g., `torch.float32` for continuous images, `torch.long` for discrete actions/tokens).
- Normalize image tensors to `[0.0, 1.0]` by dividing pixel values by `255.0` when feeding into neural networks, and multiply back by `255` when converting back to NumPy arrays (`np.uint8`).

---

## 4. Architectural Patterns

### 4.1. File Structure
- `video_world_model.py`: Original continuous CNN video prediction logic and data collection.
- `vq_vae.py`: Vector Quantized Variational Autoencoder for tokenizing continuous frames into discrete latent representations.
- `transformer_world.py`: Autoregressive Transformer for predicting the next sequence of frame tokens.
- `dream_doom.py`: The interactive Gymnasium environment that glues the models together and handles user input/rendering.
- `rl_doom.py`: The wrapper for the actual ViZDoom engine (used for collecting true transition data).
- `data/`: Directory for `.npz` dataset files (ignored by git).
- `models/`: Directory for `.safetensors` model weights (ignored by git).

### 4.2. Error Handling
- Use specific exception types (e.g., `ValueError`, `FileNotFoundError`) rather than bare `Exception` where possible.
- Provide descriptive error messages that help developers debug issues quickly.
- When loading models via `torch.load()`, degrade gracefully or log a clear warning if weights are missing, rather than crashing (e.g., initializing random weights with a `WARNING` print statement).

---

## 5. Operational Guidelines for Agents

1. **Understand Before Modifying**: Use `read`, `glob`, and `grep` to analyze the context around your task. Do not assume the existence of functions or variables. Read the surrounding code before making edits.
2. **Incremental Changes**: When refactoring or adding large features, break them down into smaller, verifiable steps.
3. **Paths**: Always use absolute paths when reading or writing files via tools. If the user provides a relative path, resolve it against the project root (`/Users/homer/Projects/Worldlet`).
4. **No Destructive Operations**: Do not delete large amounts of code, `models/`, or `data/` directories without explicit user confirmation.
5. **Commenting**: Add inline comments for complex logic (e.g., tensor shape transformations `(B, C, H, W) -> (B, H, W, C)`, sequence offsets like `action + 512`), but avoid obvious comments. Explain *why*, not *what*.
6. **No Placeholder Implementations**: Provide complete, functional code. Do not leave `TODO`s or `pass` blocks unless specifically instructed to outline a skeleton structure.
7. **Proactive Formatting**: Before concluding a task that involved writing Python code, execute `ruff format .` via bash to ensure styling compliance.

---

## 6. Known Quirks / Specifics
- **Sequence Lengths**: In `transformer_world.py`, the sequence length is tightly coupled to the `VQVAE` latent spatial size. A 64x64 image encodes to an 8x8 latent grid (64 tokens). Pay extreme attention to tensor shapes and sequence indexing when modifying transformer context logic.
- **Action Offsets**: Actions are injected into the transformer sequence by offsetting their integer values by the visual vocabulary size (e.g., `action + 512`). This prevents the model from confusing action tokens with visual tokens.
- **Headless Rendering**: When collecting initial true frames via `rl_doom.py` without a display, `os.environ["SDL_VIDEODRIVER"] = "dummy"` is used to prevent PyGame/ViZDoom from crashing in headless environments.