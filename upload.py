from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "homerquan/Worldlet-Concept-Demo"

# 1. Upload README
print("Uploading README.md...")
api.upload_file(
    path_or_fileobj="hf_readme.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)

# 2. Upload specific model files (safetensors only)
model_files = [
    "vq_vae_doom.safetensors",
    "transformer_world_doom.safetensors",
    "dit_world_doom.safetensors",
    "video_predict_model_doom.safetensors",
]

for f in model_files:
    local_path = os.path.join("models", f)
    if os.path.exists(local_path):
        print(f"Uploading {f}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"models/{f}",
            repo_id=repo_id,
            repo_type="model",
        )

# 3. Upload specific data files
data_files = ["transitions_doom_diverse.npz", "transitions_doom.npz"]

for f in data_files:
    local_path = os.path.join("data", f)
    if os.path.exists(local_path):
        print(f"Uploading {f}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"data/{f}",
            repo_id=repo_id,
            repo_type="model",
        )

print("Upload complete!")
