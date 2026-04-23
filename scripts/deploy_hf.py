import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")  # set HF_TOKEN env var before running
api = HfApi(token=token)
user = api.whoami()["name"]
repo_id = f"{user}/healthcare-qa-api"

print(f"Creating Hugging Face Space: {repo_id}")
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

print("Space created. Pushing files...")
# Push the backend directory to HF Spaces
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[".git", "*venv*", "venv*", "data/*", "models/*", "*.zip", "*.pdf", "__pycache__", ".next", "node_modules", "*.parquet", "*.log"]
)
print(f"Deployment complete: https://huggingface.co/spaces/{repo_id}")
