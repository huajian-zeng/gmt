#!/bin/bash

# Download the pretrained GMT checkpoint from Hugging Face.
# Usage: bash scripts/download_pretrained.sh

set -e

REPO_ID="huajian-zeng/gmt-adt"
SAVE_DIR="pretrained/adt"
mkdir -p "${SAVE_DIR}"

echo "Downloading pretrained GMT checkpoint from https://huggingface.co/${REPO_ID} ..."
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="model",
    local_dir="${SAVE_DIR}",
    allow_patterns=["adt.pth", "val_sequences.txt"],
)
PY

echo "Done! Checkpoint saved to ${SAVE_DIR}/adt.pth"
