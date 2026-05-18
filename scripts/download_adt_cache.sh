#!/bin/bash

# Download the preprocessed ADT trajectory cache from Hugging Face.
# Usage: bash scripts/download_adt_cache.sh

set -e

REPO_ID="huajian-zeng/gmt-adt-cache"
SAVE_DIR="adt_cache"
mkdir -p "${SAVE_DIR}"

echo "Downloading ADT cache from https://huggingface.co/datasets/${REPO_ID} ..."
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="dataset",
    local_dir="${SAVE_DIR}",
    allow_patterns=["*.pkl"],
)
PY

echo "Done! Cache saved to ${SAVE_DIR}/"
