#!/bin/bash

# Download pretrained models from Google Drive
# Usage: bash scripts/download_pretrained.sh

set -e

SAVE_DIR="pretrained"
mkdir -p ${SAVE_DIR}

ZIP_FILE_ID="1fqdo3MwRGsW_Y0OF5i80-UX2tJmXelBr"
ZIP_FILE_NAME="pretrained_models.zip"
echo "Downloading pretrained models..."
gdown --id ${ZIP_FILE_ID} -O ${SAVE_DIR}/${ZIP_FILE_NAME}

echo "Extracting..."
unzip -o ${SAVE_DIR}/${ZIP_FILE_NAME} -d ${SAVE_DIR}
rm ${SAVE_DIR}/${ZIP_FILE_NAME}

echo "Done! Models saved to ${SAVE_DIR}/"
