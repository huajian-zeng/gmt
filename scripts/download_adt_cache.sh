#!/bin/bash

# Download ADT dataset cache from Google Drive
# Usage: bash scripts/download_adt_cache.sh

set -e

SAVE_DIR="adt_cache"
mkdir -p ${SAVE_DIR}

# TODO: Replace with actual Google Drive file ID
ZIP_FILE_ID="1xuulpXEEJ3VweXH_8BWv8S4H-LBv63nK"
ZIP_FILE_NAME="adt_cache.zip"

echo "Downloading ADT dataset cache..."
gdown --id ${ZIP_FILE_ID} -O ${SAVE_DIR}/${ZIP_FILE_NAME}

echo "Extracting..."
unzip -o ${SAVE_DIR}/${ZIP_FILE_NAME} -d ${SAVE_DIR}
rm ${SAVE_DIR}/${ZIP_FILE_NAME}

echo "Done! Cache saved to ${SAVE_DIR}/"
