#!/bin/bash
# Demo script for running inference on a single ADT sequence using cached data

# Configuration
MODEL_PATH="./pretrained/adt/adt.pth"
CACHE_DIR="./adt_cache"
OUTPUT_DIR="./demo_results"
SEQUENCE="Apartment_release_work_seq136_M1292"

# Run evaluation on the specified sequence (cache-only mode, no raw data needed)
python -m scripts.eval \
    --model_path ${MODEL_PATH} \
    --global_cache_dir ${CACHE_DIR} \
    --sequences ${SEQUENCE} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_workers 0 \
    --use_rerun \
    --num_vis_samples 10 \
    --force_use_cache

echo ""
echo "Demo completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Rerun visualizations saved to: ${OUTPUT_DIR}/*/rerun_visualizations/"
echo ""

