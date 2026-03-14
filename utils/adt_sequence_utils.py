#!/usr/bin/env python3
# Utilities for handling Aria Digital Twin sequences

import os
import glob
import random
import logging
from typing import List, Tuple, Optional, Dict, Any

def find_adt_sequences(base_dir: str) -> List[str]:
    """
    Scan a directory recursively to find Aria Digital Twin sequences.
    
    Args:
        base_dir: Base directory containing ADT sequences
        
    Returns:
        List of full paths to ADT sequence directories
    """
    # Check if the base directory exists
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory {base_dir} does not exist")
    
    # Look for sequence directories that match typical ADT sequence patterns
    # ADT sequences typically have structure markers like:
    # - data_provider.json file
    # - aria_device_calibration subdirectory
    # - ground_truth_data subdirectory
    
    sequence_candidates = []
    
    # First, look for directories containing data_provider.json
    for root, dirs, files in os.walk(base_dir):
        if "data_provider.json" in files:
            # This is likely an ADT sequence
            sequence_candidates.append(root)
    
    # If we didn't find any, try a more permissive approach
    if not sequence_candidates:
        # Look for directories with typical ADT sequence structure
        for root, dirs, files in os.walk(base_dir):
            if ("aria_device_calibration" in dirs or 
                "ground_truth_data" in dirs or 
                "mps" in dirs):
                sequence_candidates.append(root)
    
    # Filter out duplicates and sort
    sequence_paths = sorted(list(set(sequence_candidates)))
    
    print(f"Found {len(sequence_paths)} potential ADT sequences in {base_dir}")
    return sequence_paths

def create_train_test_split(
    sequence_paths: List[str], 
    train_ratio: float = 0.9,
    output_dir: str = ".",
    random_seed: int = 42,
    write_to_file: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Create a random train-test split of sequences and save it to files.
    
    Args:
        sequence_paths: List of sequence paths
        train_ratio: Ratio of sequences to use for training (default: 0.9)
        output_dir: Directory to save the split files
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_sequences, test_sequences)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle the sequences
    sequences = sequence_paths.copy()
    random.shuffle(sequences)
    
    # Calculate split point
    train_size = int(len(sequences) * train_ratio)
    
    # Split the sequences
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    

    
    if write_to_file:
        # Save the splits to files
        train_file = os.path.join(output_dir, "train_sequences.txt")
        test_file = os.path.join(output_dir, "test_sequences.txt")

        with open(train_file, "w") as f:
            f.write("\n".join(train_sequences))
    
        with open(test_file, "w") as f:
            f.write("\n".join(test_sequences))
        
        print(f"Split saved to {train_file} and {test_file}")
    
    print(f"Created train-test split with {len(train_sequences)} training and {len(test_sequences)} testing sequences")
    
    
    return train_sequences, test_sequences

