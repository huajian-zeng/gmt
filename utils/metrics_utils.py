#!/usr/bin/env python3
"""
Metrics and data processing utilities for ADT training and evaluation.
"""

import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Import geometry utilities for rotation conversion
try:
    from utils.geometry_utils import convert_rotation_to_euler
    HAS_GEOMETRY_UTILS = True
except ImportError:
    print("Warning: geometry_utils not found. Will use fallback rotation conversion.")
    HAS_GEOMETRY_UTILS = False


def transform_coords_for_visualization(tensor_3d: torch.Tensor, coord_mapping: list = None) -> torch.Tensor:
    """
    Applies coordinate transformation to a 3D tensor for visualization.
    
    Args:
        tensor_3d: Input tensor with last dimension of 3 (x, y, z coordinates)
        coord_mapping: List specifying the mapping, e.g., ['x', '-z', 'y'] for ADT dataset,
                      ['x', 'y', 'z'] for HD-EPIC dataset (no transformation).
                      Default is ['x', '-z', 'y'] for backward compatibility.
        
    Returns:
        torch.Tensor: Transformed tensor according to the specified mapping
    """
    if tensor_3d is None or tensor_3d.numel() == 0:
        return tensor_3d
    
    # Default mapping for backward compatibility (ADT dataset)
    if coord_mapping is None:
        coord_mapping = ['x', '-z', 'y']
    
    # Ensure it's a tensor
    if not isinstance(tensor_3d, torch.Tensor):
        tensor_3d = torch.tensor(tensor_3d)

    if tensor_3d.shape[-1] != 3:
        # print(f"Warning: transform_coords_for_visualization expects last dim to be 3, got {tensor_3d.shape}. Skipping transformation.")
        return tensor_3d

    # Extract coordinates
    coords = {'x': tensor_3d[..., 0], 'y': tensor_3d[..., 1], 'z': tensor_3d[..., 2]}
    
    # Apply mapping
    transformed_coords = []
    for axis in coord_mapping:
        if axis.startswith('-'):
            # Handle negative mapping
            coord_name = axis[1:]
            transformed_coords.append(-coords[coord_name])
        else:
            transformed_coords.append(coords[axis])
    
    transformed_tensor = torch.stack(transformed_coords, dim=-1)
    return transformed_tensor


def transform_coords_for_visualization_numpy(coords_3d: np.ndarray, coord_mapping: list = None) -> np.ndarray:
    """
    Applies coordinate transformation to a 3D numpy array for visualization.
    
    Args:
        coords_3d: Input numpy array with last dimension of 3 (x, y, z coordinates)
        coord_mapping: List specifying the mapping, e.g., ['x', '-z', 'y'] for ADT dataset,
                      ['x', 'y', 'z'] for HD-EPIC dataset (no transformation).
                      Default is ['x', '-z', 'y'] for backward compatibility.
        
    Returns:
        numpy.ndarray: Transformed array according to the specified mapping
    """
    if coords_3d is None or coords_3d.size == 0:
        return coords_3d
    
    # Default mapping for backward compatibility (ADT dataset)
    if coord_mapping is None:
        coord_mapping = ['x', '-z', 'y']
    
    # Ensure it's a numpy array
    if not isinstance(coords_3d, np.ndarray):
        coords_3d = np.array(coords_3d)

    if coords_3d.shape[-1] != 3:
        return coords_3d

    # Extract coordinates
    coords = {'x': coords_3d[..., 0], 'y': coords_3d[..., 1], 'z': coords_3d[..., 2]}
    
    # Apply mapping
    transformed_coords = []
    for axis in coord_mapping:
        if axis.startswith('-'):
            # Handle negative mapping
            coord_name = axis[1:]
            transformed_coords.append(-coords[coord_name])
        else:
            transformed_coords.append(coords[axis])
    
    transformed_coords = np.stack(transformed_coords, axis=-1)
    return transformed_coords


def convert_6d_to_euler_fallback(r6d_rotations):
    """
    Fallback function to convert 6D rotation to Euler angles when geometry_utils is not available.
    
    Args:
        r6d_rotations: [N, 6] tensor of 6D rotation representations
        
    Returns:
        ndarray: [N, 3] Euler angles (roll, pitch, yaw) in radians
    """
    if isinstance(r6d_rotations, torch.Tensor):
        r6d_rotations = r6d_rotations.detach().cpu().numpy()
    
    euler_angles = []
    for i in range(r6d_rotations.shape[0]):
        r6d = r6d_rotations[i]
        
        # Extract first two columns
        x = r6d[:3]
        y = r6d[3:]
        
        # Normalize and orthogonalize using Gram-Schmidt
        x = x / (np.linalg.norm(x) + 1e-8)
        y = y - np.dot(x, y) * x
        y = y / (np.linalg.norm(y) + 1e-8)
        z = np.cross(x, y)
        
        # Create rotation matrix
        R_matrix = np.column_stack([x, y, z])
        
        # Convert to Euler angles (ZYX convention)
        sy = np.sqrt(R_matrix[0, 0] * R_matrix[0, 0] + R_matrix[1, 0] * R_matrix[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R_matrix[2, 1], R_matrix[2, 2])
            pitch = np.arctan2(-R_matrix[2, 0], sy)
            yaw = np.arctan2(R_matrix[1, 0], R_matrix[0, 0])
        else:
            roll = np.arctan2(-R_matrix[1, 2], R_matrix[1, 1])
            pitch = np.arctan2(-R_matrix[2, 0], sy)
            yaw = 0
        
        euler_angles.append([roll, pitch, yaw])
    
    return np.array(euler_angles)


def extract_trajectory_orientation_changes_from_loader(dataloader, max_samples=None, position_dim=3):
    """
    Extract trajectory-level orientation change statistics from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader containing trajectory data
        max_samples: Maximum number of trajectories to process (None for all)
        position_dim: Number of position dimensions (default: 3)
        
    Returns:
        dict: Dictionary containing trajectory-level orientation change statistics
    """
    trajectory_stats = {
        'angular_velocities': [],      # Per-trajectory angular velocity statistics [N_traj, 3, stats]
        'total_rotations': [],         # Total rotation change per trajectory [N_traj, 3]
        'rotation_variances': [],      # Rotation variance per trajectory [N_traj, 3] 
        'max_angular_speeds': [],      # Maximum angular speed per trajectory [N_traj, 3]
        'trajectory_lengths': [],      # Valid length of each trajectory [N_traj]
    }
    sample_count = 0
    
    print(f"Extracting trajectory orientation changes from dataloader with {len(dataloader)} batches...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting trajectory orientation changes")):
        try:
            # Get full poses from batch
            full_poses = batch['full_poses']  # [B, seq_len, 9]
            attention_mask = batch.get('full_attention_mask', None)  # [B, seq_len]
            
            batch_size, seq_len, total_dim = full_poses.shape
            rotation_dim = total_dim - position_dim
            
            # Extract rotation components (6D representation)
            rotations = full_poses[:, :, position_dim:position_dim + rotation_dim]  # [B, seq_len, 6]
            
            # Process each trajectory in the batch
            for i in range(batch_size):
                trajectory_rotations = rotations[i]  # [seq_len, 6]
                
                # Apply attention mask to get valid frames
                if attention_mask is not None:
                    mask = attention_mask[i]  # [seq_len]
                    valid_indices = torch.where(mask > 0)[0]
                    if len(valid_indices) < 2:  # Need at least 2 frames to compute changes
                        continue
                    trajectory_rotations = trajectory_rotations[valid_indices]  # [valid_len, 6]
                
                valid_length = trajectory_rotations.shape[0]
                if valid_length < 2:
                    continue
                
                # Convert 6D rotations to Euler angles
                if HAS_GEOMETRY_UTILS:
                    euler_angles = convert_rotation_to_euler(trajectory_rotations)  # [valid_len, 3]
                else:
                    euler_angles = convert_6d_to_euler_fallback(trajectory_rotations)  # [valid_len, 3]
                
                if euler_angles is None or len(euler_angles) < 2:
                    continue
                
                # Convert to torch tensor if needed
                if isinstance(euler_angles, np.ndarray):
                    euler_angles = torch.from_numpy(euler_angles).float()
                
                # Calculate frame-to-frame angular changes
                angular_diffs = euler_angles[1:] - euler_angles[:-1]  # [valid_len-1, 3]
                
                # Handle angle wrapping (for angles around ±π)
                angular_diffs = torch.remainder(angular_diffs + np.pi, 2*np.pi) - np.pi
                
                # Calculate trajectory-level statistics
                # 1. Angular velocity statistics (mean, std, max of frame-to-frame changes)
                angular_vel_mean = torch.mean(torch.abs(angular_diffs), dim=0)  # [3]
                angular_vel_std = torch.std(torch.abs(angular_diffs), dim=0)   # [3]
                angular_vel_max = torch.max(torch.abs(angular_diffs), dim=0)[0]  # [3]
                
                # Stack statistics: [3, 3] where second dim is [mean, std, max]
                angular_vel_stats = torch.stack([angular_vel_mean, angular_vel_std, angular_vel_max], dim=1)
                
                # 2. Total rotation change (total absolute change across trajectory)
                total_rotation = torch.sum(torch.abs(angular_diffs), dim=0)  # [3]
                
                # 3. Rotation variance (variance of angles across the trajectory)
                rotation_variance = torch.var(euler_angles, dim=0)  # [3]
                
                # 4. Maximum angular speed (max of absolute angular velocities)
                max_angular_speed = torch.max(torch.abs(angular_diffs), dim=0)[0]  # [3]
                
                # Store trajectory-level statistics
                trajectory_stats['angular_velocities'].append(angular_vel_stats.numpy())
                trajectory_stats['total_rotations'].append(total_rotation.numpy())
                trajectory_stats['rotation_variances'].append(rotation_variance.numpy())
                trajectory_stats['max_angular_speeds'].append(max_angular_speed.numpy())
                trajectory_stats['trajectory_lengths'].append(valid_length)
                
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break
            
            if max_samples is not None and sample_count >= max_samples:
                break
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Convert lists to numpy arrays for easier analysis
    for key in ['angular_velocities', 'total_rotations', 'rotation_variances', 'max_angular_speeds']:
        if trajectory_stats[key]:
            trajectory_stats[key] = np.array(trajectory_stats[key])
    
    print(f"Extracted orientation change statistics for {sample_count} trajectories")
    return trajectory_stats


def plot_trajectory_orientation_distribution(train_loader, val_loader, output_path=None, max_samples_per_loader=1000, wandb_run=None):
    """
    Visualize and compare trajectory-level orientation change distributions between training and validation sets.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        output_path: Path to save the plot (optional)
        max_samples_per_loader: Maximum trajectories to extract from each loader
        wandb_run: WandB run object for logging (optional)
        
    Returns:
        dict: Statistics about the trajectory-level orientation change distributions
    """
    print("=== Analyzing Trajectory-Level Orientation Change Distributions ===")
    
    # Extract trajectory orientation change statistics from both loaders
    print("Extracting training trajectory orientation changes...")
    train_stats = extract_trajectory_orientation_changes_from_loader(train_loader, max_samples=max_samples_per_loader)
    
    print("Extracting validation trajectory orientation changes...")
    val_stats = extract_trajectory_orientation_changes_from_loader(val_loader, max_samples=max_samples_per_loader)
    
    if (len(train_stats['angular_velocities']) == 0 or len(val_stats['angular_velocities']) == 0):
        print("Error: No valid trajectory orientation changes found in one or both datasets!")
        return {}
    
    # Convert from radians to degrees for better interpretability
    angle_names = ['Roll', 'Pitch', 'Yaw']
    stats_names = ['Mean Angular Velocity', 'Std Angular Velocity', 'Max Angular Velocity']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Trajectory-Level Orientation Change Distribution Comparison: Training vs Validation', fontsize=16)
    
    colors = ['blue', 'red']
    labels = ['Training', 'Validation']
    comparison_stats = {}
    
    # Plot 1: Angular Velocity Statistics (mean, std, max for each angle)
    for angle_idx in range(3):
        for stat_idx in range(3):
            ax = axes[stat_idx, angle_idx]
            
            train_data = np.rad2deg(train_stats['angular_velocities'][:, angle_idx, stat_idx])
            val_data = np.rad2deg(val_stats['angular_velocities'][:, angle_idx, stat_idx])
            
            # Plot histograms
            bins = np.linspace(0, max(np.max(train_data), np.max(val_data)), 30)
            ax.hist(train_data, bins=bins, alpha=0.7, color=colors[0], 
                   label=f'{labels[0]} (n={len(train_data)})', density=True)
            ax.hist(val_data, bins=bins, alpha=0.7, color=colors[1], 
                   label=f'{labels[1]} (n={len(val_data)})', density=True)
            
            ax.set_xlabel(f'{stats_names[stat_idx]} (degrees/frame)')
            ax.set_ylabel('Density')
            ax.set_title(f'{angle_names[angle_idx]} - {stats_names[stat_idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate comparison statistics
            key = f'{angle_names[angle_idx].lower()}_{stats_names[stat_idx].lower().replace(" ", "_")}'
            comparison_stats[f'{key}_train'] = {
                'mean': np.mean(train_data),
                'std': np.std(train_data),
                'median': np.median(train_data),
            }
            comparison_stats[f'{key}_val'] = {
                'mean': np.mean(val_data),
                'std': np.std(val_data),
                'median': np.median(val_data),
            }
    
    # Plot 2: Total Rotation Change per Trajectory
    ax_total = axes[0, 3]
    train_total_rotation_norm = np.rad2deg(np.linalg.norm(train_stats['total_rotations'], axis=1))
    val_total_rotation_norm = np.rad2deg(np.linalg.norm(val_stats['total_rotations'], axis=1))
    
    bins = np.linspace(0, max(np.max(train_total_rotation_norm), np.max(val_total_rotation_norm)), 30)
    ax_total.hist(train_total_rotation_norm, bins=bins, alpha=0.7, color=colors[0], 
                 label=f'{labels[0]}', density=True)
    ax_total.hist(val_total_rotation_norm, bins=bins, alpha=0.7, color=colors[1], 
                 label=f'{labels[1]}', density=True)
    ax_total.set_xlabel('Total Rotation Change (degrees)')
    ax_total.set_ylabel('Density')
    ax_total.set_title('Total Rotation Change per Trajectory')
    ax_total.legend()
    ax_total.grid(True, alpha=0.3)
    
    # Plot 3: Rotation Variance per Trajectory
    ax_var = axes[1, 3]
    train_rotation_var_norm = np.rad2deg(np.sqrt(np.mean(train_stats['rotation_variances'], axis=1)))
    val_rotation_var_norm = np.rad2deg(np.sqrt(np.mean(val_stats['rotation_variances'], axis=1)))
    
    bins = np.linspace(0, max(np.max(train_rotation_var_norm), np.max(val_rotation_var_norm)), 30)
    ax_var.hist(train_rotation_var_norm, bins=bins, alpha=0.7, color=colors[0], 
               label=f'{labels[0]}', density=True)
    ax_var.hist(val_rotation_var_norm, bins=bins, alpha=0.7, color=colors[1], 
               label=f'{labels[1]}', density=True)
    ax_var.set_xlabel('RMS Rotation Variance (degrees)')
    ax_var.set_ylabel('Density')
    ax_var.set_title('Rotation Variance per Trajectory')
    ax_var.legend()
    ax_var.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory Length Distribution
    ax_len = axes[2, 3]
    ax_len.hist(train_stats['trajectory_lengths'], bins=30, alpha=0.7, color=colors[0], 
               label=f'{labels[0]}', density=True)
    ax_len.hist(val_stats['trajectory_lengths'], bins=30, alpha=0.7, color=colors[1], 
               label=f'{labels[1]}', density=True)
    ax_len.set_xlabel('Trajectory Length (frames)')
    ax_len.set_ylabel('Density')
    ax_len.set_title('Trajectory Length Distribution')
    ax_len.legend()
    ax_len.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory orientation change distribution plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics and detect potential issues
    print("\n=== Trajectory-Level Orientation Change Statistics Summary ===")
    
    issues_detected = []
    
    # Check angular velocity differences
    for angle_idx, angle_name in enumerate(['roll', 'pitch', 'yaw']):
        train_mean_vel = np.mean(train_stats['angular_velocities'][:, angle_idx, 0])
        val_mean_vel = np.mean(val_stats['angular_velocities'][:, angle_idx, 0])
        mean_vel_diff = np.abs(np.rad2deg(train_mean_vel - val_mean_vel))
        
        train_max_vel = np.mean(train_stats['angular_velocities'][:, angle_idx, 2])
        val_max_vel = np.mean(val_stats['angular_velocities'][:, angle_idx, 2])
        max_vel_diff = np.abs(np.rad2deg(train_max_vel - val_max_vel))
        
        print(f"\n{angle_name.upper()} Angular Velocity:")
        print(f"  Training   - Mean: {np.rad2deg(train_mean_vel):.2f}°/frame, Max: {np.rad2deg(train_max_vel):.2f}°/frame")
        print(f"  Validation - Mean: {np.rad2deg(val_mean_vel):.2f}°/frame, Max: {np.rad2deg(val_max_vel):.2f}°/frame")
        print(f"  Difference - Mean: {mean_vel_diff:.2f}°/frame, Max: {max_vel_diff:.2f}°/frame")
        
        # Detect issues
        if mean_vel_diff > 5.0:  # More than 5 degrees/frame difference
            issues_detected.append(f"{angle_name.upper()}: Large mean angular velocity difference ({mean_vel_diff:.1f}°/frame)")
        if max_vel_diff > 10.0:  # More than 10 degrees/frame difference
            issues_detected.append(f"{angle_name.upper()}: Large max angular velocity difference ({max_vel_diff:.1f}°/frame)")
    
    # Check total rotation differences
    train_total_mean = np.mean(train_total_rotation_norm)
    val_total_mean = np.mean(val_total_rotation_norm)
    total_diff = np.abs(train_total_mean - val_total_mean)
    
    print(f"\nTotal Rotation per Trajectory:")
    print(f"  Training   - Mean: {train_total_mean:.2f}°")
    print(f"  Validation - Mean: {val_total_mean:.2f}°")
    print(f"  Difference: {total_diff:.2f}°")
    
    if total_diff > 20.0:  # More than 20 degrees difference
        issues_detected.append(f"TOTAL_ROTATION: Large difference in total rotation per trajectory ({total_diff:.1f}°)")
    
    # Check trajectory length differences
    train_len_mean = np.mean(train_stats['trajectory_lengths'])
    val_len_mean = np.mean(val_stats['trajectory_lengths'])
    len_diff = np.abs(train_len_mean - val_len_mean)
    
    print(f"\nTrajectory Lengths:")
    print(f"  Training   - Mean: {train_len_mean:.1f} frames")
    print(f"  Validation - Mean: {val_len_mean:.1f} frames")
    print(f"  Difference: {len_diff:.1f} frames")
    
    if len_diff > 5.0:  # More than 5 frames difference
        issues_detected.append(f"TRAJECTORY_LENGTH: Large difference in average trajectory length ({len_diff:.1f} frames)")
    
    # Report issues
    if issues_detected:
        print(f"\n⚠️  Potential Issues Detected:")
        for issue in issues_detected:
            print(f"  - {issue}")
        print(f"\nThese differences might contribute to orientation loss issues.")
        print(f"Consider:")
        print(f"  - Data augmentation for orientation changes")
        print(f"  - Adjusting lambda_ori weight based on these differences")
        print(f"  - Trajectory-specific normalization")
    else:
        print(f"\n✅ No major distribution differences detected.")
    
    # Store additional statistics
    comparison_stats.update({
        'total_rotation_train_mean': train_total_mean,
        'total_rotation_val_mean': val_total_mean,
        'trajectory_length_train_mean': train_len_mean,
        'trajectory_length_val_mean': val_len_mean,
        'issues_detected': issues_detected,
    })
    
    return comparison_stats




class GradientTracker:
    """
    Class to track gradients during training for different model components.
    """
    
    def __init__(self):
        self.gradient_logs = defaultdict(list)
        self.iteration_count = 0
    
    def log_gradients(self, model, loss_components=None):
        """
        Log gradient magnitudes for different model components.
        
        Args:
            model: The neural network model
            loss_components: Optional dict of individual loss components
        """
        self.iteration_count += 1
        
        # Log gradients for different model components
        component_gradients = {}
        
        # Check for different model components and log their gradients
        if hasattr(model, 'outputlayer'):
            # Final output layer (predicts full trajectory including orientation)
            output_grads = []
            for param in model.outputlayer.parameters():
                if param.grad is not None:
                    output_grads.append(param.grad.detach().flatten())
            if output_grads:
                output_grad_norm = torch.norm(torch.cat(output_grads), p=2).item()
                component_gradients['output_layer'] = output_grad_norm
        
        # Motion-related components
        if hasattr(model, 'motion_linear'):
            motion_grads = []
            for param in model.motion_linear.parameters():
                if param.grad is not None:
                    motion_grads.append(param.grad.detach().flatten())
            if motion_grads:
                motion_grad_norm = torch.norm(torch.cat(motion_grads), p=2).item()
                component_gradients['motion_embedding'] = motion_grad_norm
        
        # Scene encoder components
        if hasattr(model, 'scene_encoder'):
            scene_grads = []
            for param in model.scene_encoder.parameters():
                if param.grad is not None:
                    scene_grads.append(param.grad.detach().flatten())
            if scene_grads:
                scene_grad_norm = torch.norm(torch.cat(scene_grads), p=2).item()
                component_gradients['scene_encoder'] = scene_grad_norm
        
        # Category embedding (if exists)
        if hasattr(model, 'category_embedding'):
            cat_grads = []
            for param in model.category_embedding.parameters():
                if param.grad is not None:
                    cat_grads.append(param.grad.detach().flatten())
            if cat_grads:
                cat_grad_norm = torch.norm(torch.cat(cat_grads), p=2).item()
                component_gradients['category_embedding'] = cat_grad_norm
        
        # Motion-bbox encoder
        if hasattr(model, 'motion_bbox_encoder'):
            motion_bbox_grads = []
            for param in model.motion_bbox_encoder.parameters():
                if param.grad is not None:
                    motion_bbox_grads.append(param.grad.detach().flatten())
            if motion_bbox_grads:
                motion_bbox_grad_norm = torch.norm(torch.cat(motion_bbox_grads), p=2).item()
                component_gradients['motion_bbox_encoder'] = motion_bbox_grad_norm
        
        # Output encoder
        if hasattr(model, 'output_encoder'):
            output_enc_grads = []
            for param in model.output_encoder.parameters():
                if param.grad is not None:
                    output_enc_grads.append(param.grad.detach().flatten())
            if output_enc_grads:
                output_enc_grad_norm = torch.norm(torch.cat(output_enc_grads), p=2).item()
                component_gradients['output_encoder'] = output_enc_grad_norm
        
        # Overall model gradient norm
        total_grads = []
        for param in model.parameters():
            if param.grad is not None:
                total_grads.append(param.grad.detach().flatten())
        if total_grads:
            total_grad_norm = torch.norm(torch.cat(total_grads), p=2).item()
            component_gradients['total_model'] = total_grad_norm
        
        # Store the gradients
        for component, grad_norm in component_gradients.items():
            self.gradient_logs[component].append(grad_norm)
        
        # Also log loss components if provided
        if loss_components:
            for loss_name, loss_value in loss_components.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                self.gradient_logs[f'loss_{loss_name}'].append(loss_value)
    
    def get_logs(self):
        """
        Get the current gradient logs.
        
        Returns:
            dict: Dictionary of gradient logs
        """
        return dict(self.gradient_logs)
    
    def save_logs(self, save_path):
        """
        Save gradient logs to a file.
        
        Args:
            save_path: Path to save the logs
        """
        logs_dict = self.get_logs()
        torch.save(logs_dict, save_path)
        print(f"Gradient logs saved to: {save_path}")
    
    def load_logs(self, load_path):
        """
        Load gradient logs from a file.
        
        Args:
            load_path: Path to load the logs from
        """
        if os.path.exists(load_path):
            logs_dict = torch.load(load_path)
            self.gradient_logs = defaultdict(list, logs_dict)
            self.iteration_count = len(list(logs_dict.values())[0]) if logs_dict else 0
            print(f"Gradient logs loaded from: {load_path}")
        else:
            print(f"Warning: Gradient log file not found: {load_path}")


def plot_gradient_magnitudes(gradient_logs, component_names=None, output_path=None, 
                           window_size=50, show_losses=True, wandb_run=None, step=None):
    """
    Visualize gradient magnitudes over training iterations.
    
    Args:
        gradient_logs: Dictionary containing gradient logs for different components
        component_names: List of component names to plot (None for all)
        output_path: Path to save the plot (optional)
        window_size: Window size for moving average smoothing
        show_losses: Whether to include loss curves in the plot
        wandb_run: WandB run object for logging (optional)
        step: Training step/epoch for WandB logging (optional)
        
    Returns:
        dict: Summary statistics of gradients
    """
    if not gradient_logs:
        print("No gradient logs provided!")
        return {}
    
    # Filter logs based on component_names
    if component_names is not None:
        filtered_logs = {name: gradient_logs[name] for name in component_names if name in gradient_logs}
    else:
        filtered_logs = gradient_logs
    
    # Separate gradient logs from loss logs
    grad_logs = {k: v for k, v in filtered_logs.items() if not k.startswith('loss_')}
    loss_logs = {k: v for k, v in filtered_logs.items() if k.startswith('loss_')}
    
    # Determine subplot layout
    if show_losses and loss_logs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None
    
    # Plot gradient magnitudes
    iterations = range(len(list(grad_logs.values())[0])) if grad_logs else []
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(grad_logs)))
    
    for i, (component, grad_values) in enumerate(grad_logs.items()):
        if len(grad_values) > 0:
            # Apply moving average smoothing if window_size > 1
            if window_size > 1 and len(grad_values) > window_size:
                smoothed_values = np.convolve(grad_values, np.ones(window_size)/window_size, mode='valid')
                smoothed_iterations = iterations[window_size-1:]
                ax1.plot(smoothed_iterations, smoothed_values, label=f'{component} (smoothed)', 
                        color=colors[i], linewidth=2)
                ax1.plot(iterations, grad_values, alpha=0.3, color=colors[i], linewidth=0.5)
            else:
                ax1.plot(iterations, grad_values, label=component, color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Gradient Magnitude (L2 Norm)')
    ax1.set_title('Gradient Magnitudes by Model Component')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot loss curves if requested
    if show_losses and loss_logs and ax2 is not None:
        loss_colors = plt.cm.Set1(np.linspace(0, 1, len(loss_logs)))
        
        for i, (loss_name, loss_values) in enumerate(loss_logs.items()):
            if len(loss_values) > 0:
                # Remove 'loss_' prefix from legend
                display_name = loss_name.replace('loss_', '')
                
                # Apply moving average smoothing
                if window_size > 1 and len(loss_values) > window_size:
                    smoothed_values = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
                    smoothed_iterations = iterations[window_size-1:]
                    ax2.plot(smoothed_iterations, smoothed_values, label=f'{display_name} (smoothed)', 
                            color=loss_colors[i], linewidth=2)
                    ax2.plot(iterations, loss_values, alpha=0.3, color=loss_colors[i], linewidth=0.5)
                else:
                    ax2.plot(iterations, loss_values, label=display_name, color=loss_colors[i], linewidth=2)
        
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Loss Value')
        ax2.set_title('Loss Components Over Time')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    # Save plot if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gradient magnitude plot saved to: {output_path}")
    
    # Calculate and return summary statistics
    stats = {}
    for component, grad_values in grad_logs.items():
        if len(grad_values) > 0:
            stats[component] = {
                'mean': np.mean(grad_values),
                'std': np.std(grad_values),
                'max': np.max(grad_values),
                'min': np.min(grad_values),
                'final': grad_values[-1] if len(grad_values) > 0 else 0,
                'trend': 'increasing' if len(grad_values) > 10 and grad_values[-1] > np.mean(grad_values[:10]) else 'decreasing'
            }
    
    plt.show()
    
    # Print summary
    print("\n=== Gradient Magnitude Summary ===")
    for component, component_stats in stats.items():
        print(f"\n{component}:")
        print(f"  Mean: {component_stats['mean']:.6f}")
        print(f"  Final: {component_stats['final']:.6f}")
        print(f"  Max: {component_stats['max']:.6f}")
        print(f"  Trend: {component_stats['trend']}")
    
    return stats


def compute_frechet_distance(pred_traj, gt_traj, mask=None):
    """
    Compute the Fréchet distance between two trajectories.
    
    Args:
        pred_traj (torch.Tensor): Predicted trajectory [T, 3]
        gt_traj (torch.Tensor): Ground truth trajectory [T, 3]
        mask (torch.Tensor): Mask for valid points [T], optional
        
    Returns:
        torch.Tensor: Fréchet distance (scalar)
    """
    # Convert to numpy for computation
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.detach().cpu().numpy()
    if isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Apply mask if provided
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return torch.tensor(0.0)
        pred_traj = pred_traj[valid_indices]
        gt_traj = gt_traj[valid_indices]
    
    # Handle case with insufficient points
    if len(pred_traj) < 2 or len(gt_traj) < 2:
        return torch.tensor(0.0)
    
    # Compute distance matrix between all points
    # pred_traj: [n, 3], gt_traj: [m, 3]
    n, m = len(pred_traj), len(gt_traj)
    
    # Create distance matrix
    distance_matrix = cdist(pred_traj, gt_traj, metric='euclidean')
    
    # Dynamic programming to compute Fréchet distance
    # Initialize DP table
    dp = np.full((n, m), np.inf)
    dp[0, 0] = distance_matrix[0, 0]
    
    # Fill first row
    for j in range(1, m):
        dp[0, j] = max(dp[0, j-1], distance_matrix[0, j])
    
    # Fill first column
    for i in range(1, n):
        dp[i, 0] = max(dp[i-1, 0], distance_matrix[i, 0])
    
    # Fill the rest of the table
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i-1, j], dp[i-1, j-1], dp[i, j-1]),
                distance_matrix[i, j]
            )
    
    frechet_dist = dp[n-1, m-1]
    return torch.tensor(frechet_dist, dtype=torch.float32)


def compute_angular_cosine_similarity(pred_traj, gt_traj, mask=None):
    """
    Compute the Angular Cosine Similarity between movement directions of two trajectories.
    
    Args:
        pred_traj (torch.Tensor): Predicted trajectory [T, 3]
        gt_traj (torch.Tensor): Ground truth trajectory [T, 3]
        mask (torch.Tensor): Mask for valid points [T], optional
        
    Returns:
        torch.Tensor: Angular cosine similarity (scalar, 1=perfect, 0=orthogonal, -1=opposite)
    """
    device = pred_traj.device
    
    # Apply mask if provided
    if mask is not None:
        valid_indices = torch.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=device)
        pred_traj = pred_traj[valid_indices]
        gt_traj = gt_traj[valid_indices]
    
    # Handle case with insufficient points
    if pred_traj.shape[0] < 2 or gt_traj.shape[0] < 2:
        return torch.tensor(0.0, device=device)
    
    # Compute movement vectors (direction between consecutive points)
    pred_directions = pred_traj[1:] - pred_traj[:-1]  # [T-1, 3]
    gt_directions = gt_traj[1:] - gt_traj[:-1]        # [T-1, 3]
    
    # Remove zero-length vectors
    pred_norms = torch.norm(pred_directions, dim=-1)  # [T-1]
    gt_norms = torch.norm(gt_directions, dim=-1)      # [T-1]
    
    # Find non-zero movement vectors
    valid_pred = pred_norms > 1e-6
    valid_gt = gt_norms > 1e-6
    valid_both = valid_pred & valid_gt
    
    if torch.sum(valid_both) == 0:
        return torch.tensor(0.0, device=device)
    
    # Filter to valid movements
    pred_dirs_valid = pred_directions[valid_both]    # [N_valid, 3]
    gt_dirs_valid = gt_directions[valid_both]        # [N_valid, 3]
    pred_norms_valid = pred_norms[valid_both]        # [N_valid]
    gt_norms_valid = gt_norms[valid_both]            # [N_valid]
    
    # Normalize direction vectors
    pred_dirs_normalized = pred_dirs_valid / pred_norms_valid.unsqueeze(-1)  # [N_valid, 3]
    gt_dirs_normalized = gt_dirs_valid / gt_norms_valid.unsqueeze(-1)        # [N_valid, 3]
    
    # Compute cosine similarity for each pair of direction vectors
    cosine_similarities = torch.sum(pred_dirs_normalized * gt_dirs_normalized, dim=-1)  # [N_valid]
    
    # Average cosine similarity across all valid direction pairs
    mean_cosine_similarity = torch.mean(cosine_similarities)
    
    return mean_cosine_similarity


def check_gt_trajectory_collision(gt_trajectory, semantic_bbox_info, semantic_bbox_mask, trajectory_mask=None):
    """
    Check if ground truth trajectory has collision with semantic bounding boxes.
    This is used to exclude GT trajectories that inherently have collisions from evaluation.
    
    Args:
        gt_trajectory (torch.Tensor): Ground truth trajectory positions [T, 3]
        semantic_bbox_info (torch.Tensor): Semantic bbox info [max_bboxes, 12]
                                          Format: [center_x, center_y, center_z, size_x, size_y, size_z, rot_6d]
        semantic_bbox_mask (torch.Tensor): Mask for valid bboxes [max_bboxes]
        trajectory_mask (torch.Tensor): Mask for valid trajectory points [T], optional
        
    Returns:
        has_collision (bool): True if GT trajectory has any collision
        collision_info (dict): Dictionary with collision details (collision_points, collision_bboxes, etc.)
    """
    device = gt_trajectory.device
    T = gt_trajectory.shape[0]
    
    # Apply trajectory mask if provided
    if trajectory_mask is not None:
        valid_indices = torch.where(trajectory_mask > 0)[0]
        if len(valid_indices) == 0:
            return False, {'collision_points': [], 'collision_bboxes': []}
        trajectory_points = gt_trajectory[valid_indices]
    else:
        trajectory_points = gt_trajectory
        valid_indices = torch.arange(T, device=device)
    
    # Check if there are any valid bboxes
    if semantic_bbox_mask is None or semantic_bbox_mask.sum() == 0:
        return False, {'collision_points': [], 'collision_bboxes': []}
    
    # Get valid bboxes
    valid_bbox_indices = torch.where(semantic_bbox_mask > 0)[0]
    valid_bboxes = semantic_bbox_info[valid_bbox_indices]  # [N_valid_bboxes, 12]
    
    # Extract bbox components
    bbox_centers = valid_bboxes[:, :3]  # [N_valid_bboxes, 3]
    bbox_sizes = valid_bboxes[:, 3:6]   # [N_valid_bboxes, 3]
    bbox_half_sizes = bbox_sizes / 2.0  # [N_valid_bboxes, 3]
    
    # Track collision information
    collision_points = []
    collision_bboxes = []
    
    # Check collision for each trajectory point against all bboxes
    for i, point_idx in enumerate(valid_indices):
        point = trajectory_points[i:i+1]  # [1, 3]
        
        # Compute distance from point to each bbox center
        distances = torch.abs(point - bbox_centers)  # [N_valid_bboxes, 3]
        
        # Check if point is inside any bbox (axis-aligned)
        inside_bbox = torch.all(distances <= bbox_half_sizes, dim=1)  # [N_valid_bboxes]
        
        # If inside any bbox, record collision
        if torch.any(inside_bbox):
            collision_points.append(point_idx.item())
            colliding_bbox_indices = torch.where(inside_bbox)[0]
            for bbox_idx in colliding_bbox_indices:
                collision_bboxes.append(valid_bbox_indices[bbox_idx].item())
    
    has_collision = len(collision_points) > 0
    
    collision_info = {
        'collision_points': collision_points,  # List of trajectory point indices that collide
        'collision_bboxes': list(set(collision_bboxes)),  # List of unique bbox indices that collide
        'num_collision_points': len(collision_points),
        'collision_ratio': len(collision_points) / len(valid_indices) if len(valid_indices) > 0 else 0.0
    }
    
    return has_collision, collision_info


def compute_bbox_trajectory_collision(trajectory, semantic_bbox_info, semantic_bbox_mask, trajectory_mask=None):
    """
    Compute collision between a trajectory and semantic bounding boxes.
    A trajectory is considered to have collision if ANY point collides with any bbox.
    
    Args:
        trajectory (torch.Tensor): Trajectory positions [T, 3]
        semantic_bbox_info (torch.Tensor): Semantic bbox info [max_bboxes, 12]
                                          Format: [center_x, center_y, center_z, size_x, size_y, size_z, rot_6d]
        semantic_bbox_mask (torch.Tensor): Mask for valid bboxes [max_bboxes]
        trajectory_mask (torch.Tensor): Mask for valid trajectory points [T], optional
        
    Returns:
        has_collision (torch.Tensor): Binary indicator - 1.0 if trajectory has any collision, 0.0 otherwise (scalar)
        collision_mask (torch.Tensor): Binary mask indicating collision for each trajectory point [T]
    """
    device = trajectory.device
    T = trajectory.shape[0]
    
    # Initialize collision mask
    collision_mask = torch.zeros(T, dtype=torch.bool, device=device)
    
    # Apply trajectory mask if provided
    if trajectory_mask is not None:
        valid_indices = torch.where(trajectory_mask > 0)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device), collision_mask
        trajectory_points = trajectory[valid_indices]
    else:
        trajectory_points = trajectory
        valid_indices = torch.arange(T, device=device)
    
    # Check if there are any valid bboxes
    if semantic_bbox_mask is None or semantic_bbox_mask.sum() == 0:
        return torch.tensor(0.0, device=device), collision_mask
    
    # Get valid bboxes
    valid_bbox_indices = torch.where(semantic_bbox_mask > 0)[0]
    valid_bboxes = semantic_bbox_info[valid_bbox_indices]  # [N_valid_bboxes, 12]
    
    # Extract bbox components
    bbox_centers = valid_bboxes[:, :3]  # [N_valid_bboxes, 3]
    bbox_sizes = valid_bboxes[:, 3:6]   # [N_valid_bboxes, 3]
    # Note: We'll use axis-aligned bbox collision for simplicity, ignoring rotation
    
    # Compute half-sizes for easier collision detection
    bbox_half_sizes = bbox_sizes / 2.0  # [N_valid_bboxes, 3]
    
    # Check collision for each trajectory point against all bboxes
    for i, point_idx in enumerate(valid_indices):
        point = trajectory_points[i:i+1]  # [1, 3]
        
        # Compute distance from point to each bbox center
        distances = torch.abs(point - bbox_centers)  # [N_valid_bboxes, 3]
        
        # Check if point is inside any bbox (axis-aligned)
        inside_bbox = torch.all(distances <= bbox_half_sizes, dim=1)  # [N_valid_bboxes]
        
        # If inside any bbox, mark collision
        if torch.any(inside_bbox):
            collision_mask[point_idx] = True
    
    # Check if ANY point has collision
    # If any point collides, the entire trajectory is considered to have failed
    if trajectory_mask is not None:
        has_any_collision = collision_mask[valid_indices].any()
    else:
        has_any_collision = collision_mask.any()
    
    # Return binary result: 1.0 if collision exists, 0.0 otherwise
    has_collision = torch.tensor(1.0 if has_any_collision else 0.0, device=device)
    
    return has_collision, collision_mask


def compute_metrics_for_sample(pred_future, gt_future, future_mask):
    """
    Compute L1 Mean, L2 Mean (for RMSE), FDE, Fréchet Distance, and Angular Cosine Similarity for a single trajectory sample.

    Args:
        pred_future (torch.Tensor): Predicted future trajectory [Fut_Len, 3] (position only)
        gt_future (torch.Tensor): Ground truth future trajectory [Fut_Len, 3] (position only)
        future_mask (torch.Tensor): Mask for valid future points [Fut_Len]

    Returns:
        l1_mean (torch.Tensor): Mean L1 distance (scalar)
        rmse_ade (torch.Tensor): Mean L2 distance (RMSE analog for ADE) (scalar)
        fde (torch.Tensor): Final Displacement Error (scalar)
        frechet_distance (torch.Tensor): Fréchet distance between trajectories (scalar)
        angular_cosine_similarity (torch.Tensor): Angular cosine similarity (scalar)
    """
    # Ensure tensors are on the same device
    device = pred_future.device
    gt_future = gt_future.to(device)
    future_mask = future_mask.to(device)

    # Calculate L1 distance for all points
    l1_diff = torch.abs(pred_future - gt_future)  # Shape: [Fut_Len, 3]
    # Calculate per-timestep L2 distance (Euclidean norm)
    l2_dist_per_step = torch.norm(pred_future - gt_future, dim=-1)  # Shape: [Fut_Len]

    # Expand mask to match 3D coordinates (for L1)
    future_mask_expanded = future_mask.unsqueeze(-1).expand_as(l1_diff)  # Shape: [Fut_Len, 3]

    # Count valid points (use original 1D mask)
    num_valid_points = future_mask.sum()
    if num_valid_points == 0:
        # Return zeros if no valid points to avoid division by zero
        return (torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device))
        
    num_valid_coords = num_valid_points * 3  # Total number of valid coordinate values

    # --- L1 Mean (MAE) Calculation ---
    # Mask out invalid points
    masked_l1_diff = l1_diff * future_mask_expanded
    # Sum distances over all valid coordinates
    sum_l1_diff = masked_l1_diff.sum()
    # Calculate average L1 distance
    l1_mean = sum_l1_diff / num_valid_coords

    # --- RMSE / ADE (L2) Calculation ---
    # Mask out invalid points
    masked_l2_dist = l2_dist_per_step * future_mask  # Use 1D mask here
    # Sum L2 distances over all valid timesteps
    sum_l2_dist = masked_l2_dist.sum()
    # Calculate average L2 distance over valid timesteps
    rmse_ade = sum_l2_dist / num_valid_points

    # --- FDE Calculation ---
    # Find the index of the last valid point (using 1D mask)
    last_valid_index = num_valid_points.long() - 1
    last_valid_index = torch.clamp(last_valid_index, min=0)  # Clamp index to avoid negative indices

    # Get the predicted and ground truth points at the final valid timestep
    final_pred_point = pred_future[last_valid_index]
    final_gt_point = gt_future[last_valid_index]

    # Calculate FDE as the L2 distance between these final points
    fde = torch.norm(final_pred_point - final_gt_point, dim=-1)

    # --- Fréchet Distance Calculation ---
    frechet_distance = compute_frechet_distance(pred_future, gt_future, future_mask)

    # --- Angular Cosine Similarity Calculation ---
    angular_cosine_similarity = compute_angular_cosine_similarity(pred_future, gt_future, future_mask)

    return l1_mean, rmse_ade, fde, frechet_distance, angular_cosine_similarity


def collate_fn(batch, dataset, num_sample_points):
    """
    Custom collate function to handle trajectory-specific point clouds.

    Args:
        batch (list): A list of sample dictionaries from the dataset.
                      Each dict must contain 'dataset_idx' and other required data.
        dataset: The instance of the dataset being used.
                 Needed to access get_scene_pointcloud.
        num_sample_points (int): The number of points to sample from each point cloud.

    Returns:
        dict: A batch dictionary suitable for the model, including batched point clouds.
    """
    # Debug: Check for problematic fields in first item
    if len(batch) > 1:
        first_item_keys = set(batch[0].keys())
        problematic_fields = []
        for key in first_item_keys:
            if key in batch[0] and hasattr(batch[0][key], 'shape'):
                shapes = [item[key].shape if key in item and hasattr(item[key], 'shape') else 'missing' 
                         for item in batch]
                if len(set(str(s) for s in shapes)) > 1:
                    problematic_fields.append((key, shapes))
        
        if problematic_fields:
            print(f"Warning: Found fields with inconsistent shapes in batch:")
            for field, shapes in problematic_fields:
                print(f"  {field}: {shapes}")
    # Create a list to store point clouds for each item in the batch
    point_cloud_batch_list = []
    
    # Debug info
    # if len(batch) > 1:
    #     print(f"Processing batch of size {len(batch)} with target point cloud size {num_sample_points}")
    
    for i, item in enumerate(batch):
        # First check if the item has its own trajectory-specific point cloud
        if 'trajectory_specific_pointcloud' in item and item['trajectory_specific_pointcloud'] is not None:
            point_cloud = item['trajectory_specific_pointcloud']
            
            # Convert to tensor if it's a numpy array
            if isinstance(point_cloud, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud).float()
                
        else:
            # Fallback to getting from dataset if not included in the item
            dataset_idx = item.get('dataset_idx', 0)
            point_cloud = dataset.get_scene_pointcloud(dataset_idx)
        
        if point_cloud is None:
            print(f"Warning: Failed to get point cloud for dataset_idx {dataset_idx} in batch. Using zeros.")
            # Create a dummy point cloud if loading fails
            point_cloud = torch.zeros((num_sample_points, 3), dtype=torch.float32)
        elif isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float()
        
        if i == 0:  # Only print for first item to avoid log spam
            # print(f"Using full scene point cloud with {point_cloud.shape[0]} points")
            pass
    
        # Sample the point cloud to ensure consistent size
        if point_cloud.shape[0] >= num_sample_points:
            # Randomly sample points without replacement
            indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=False)
            sampled_pc = point_cloud[indices]
        else:
            # If not enough points, sample with replacement
            if i == 0:  # Only print for first item to avoid log spam
                # print(f"Warning: Point cloud has only {point_cloud.shape[0]} points. Sampling with replacement to get {num_sample_points}.")
                pass
            indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=True)
            sampled_pc = point_cloud[indices]
        
        # Add to batch list
        point_cloud_batch_list.append(sampled_pc)
    
    # Stack the point clouds
    batched_point_clouds = torch.stack(point_cloud_batch_list, dim=0)
    
    # Remove all point cloud related fields and dataset_idx from items to avoid issues with default_collate
    # These fields have variable sizes and need special handling
    excluded_keys = [
        'trajectory_specific_pointcloud', 
        'scene_pointcloud',  # This was causing the tensor size mismatch!
        'scene_pointcloud_source',
        'dataset_idx'
    ]
    
    batch_copy = []
    for item in batch:
        item_copy = {k: v for k, v in item.items() if k not in excluded_keys}
        batch_copy.append(item_copy)
    
    # Collate the rest using default_collate
    collated_batch = default_collate(batch_copy)
    
    # Add the batched point clouds
    collated_batch['point_cloud'] = batched_point_clouds
    
    return collated_batch

def eval_collate_fn(batch, dataset, num_sample_points):
    """
    Custom collate function for evaluation (alias for collate_fn with simplified excluded keys).
    """
    return collate_fn(batch, dataset, num_sample_points)


def clean_category_string(category, default_fallback="object"):
    """
    Clean and normalize category strings for use in embeddings and visualizations.
    
    Args:
        category: The input category, can be string, tuple, list, or None
        default_fallback: Default string to use when category is invalid or empty
        
    Returns:
        str: Cleaned category string
    """
    # Handle category extraction properly
    if isinstance(category, (tuple, list)):
        # Extract string from tuple/list if necessary
        clean_cat = category[0] if len(category) > 0 else default_fallback
    else:
        clean_cat = category
    
    # Convert to string and validate
    clean_cat = str(clean_cat) if clean_cat is not None else default_fallback
    
    # Clean the category string - remove/replace special characters
    clean_cat = (clean_cat
                 .replace(' ', '_')
                 .replace('-', '_')
                 .replace('(', '')
                 .replace(')', '')
                 .replace("'", '')
                 .replace('"', '')
                 .replace(',', '')
                 .replace('[', '')
                 .replace(']', '')
                 .replace('{', '')
                 .replace('}', ''))
    
    # Ensure it's not empty and handle special cases
    if (not clean_cat or 
        clean_cat.isspace() or 
        clean_cat == '' or 
        clean_cat == 'unknown' or 
        clean_cat == '<unconditional>'):
        clean_cat = default_fallback
    
    return clean_cat

def process_category_list(categories, default_fallback="object"):
    """
    Process a list of categories, cleaning each one.
    
    Args:
        categories: List of categories (can contain tuples, strings, etc.)
        default_fallback: Default string to use for invalid categories
        
    Returns:
        List[str]: List of cleaned category strings
    """
    if categories is None:
        return []
    
    if not isinstance(categories, list):
        # If it's a single category, wrap it in a list
        categories = [categories]
    
    processed_categories = []
    for cat in categories:
        clean_cat = clean_category_string(cat, default_fallback)
        processed_categories.append(clean_cat)
    
    return processed_categories
