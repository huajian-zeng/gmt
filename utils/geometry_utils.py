#!/usr/bin/env python3
"""
Geometry utilities for 6D rotation representations and coordinate transformations.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def rotation_matrix_to_6d(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to 6D representation (first two columns).
    
    Args:
        rotation_matrix: 3x3 rotation matrix (numpy array)
        
    Returns:
        ndarray: 6D rotation representation [rx1, ry1, rz1, rx2, ry2, rz2]
    """
    # Extract first two columns of rotation matrix
    col1 = rotation_matrix[:, 0]  # First column
    col2 = rotation_matrix[:, 1]  # Second column
    
    # Concatenate into 6D representation
    r6d = np.concatenate([col1, col2])
    
    return r6d


def rotation_6d_to_matrix(r6d):
    """
    Convert 6D representation to rotation matrix using Gram-Schmidt process.
    
    Args:
        r6d: 6D rotation representation [rx1, ry1, rz1, rx2, ry2, rz2]
        
    Returns:
        ndarray: 3x3 rotation matrix
    """
    # Extract two columns
    x = r6d[:3]  # First column
    y = r6d[3:]  # Second column
    
    # Normalize first column
    x = x / np.linalg.norm(x)
    
    # Make second column orthogonal to first using Gram-Schmidt
    t = y - np.dot(x, y) * x
    y = t / np.linalg.norm(t)
    
    # Third column is cross product
    z = np.cross(x, y)
    
    # Stack to create rotation matrix
    return np.column_stack([x, y, z])


def rotation_6d_to_matrix_torch(r6d):
    """
    PyTorch version: Convert 6D representation to rotation matrix.
    Numerically stable version with epsilon to prevent division by zero.
    
    Args:
        r6d: 6D rotation representation tensor [..., 6]
        
    Returns:
        torch.Tensor: rotation matrix tensor [..., 3, 3]
    """
    eps = 1e-8  # Small epsilon to prevent division by zero
    
    # Extract two columns
    x = r6d[..., :3]  # First column
    y = r6d[..., 3:]  # Second column
    
    # Normalize first column with epsilon for numerical stability
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    x = x / torch.clamp(x_norm, min=eps)
    
    # Make second column orthogonal to first using Gram-Schmidt
    t = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    t_norm = torch.norm(t, dim=-1, keepdim=True)
    
    # Handle degenerate case where t becomes very small (vectors nearly parallel)
    # In this case, create a random orthogonal vector to x
    mask = t_norm.squeeze(-1) < eps * 10  # More conservative threshold
    if mask.any():
        # Create orthogonal vector by swapping and negating components
        orthogonal = torch.zeros_like(x)
        orthogonal[..., 0] = -x[..., 1]  
        orthogonal[..., 1] = x[..., 0]
        orthogonal[..., 2] = 0
        # If x is close to z-axis, use different approach
        z_axis_mask = torch.norm(x[..., :2], dim=-1) < eps
        orthogonal[z_axis_mask, 0] = 1
        orthogonal[z_axis_mask, 1] = 0
        orthogonal[z_axis_mask, 2] = 0
        
        # Normalize the orthogonal vector
        orthogonal = orthogonal / torch.clamp(torch.norm(orthogonal, dim=-1, keepdim=True), min=eps)
        
        # Replace degenerate t with orthogonal vector
        t[mask] = orthogonal[mask]
        t_norm = torch.norm(t, dim=-1, keepdim=True)
    
    y = t / torch.clamp(t_norm, min=eps)
    
    # Third column is cross product
    z = torch.cross(x, y, dim=-1)
    
    # Ensure z is also normalized (should be automatically, but for safety)
    z = z / torch.clamp(torch.norm(z, dim=-1, keepdim=True), min=eps)
    
    # Stack to create rotation matrix
    rotation_matrix = torch.stack([x, y, z], dim=-1)
    
    # Final check: replace any remaining NaN with identity matrix
    nan_mask = torch.isnan(rotation_matrix).any(dim=(-2, -1))
    if nan_mask.any():
        identity = torch.eye(3, device=rotation_matrix.device, dtype=rotation_matrix.dtype)
        identity = identity.expand_as(rotation_matrix)
        rotation_matrix[nan_mask] = identity[nan_mask]
    
    return rotation_matrix


def rotation_matrix_to_6d_torch(rotation_matrix):
    """
    PyTorch version: Convert rotation matrix to 6D representation.
    
    Args:
        rotation_matrix: rotation matrix tensor [..., 3, 3]
        
    Returns:
        torch.Tensor: 6D rotation representation tensor [..., 6]
    """
    # Extract first two columns
    col1 = rotation_matrix[..., :, 0]  # First column
    col2 = rotation_matrix[..., :, 1]  # Second column
    
    # Concatenate into 6D representation
    return torch.cat([col1, col2], dim=-1)


def convert_rotation_to_euler(rotations):
    """
    Convert rotations to Euler angles for visualization.
    
    Args:
        rotations: Rotation data - either [N, 3] (already Euler) or [N, 6] (6D representation)
        
    Returns:
        ndarray: Euler angles [N, 3] (roll, pitch, yaw) or None if conversion failed
    """
    if rotations is None:
        return None
    
    if isinstance(rotations, torch.Tensor):
        rotations = rotations.detach().cpu().numpy()
    
    if rotations.shape[-1] == 3:
        # Already Euler angles
        return rotations
    elif rotations.shape[-1] == 6:
        # Convert 6D to Euler
        result = []
        for i in range(rotations.shape[0]):
            r6d = rotations[i]
            # Extract first two columns
            x = r6d[:3]
            y = r6d[3:]
            
            # Normalize and orthogonalize
            x = x / (np.linalg.norm(x) + 1e-8)
            y = y - np.dot(x, y) * x
            y = y / (np.linalg.norm(y) + 1e-8)
            z = np.cross(x, y)
            
            # Create rotation matrix
            R_matrix = np.column_stack([x, y, z])
            
            # Convert to Euler angles (simplified)
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
            
            result.append([roll, pitch, yaw])
        return np.array(result)
    else:
        print(f"Warning: Unexpected rotation dimension {rotations.shape[-1]}. Expected 3 or 6.")
        return None
