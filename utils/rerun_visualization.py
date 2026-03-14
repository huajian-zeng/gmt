#!/usr/bin/env python3
"""
Rerun Visualization Utilities for ADT
--------------------------------------
This module provides functions to visualize trajectory data using Rerun.
It supports visualizing past trajectories, ground truth futures, predictions,
and point clouds.
"""

import os
import numpy as np
import torch
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Union, Any

# Import coordinate transformation function from metrics_utils
from .metrics_utils import (
    transform_coords_for_visualization_numpy,
    transform_coords_for_visualization,
    clean_category_string
)

# Try to import rerun - if not available, disable functionality
try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    print("Warning: rerun-sdk not found. Rerun visualization will be disabled.")
    HAS_RERUN = False
    rr = None

def initialize_rerun(recording_name="adt_evaluation", spawn=True, output_dir=None) -> Tuple[bool, Optional[str]]:
    """
    Initialize Rerun for visualization.

    Args:
        recording_name: Name of the Rerun recording. This will be used as the application ID.
        spawn: Whether to spawn a new Rerun viewer. For evaluation, this is often set to False
               if the goal is to save the recording to a file without viewing it immediately.
        output_dir: Directory to save .rrd files. If provided, the output path is returned
                    for later saving after data is logged.

    Returns:
        Tuple[bool, Optional[str]]:
            - bool: Whether Rerun was successfully initialized.
            - Optional[str]: The target path for the .rrd file if output_dir was provided, else None.
    """
    if not HAS_RERUN:
        print("Warning: rerun module not available. Cannot initialize.")
        return False, None

    out_path: Optional[str] = None

    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{recording_name}.rrd")
            # Only initialize rerun here, save() must be called AFTER logging data
            rr.init(recording_name, spawn=spawn, recording_id=recording_name)
            print(f"Initialized Rerun with recording name: {recording_name}. Will save to {out_path}")
            return True, out_path
        else:
            rr.init(recording_name, spawn=spawn, recording_id=recording_name)
            return True, None

    except Exception as e:
        print(f"Error initializing Rerun: {e}")
        return False, out_path

def downsample_point_cloud(point_cloud, downsample_factor=1000):
    """
    Downsample a point cloud by selecting every nth point.
    
    Args:
        point_cloud: Point cloud data of shape [N, 3] or [N, 4] if it includes intensity
        downsample_factor: Factor by which to downsample (1 = no downsampling)
        
    Returns:
        Downsampled point cloud
    """
    if downsample_factor <= 1:
        return point_cloud
    
    if point_cloud.shape[0] > 1000000:
        num_samples = point_cloud.shape[0] // downsample_factor
        indices = np.random.choice(point_cloud.shape[0], size=num_samples, replace=False)
        return point_cloud[indices]
    
    return point_cloud[::downsample_factor]

def extract_trajectory_specific_point_cloud(point_cloud, trajectory, radius=1.0, max_points=100000):
    """
    Extract points from a point cloud that are near a trajectory.
    
    Args:
        point_cloud: Point cloud data of shape [N, 3] or [N, 4] if it includes intensity
        trajectory: Trajectory positions of shape [T, 3]
        radius: Radius around trajectory points to include
        max_points: Maximum number of points to return
        
    Returns:
        Point cloud data near the trajectory
    """
    if point_cloud is None or point_cloud.shape[0] == 0 or trajectory.shape[0] == 0:
        return None
    
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    
    pc_positions = point_cloud[:, :3]
    
    near_points_indices = []
    for traj_point in trajectory:
        distances = np.linalg.norm(pc_positions - traj_point, axis=1)
        near_indices = np.where(distances < radius)[0]
        near_points_indices.extend(near_indices)
    
    near_points_indices = list(set(near_points_indices))
    
    if len(near_points_indices) > max_points:
        near_points_indices = np.random.choice(near_points_indices, size=max_points, replace=False)
    
    return point_cloud[near_points_indices]

def convert_6d_rotation_to_direction(rotation_6d, arrow_length=0.3):
    """
    Convert 6D rotation representation to direction vectors for arrow visualization.
    
    Args:
        rotation_6d: 6D rotation representation [6] (first two columns of rotation matrix)
        arrow_length: Length of the direction arrow
        
    Returns:
        direction_vector: [3] direction vector for the arrow
    """
    if rotation_6d is None or len(rotation_6d) < 6:
        # Default forward direction
        return np.array([arrow_length, 0.0, 0.0])
    
    try:
        # Convert 6D rotation to rotation matrix
        # 6D rotation: [a1, a2, a3, b1, b2, b3] where a and b are first two columns
        a = rotation_6d[:3]
        b = rotation_6d[3:6]
        
        # Normalize a
        a = a / (np.linalg.norm(a) + 1e-8)
        
        # Make b orthogonal to a
        b = b - np.dot(b, a) * a
        b = b / (np.linalg.norm(b) + 1e-8)
        
        # Get the third column by cross product
        c = np.cross(a, b)
        
        # Forward direction (use first column as forward)
        forward_dir = a * arrow_length
        
        return forward_dir
        
    except Exception as e:
        print(f"Error converting 6D rotation to direction: {e}")
        # Return default forward direction
        return np.array([arrow_length, 0.0, 0.0])

def get_category_color(category):
    """
    Get a color for a specific object category.
    
    Args:
        category: String category name (cleaned for Rerun paths)
        
    Returns:
        List[float]: RGBA color values [r, g, b, a] in range [0, 1]
    """
    # Define a color mapping for common categories
    category_colors = {
        # Furniture
        'chair': [0.8, 0.4, 0.2, 0.8],      # Brown
        'table': [0.6, 0.3, 0.1, 0.8],     # Dark brown
        'dining_table': [0.6, 0.3, 0.1, 0.8],  # Dark brown
        'desk': [0.5, 0.3, 0.1, 0.8],      # Dark brown
        'sofa': [0.2, 0.6, 0.8, 0.8],      # Blue
        'bed': [0.7, 0.2, 0.7, 0.8],       # Purple
        'cabinet': [0.4, 0.6, 0.2, 0.8],   # Green
        'shelf': [0.3, 0.5, 0.3, 0.8],     # Dark green
        'bookshelf': [0.3, 0.5, 0.3, 0.8], # Dark green
        
        # Electronics
        'tv': [0.1, 0.1, 0.1, 0.8],        # Black
        'monitor': [0.2, 0.2, 0.2, 0.8],   # Dark gray
        'computer': [0.3, 0.3, 0.3, 0.8],  # Gray
        'laptop': [0.4, 0.4, 0.4, 0.8],    # Light gray
        
        # Kitchen items
        'refrigerator': [0.9, 0.9, 0.9, 0.8],  # White
        'microwave': [0.8, 0.8, 0.8, 0.8],     # Light gray
        'oven': [0.2, 0.2, 0.2, 0.8],          # Black
        'cutting_board': [0.7, 0.5, 0.3, 0.8], # Wood brown
        
        # Food and kitchen tools
        'food_object': [0.3, 1.0, 0.3, 0.8],   # Bright green
        'food': [0.3, 1.0, 0.3, 0.8],          # Bright green
        'fork': [0.7, 0.7, 0.7, 0.8],          # Silver
        'spoon': [0.7, 0.7, 0.7, 0.8],         # Silver
        'knife': [0.6, 0.6, 0.6, 0.8],         # Dark silver
        
        # Containers and bottles
        'container': [0.3, 0.7, 0.9, 0.8],     # Light blue
        'bottle': [0.1, 0.3, 0.6, 0.8],        # Dark blue
        'can': [0.5, 0.5, 0.5, 0.8],           # Gray
        'vase': [0.8, 0.6, 0.4, 0.8],          # Beige
        'coaster': [0.6, 0.4, 0.2, 0.8],       # Brown
        
        # Plants and decorations
        'plant': [0.2, 0.8, 0.2, 0.8],     # Green
        'flower': [1.0, 0.6, 0.8, 0.8],    # Pink
        
        # Storage
        'box': [0.7, 0.5, 0.3, 0.8],       # Cardboard brown
        'bag': [0.6, 0.6, 0.6, 0.8],       # Gray
        'basket': [0.8, 0.7, 0.4, 0.8],    # Wicker color
        
        # Lighting
        'lamp': [1.0, 1.0, 0.6, 0.8],      # Light yellow
        'light': [1.0, 1.0, 0.8, 0.8],     # Warm white
        
        # Other common objects
        'door': [0.6, 0.4, 0.2, 0.8],      # Wood brown
        'window': [0.8, 0.9, 1.0, 0.6],    # Light blue, transparent
        'wall': [0.9, 0.9, 0.9, 0.4],      # Light gray, transparent
        'floor': [0.7, 0.6, 0.5, 0.3],     # Floor color, transparent
        
        # Default/unknown
        'unknown': [0.8, 0.8, 0.2, 0.7],   # Yellow (original default)
    }
    
    # Normalize category name (lowercase, already cleaned)
    normalized_category = category.lower()
    
    # Check for exact match first
    if normalized_category in category_colors:
        return category_colors[normalized_category]
    
    # Check for partial matches
    for cat_key, color in category_colors.items():
        if normalized_category in cat_key or cat_key in normalized_category:
            return color
    
    # If no match found, generate a hash-based color for consistency
    import hashlib
    hash_value = int(hashlib.md5(category.encode()).hexdigest()[:8], 16)
    
    # Generate RGB values from hash
    r = ((hash_value >> 16) & 0xFF) / 255.0
    g = ((hash_value >> 8) & 0xFF) / 255.0
    b = (hash_value & 0xFF) / 255.0
    
    # Ensure minimum brightness and alpha
    r = max(0.3, r)
    g = max(0.3, g) 
    b = max(0.3, b)
    
    return [r, g, b, 0.8]

# transform_coords_for_visualization_numpy is now imported from metrics_utils

def create_bbox_corners_edges(bbox_corners, coord_mapping=None):
    """
    Create edges for moving object bounding boxes from corner points.
    
    Args:
        bbox_corners: [8, 3] array containing the 8 corner points of a bounding box
        coord_mapping: Coordinate mapping for visualization, defaults to ['x', '-z', 'y'] if None
        
    Returns:
        List of bbox edges for visualization (after coordinate transformation)
    """
    if bbox_corners is None or bbox_corners.shape[0] != 8:
        return []
    
    # Apply coordinate transformation for visualization
    corners_transformed = transform_coords_for_visualization_numpy(bbox_corners, coord_mapping)
    
    # Define the 12 edges of the box (same structure as semantic bboxes)
    edge_indices = [
        # Bottom face edges
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face edges  
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    bbox_edges = []
    for start_idx, end_idx in edge_indices:
        edge = [corners_transformed[start_idx].tolist(), corners_transformed[end_idx].tolist()]
        bbox_edges.append(edge)
    
    return bbox_edges

def create_semantic_bbox_edges(bbox_info, coord_mapping=None):
    """
    Create edges for semantic bounding boxes from bbox info.
    
    Args:
        bbox_info: [N, 12] array where each row contains bbox info:
                  [center_x, center_y, center_z, size_x, size_y, size_z, 
                   rot_6d_0, rot_6d_1, rot_6d_2, rot_6d_3, rot_6d_4, rot_6d_5]
                  The last 6 dimensions represent 6D rotation (first two columns of rotation matrix)
        coord_mapping: Coordinate mapping for visualization, defaults to ['x', '-z', 'y'] if None
        
    Returns:
        List of bbox edges for visualization (after coordinate transformation)
    """
    if bbox_info is None or bbox_info.shape[0] == 0:
        return []
    
    all_edges = []
    
    for i in range(bbox_info.shape[0]):
        # Extract center and size from the first 6 dimensions
        center = bbox_info[i, :3]  # [center_x, center_y, center_z]
        size = bbox_info[i, 3:6]   # [size_x, size_y, size_z]
        
        # Extract rotation information if available (6D rotation representation)
        if bbox_info.shape[1] >= 12:
            rotation_6d = bbox_info[i, 6:12]  # [rot_6d_0, rot_6d_1, rot_6d_2, rot_6d_3, rot_6d_4, rot_6d_5]
            
            # Convert 6D rotation to rotation matrix
            rotation_matrix = convert_6d_to_rotation_matrix(rotation_6d)
        else:
            # If no rotation info, use identity matrix (no rotation)
            rotation_matrix = np.eye(3)
        
        # Create 8 corners of the bounding box in object-local coordinate system
        # (relative to the bbox center, before rotation)
        half_size = size / 2
        local_corners = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],  # bottom back left
            [+half_size[0], -half_size[1], -half_size[2]],  # bottom back right
            [+half_size[0], +half_size[1], -half_size[2]],  # bottom front right
            [-half_size[0], +half_size[1], -half_size[2]],  # bottom front left
            [-half_size[0], -half_size[1], +half_size[2]],  # top back left
            [+half_size[0], -half_size[1], +half_size[2]],  # top back right
            [+half_size[0], +half_size[1], +half_size[2]],  # top front right
            [-half_size[0], +half_size[1], +half_size[2]],  # top front left
        ])
        
        # Apply rotation to each corner
        rotated_corners = np.zeros_like(local_corners)
        for j in range(8):
            rotated_corners[j] = rotation_matrix @ local_corners[j]
        
        # Translate to world position (add center)
        world_corners = rotated_corners + center
        
        # Apply coordinate transformation for visualization
        corners_transformed = transform_coords_for_visualization_numpy(world_corners, coord_mapping)
        
        # Define the 12 edges of the box
        edge_indices = [
            # Bottom face edges
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face edges  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        bbox_edges = []
        for start_idx, end_idx in edge_indices:
            edge = [corners_transformed[start_idx].tolist(), corners_transformed[end_idx].tolist()]
            bbox_edges.append(edge)
        
        all_edges.extend(bbox_edges)
    
    return all_edges

def convert_6d_to_rotation_matrix(rotation_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        rotation_6d: [6] array representing the first two columns of a rotation matrix
                    [a1, a2, a3, b1, b2, b3] where a and b are the first two columns
        
    Returns:
        np.ndarray: [3, 3] rotation matrix
    """
    if rotation_6d is None or len(rotation_6d) < 6:
        return np.eye(3)
    
    try:
        # Extract first two columns
        a = rotation_6d[:3]  # First column
        b = rotation_6d[3:6]  # Second column
        
        # Normalize first column
        a = a / (np.linalg.norm(a) + 1e-8)
        
        # Make second column orthogonal to first column using Gram-Schmidt
        b = b - np.dot(b, a) * a
        b = b / (np.linalg.norm(b) + 1e-8)
        
        # Compute third column as cross product
        c = np.cross(a, b)
        c = c / (np.linalg.norm(c) + 1e-8)  # Normalize for safety
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack((a, b, c))
        
        # Ensure it's a proper rotation matrix (det = 1)
        if np.linalg.det(rotation_matrix) < 0:
            # If determinant is negative, flip the third column
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
        
        return rotation_matrix
        
    except Exception as e:
        print(f"Error converting 6D rotation to matrix: {e}")
        return np.eye(3)  # Return identity matrix as fallback

def visualize_trajectory_rerun(
    past_positions, 
    future_positions_gt=None, 
    future_positions_pred=None,
    past_mask=None, 
    future_mask_gt=None,
    point_cloud=None,
    past_orientations=None,
    future_orientations_gt=None,
    future_orientations_pred=None,
    semantic_bbox_info=None,
    semantic_bbox_mask=None,
    semantic_bbox_categories=None,
    bbox_corners=None,  # Add bbox_corners parameter
    object_name="object",
    sequence_name="sequence",
    segment_idx=None,
    method_name="baseline",
    pred_color=None,
    arrow_length=0.3,
    line_width=0.02,
    point_size=0.03,
    show_arrows=True,
    show_semantic_bboxes=True,
    show_bbox_corners=True,  # Add parameter to control bbox_corners visibility
    coord_mapping=None
):
    """
    Visualize trajectory data using Rerun.
    
    Args:
        past_positions: Past trajectory positions [history_length, 3] (already transformed for visualization)
        future_positions_gt: Ground truth future positions [future_length, 3] (already transformed for visualization)
        future_positions_pred: Predicted future positions [future_length, 3] (already transformed for visualization)
        past_mask: Optional mask for past positions [history_length]
        future_mask_gt: Optional mask for ground truth future positions [future_length]
        point_cloud: Optional point cloud data [N, 3] or [N, 4] (already transformed for visualization)
        past_orientations: Past orientations [history_length, 6] (6D rotation representation, not transformed)
        future_orientations_gt: Ground truth future orientations [future_length, 6] (6D rotation representation, not transformed)
        future_orientations_pred: Predicted future orientations [future_length, 6] (6D rotation representation, not transformed)
        semantic_bbox_info: Semantic bounding box information [N, 12] in original coordinates
                           Format: [center_x, center_y, center_z, size_x, size_y, size_z, ...]
                           Will be transformed internally after constructing bbox geometry
        semantic_bbox_mask: Semantic bounding box mask [N]
        semantic_bbox_categories: List of category names for each semantic bbox [N]
        bbox_corners: Moving object bounding box corners [trajectory_length, 8, 3] in original coordinates
                     Only displayed for past trajectory (input condition). Will be transformed internally for visualization
        object_name: Name/ID of the object, used to create a unique path in Rerun.
        sequence_name: Name of the sequence, used in the Rerun path.
        segment_idx: Optional segment index, used in the Rerun path.
        method_name: Name of the method (e.g., "baseline", "our"), used in the Rerun path for distinction.
        pred_color: Color for predicted trajectory. Can be a predefined color name string or RGBA list.
                   Available predefined colors: 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow'.
                   If None, defaults to 'red'.
        arrow_length: Length of orientation arrows.
        line_width: Width of trajectory lines.
        point_size: Size of trajectory points.
        show_arrows: Whether to show orientation arrows.
        show_semantic_bboxes: Whether to show semantic bounding boxes.
        show_bbox_corners: Whether to show moving object bounding box corners.
        coord_mapping: Coordinate mapping for visualization transformations.
    """
    if not HAS_RERUN:
        return False
    
    # Predefined colors for predicted trajectories
    PREDEFINED_COLORS = {
        'red': {
            'path_color': [1.0, 0.6, 0.6, 0.7],    # Lighter Red for path
            'point_color': [1.0, 0.3, 0.3, 0.5],   # Solid Red for points
            'arrow_color': [1.0, 0.3, 0.3, 0.8]    # Red for arrows
        },
        'orange': {
            'path_color': [1.0, 0.8, 0.4, 0.7],    # Lighter Orange for path
            'point_color': [1.0, 0.6, 0.2, 0.5],   # Solid Orange for points
            'arrow_color': [1.0, 0.6, 0.2, 0.8]    # Orange for arrows
        },
        'purple': {
            'path_color': [0.8, 0.6, 1.0, 0.7],    # Lighter Purple for path
            'point_color': [0.6, 0.3, 1.0, 0.5],   # Solid Purple for points
            'arrow_color': [0.6, 0.3, 1.0, 0.8]    # Purple for arrows
        },
        'cyan': {
            'path_color': [0.4, 0.9, 1.0, 0.7],    # Lighter Cyan for path
            'point_color': [0.2, 0.8, 1.0, 0.5],   # Solid Cyan for points
            'arrow_color': [0.2, 0.8, 1.0, 0.8]    # Cyan for arrows
        },
        'magenta': {
            'path_color': [1.0, 0.6, 0.8, 0.7],    # Lighter Magenta for path
            'point_color': [1.0, 0.3, 0.6, 0.5],   # Solid Magenta for points
            'arrow_color': [1.0, 0.3, 0.6, 0.8]    # Magenta for arrows
        },
        'yellow': {
            'path_color': [1.0, 1.0, 0.6, 0.7],    # Lighter Yellow for path
            'point_color': [1.0, 0.9, 0.3, 0.5],   # Solid Yellow for points
            'arrow_color': [1.0, 0.9, 0.3, 0.8]    # Yellow for arrows
        }
    }
    
    # Determine prediction colors
    if pred_color is None:
        # Default to red
        pred_colors = PREDEFINED_COLORS['red']
    elif isinstance(pred_color, str) and pred_color.lower() in PREDEFINED_COLORS:
        # Use predefined color
        pred_colors = PREDEFINED_COLORS[pred_color.lower()]
    elif isinstance(pred_color, (list, tuple)) and len(pred_color) >= 3:
        # Custom RGBA color provided
        if len(pred_color) == 3:
            pred_color = list(pred_color) + [0.7]  # Add alpha if not provided
        pred_colors = {
            'path_color': [pred_color[0], pred_color[1], pred_color[2], pred_color[3] * 0.9],  # Slightly more transparent for path
            'point_color': [pred_color[0] * 0.8, pred_color[1] * 0.8, pred_color[2] * 0.8, pred_color[3] * 0.7],  # Darker for points
            'arrow_color': [pred_color[0] * 0.8, pred_color[1] * 0.8, pred_color[2] * 0.8, pred_color[3]]  # Darker for arrows
        }
    else:
        # Fallback to red if invalid color provided
        print(f"Warning: Invalid pred_color '{pred_color}', falling back to red")
        pred_colors = PREDEFINED_COLORS['red']
    
    # Sanitize path components to remove characters that are problematic for Rerun.
    # This specifically addresses warnings about unescaped whitespace.
    safe_object_name = str(object_name).replace(" ", "_").replace("/", "-")
    safe_sequence_name = str(sequence_name).replace(" ", "_").replace("/", "-")
    safe_method_name = str(method_name).replace(" ", "_").replace("/", "-")
    
    vis_path = f"trajectories/{safe_sequence_name}"
    if segment_idx is not None:
        vis_path += f"/segment_{segment_idx}"
    vis_path += f"/{safe_method_name}/{safe_object_name}"
    
    # Convert all inputs to numpy arrays
    if isinstance(past_positions, torch.Tensor):
        past_positions = past_positions.detach().cpu().numpy()
    if isinstance(future_positions_gt, torch.Tensor):
        future_positions_gt = future_positions_gt.detach().cpu().numpy()
    if isinstance(future_positions_pred, torch.Tensor):
        future_positions_pred = future_positions_pred.detach().cpu().numpy()
    if isinstance(past_mask, torch.Tensor):
        past_mask = past_mask.detach().cpu().numpy()
    if isinstance(future_mask_gt, torch.Tensor):
        future_mask_gt = future_mask_gt.detach().cpu().numpy()
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()

    if isinstance(past_orientations, torch.Tensor):
        past_orientations = past_orientations.detach().cpu().numpy()
    if isinstance(future_orientations_gt, torch.Tensor):
        future_orientations_gt = future_orientations_gt.detach().cpu().numpy()
    if isinstance(future_orientations_pred, torch.Tensor):
        future_orientations_pred = future_orientations_pred.detach().cpu().numpy()
    
    # Convert bbox corners to numpy
    if isinstance(bbox_corners, torch.Tensor):
        bbox_corners = bbox_corners.detach().cpu().numpy()

    # Convert bbox corners to numpy
    if isinstance(semantic_bbox_info, torch.Tensor):
        semantic_bbox_info = semantic_bbox_info.detach().cpu().numpy()
    if isinstance(semantic_bbox_mask, torch.Tensor):
        semantic_bbox_mask = semantic_bbox_mask.detach().cpu().numpy()

    if past_mask is not None:
        valid_past_indices = np.where(past_mask > 0.5)[0]
        if len(valid_past_indices) > 0:
            past_positions_valid = past_positions[valid_past_indices]
            past_orientations_valid = past_orientations[valid_past_indices] if past_orientations is not None else None
        else:
            past_positions_valid = np.empty((0, 3))
            past_orientations_valid = np.empty((0, 6)) if past_orientations is not None else None
    else:
        past_positions_valid = past_positions
        past_orientations_valid = past_orientations

    if future_positions_gt is not None and future_mask_gt is not None:
        valid_future_gt_indices = np.where(future_mask_gt > 0.5)[0]
        if len(valid_future_gt_indices) > 0:
            future_positions_gt_valid = future_positions_gt[valid_future_gt_indices]
            future_orientations_gt_valid = future_orientations_gt[valid_future_gt_indices] if future_orientations_gt is not None else None
        else:
            future_positions_gt_valid = np.empty((0, 3))
            future_orientations_gt_valid = np.empty((0, 6)) if future_orientations_gt is not None else None
    else:
        future_positions_gt_valid = future_positions_gt
        future_orientations_gt_valid = future_orientations_gt

    future_positions_pred_valid = future_positions_pred
    future_orientations_pred_valid = future_orientations_pred
    
    # --- Determine endpose (last valid point of GT trajectory) ---
    endpose_position = None
    endpose_orientation = None
    
    if future_positions_gt_valid is not None and future_positions_gt_valid.shape[0] > 0:
        # Use the last point of GT future trajectory as endpose
        endpose_position = future_positions_gt_valid[-1]  # [3]
        if future_orientations_gt_valid is not None and future_orientations_gt_valid.shape[0] > 0:
            endpose_orientation = future_orientations_gt_valid[-1]  # [6]
    elif past_positions_valid.shape[0] > 0:
        # If no future GT, use the last point of past trajectory as endpose
        endpose_position = past_positions_valid[-1]  # [3]
        if past_orientations_valid is not None and past_orientations_valid.shape[0] > 0:
            endpose_orientation = past_orientations_valid[-1]  # [6]
    
    try:
        current_animation_step = 0 # Initialize animation_step counter first

        # Log point cloud once, at the first step of the "animation_step" timeline
        if point_cloud is not None and len(point_cloud) > 0:
            rr.set_time_sequence("animation_step", current_animation_step) # Ensure PC is on the animation timeline
            point_cloud_colors = np.ones((point_cloud.shape[0], 4), dtype=np.float32) * [0.7, 0.7, 0.7, 0.3]
            try:
                rr.log(f"{vis_path}/point_cloud", 
                      rr.Points3D(
                          positions=point_cloud[:, :3],
                          colors=point_cloud_colors,
                          radii=0.002 # Consider making this configurable or relative to scene scale
                      ))
            except Exception as e:
                print(f"Error logging point cloud: {e}")
        
        # Log semantic bounding boxes once (they are static scene elements)
        if show_semantic_bboxes and semantic_bbox_info is not None:
            # Filter valid semantic bboxes using the mask
            if semantic_bbox_mask is not None:
                valid_semantic_indices = np.where(semantic_bbox_mask > 0.5)[0]
                if len(valid_semantic_indices) > 0:
                    valid_semantic_bbox_info = semantic_bbox_info[valid_semantic_indices]
                    # Also filter categories if available
                    if semantic_bbox_categories is not None:
                        valid_semantic_categories = [semantic_bbox_categories[i] for i in valid_semantic_indices]
                    else:
                        valid_semantic_categories = ["unknown"] * len(valid_semantic_indices)
                else:
                    valid_semantic_bbox_info = None
                    valid_semantic_categories = []
            else:
                valid_semantic_bbox_info = semantic_bbox_info
                valid_semantic_categories = semantic_bbox_categories if semantic_bbox_categories is not None else ["unknown"] * len(semantic_bbox_info)
            
            if valid_semantic_bbox_info is not None and valid_semantic_bbox_info.shape[0] > 0:
                rr.set_time_sequence("animation_step", current_animation_step)
                
                print(f"Creating {valid_semantic_bbox_info.shape[0]} semantic bbox objects:")
                
                # Create individual objects for each semantic bbox
                for bbox_idx in range(valid_semantic_bbox_info.shape[0]):
                    bbox_info_single = valid_semantic_bbox_info[bbox_idx:bbox_idx+1]  # [1, 12]
                    
                    # Get category for this bbox
                    if semantic_bbox_categories and bbox_idx < len(semantic_bbox_categories):
                        category_raw = semantic_bbox_categories[bbox_idx]
                        # Use the unified string processing function
                        bbox_category = clean_category_string(category_raw, default_fallback="unknown")
                    else:
                        bbox_category = "unknown"
                    
                    # Create edges for this single bbox
                    single_bbox_edges = create_semantic_bbox_edges(bbox_info_single, coord_mapping)
                    
                    if single_bbox_edges:
                        # Create unique object name using cleaned category and index
                        bbox_object_name = f"{bbox_category}_{bbox_idx}"
                        
                        # Choose color based on category
                        bbox_color = get_category_color(bbox_category)
                        
                        print(f"  - {bbox_object_name} (color: {[f'{c:.2f}' for c in bbox_color[:3]]}, {len(single_bbox_edges)} edges)")
                        
                        rr.log(f"{vis_path}/semantic_objects/{bbox_object_name}",
                               rr.LineStrips3D(
                                   single_bbox_edges,
                                   colors=[bbox_color] * len(single_bbox_edges),
                                   radii=line_width * 0.8
                               ))
        
        # Log static endpose visualization (appears immediately and stays visible)
        if endpose_position is not None:
            rr.set_time_sequence("animation_step", current_animation_step)
            
            # Log endpose as a large, bright sphere
            rr.log(f"{vis_path}/endpose/target_point",
                   rr.Points3D(
                       positions=[endpose_position],
                       colors=[[1.0, 0.8, 0.0, 1.0]],  # Bright gold/yellow color
                       radii=point_size * 3.0  # 3x larger than normal points
                   ))
            
            # # Add a text label for the endpose
            # rr.log(f"{vis_path}/endpose/label",
            #        rr.TextDocument(
            #            "ENDPOSE",
            #            media_type=rr.MediaType.MARKDOWN
            #        ),
            #        rr.Transform3D(translation=endpose_position + np.array([0.1, 0.1, 0.1])))
            
            # Add endpose orientation arrow if available
            if show_arrows and endpose_orientation is not None:
                endpose_direction = convert_6d_rotation_to_direction(endpose_orientation, arrow_length * 1.5)
                rr.log(f"{vis_path}/endpose/orientation_arrow",
                       rr.Arrows3D(
                           origins=[endpose_position],
                           vectors=[endpose_direction],
                           colors=[[1.0, 0.8, 0.0, 1.0]]  # Bright gold/yellow arrow
                       ))
            
            print(f"Highlighted endpose at position: {endpose_position}")
        
        # current_animation_step is still 0 here. 
        # Trajectory animations will start their loops using this current_animation_step = 0,
        # and then increment it inside their respective loops.
        
        # --- Animate Past Trajectory ---
        if past_positions_valid.shape[0] > 0:
            for t in range(past_positions_valid.shape[0]):
                rr.set_time_sequence("animation_step", current_animation_step)
                # Log growing past path (line)
                rr.log(f"{vis_path}/trajectory/past_path",
                       rr.LineStrips3D(
                           [past_positions_valid[0:t+1].tolist()],
                           colors=[[0.6, 0.6, 1, 0.7]], # Lighter Blue
                           radii=line_width
                       ))
                # Log growing past points
                rr.log(f"{vis_path}/trajectory/past_points",
                       rr.Points3D(
                           positions=past_positions_valid[0:t+1],
                           colors=[0.3, 0.3, 1, 0.5], 
                           radii=point_size
                       ))
                
                # Log orientation arrows for past trajectory
                if show_arrows and past_orientations_valid is not None and t < past_orientations_valid.shape[0]:
                    current_position = past_positions_valid[t]
                    current_orientation = past_orientations_valid[t]
                    direction_vector = convert_6d_rotation_to_direction(current_orientation, arrow_length)
                    arrow_end = current_position + direction_vector
                    
                    rr.log(f"{vis_path}/arrows/past_arrow_{t}",
                           rr.Arrows3D(
                               origins=[current_position],
                               vectors=[direction_vector],
                               colors=[0.3, 0.3, 1, 0.8]  # Blue arrows
                           ))
                
                # # Log moving object bounding box for past trajectory
                # if show_bbox_corners and bbox_corners is not None and t < bbox_corners.shape[0]:
                #     current_bbox_corners = bbox_corners[t]  # [8, 3]
                #     bbox_edges = create_bbox_corners_edges(current_bbox_corners, coord_mapping)
                #     if bbox_edges:
                #         rr.log(f"{vis_path}/moving_bbox/past_bbox_{t}",
                #                rr.LineStrips3D(
                #                    bbox_edges,
                #                    colors=[[0.0, 0.0, 0.5, 0.8]] * len(bbox_edges),  # Deep blue color
                #                    radii=line_width * 0.5 
                #                ))
                
                current_animation_step += 1
        
        # Define connection point from past trajectory
        connection_point_np = past_positions_valid[-1] if past_positions_valid.shape[0] > 0 else None

        # --- Animate Future Ground Truth Trajectory ---
        if future_positions_gt_valid is not None and future_positions_gt_valid.shape[0] > 0:
            # Path for GT trajectory, possibly connected to past
            gt_path_points_for_strip = []
            if connection_point_np is not None:
                gt_path_points_for_strip.append(connection_point_np.tolist())

            for t in range(future_positions_gt_valid.shape[0]):
                rr.set_time_sequence("animation_step", current_animation_step)
                
                # Keep past trajectory (line and points) visible
                if past_positions_valid.shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/past_path",
                           rr.LineStrips3D(
                               [past_positions_valid.tolist()],
                               colors=[[0.6, 0.6, 1, 0.7]], # Lighter Blue
                               radii=line_width
                           ))
                    rr.log(f"{vis_path}/trajectory/past_points",
                           rr.Points3D(
                               positions=past_positions_valid,
                               colors=[0.3, 0.3, 1, 0.5], 
                               radii=point_size
                           ))

                # Construct current segment of GT path (line)
                current_gt_segment_list = future_positions_gt_valid[0:t+1].tolist()
                
                if connection_point_np is not None:
                    points_to_log_gt = [connection_point_np.tolist()] + current_gt_segment_list
                else:
                    points_to_log_gt = current_gt_segment_list
                
                if points_to_log_gt: # Ensure there are points to log
                    rr.log(f"{vis_path}/trajectory/gt_path",
                           rr.LineStrips3D(
                               [points_to_log_gt],
                               colors=[[0.6, 1, 0.6, 0.7]], # Lighter Green
                               radii=line_width
                           ))
                
                # Log growing GT points
                if future_positions_gt_valid[0:t+1].shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/gt_points",
                           rr.Points3D(
                               positions=future_positions_gt_valid[0:t+1],
                               colors=[0.3, 1, 0.3, 0.5], # Solid Green
                               radii=point_size
                           ))
                
                # Log orientation arrows for GT future trajectory
                if show_arrows and future_orientations_gt_valid is not None and t < future_orientations_gt_valid.shape[0]:
                    current_position = future_positions_gt_valid[t]
                    current_orientation = future_orientations_gt_valid[t]
                    direction_vector = convert_6d_rotation_to_direction(current_orientation, arrow_length)
                    
                    rr.log(f"{vis_path}/arrows/gt_arrow_{t}",
                           rr.Arrows3D(
                               origins=[current_position],
                               vectors=[direction_vector],
                               colors=[0.3, 1, 0.3, 0.8]  # Green arrows
                           ))
                
                current_animation_step += 1
        
        # --- Animate Future Predicted Trajectory ---
        if future_positions_pred_valid is not None and future_positions_pred_valid.shape[0] > 0:
            # Path for Pred trajectory, possibly connected to past
            pred_path_points_for_strip = []
            if connection_point_np is not None:
                pred_path_points_for_strip.append(connection_point_np.tolist())

            # Determine full GT path if it existed, to keep it visible
            full_gt_path_list = []
            if future_positions_gt_valid is not None and future_positions_gt_valid.shape[0] > 0:
                if connection_point_np is not None:
                    full_gt_path_list = [connection_point_np.tolist()] + future_positions_gt_valid.tolist()
                else:
                    full_gt_path_list = future_positions_gt_valid.tolist()

            for t in range(future_positions_pred_valid.shape[0]):
                rr.set_time_sequence("animation_step", current_animation_step)

                # Keep past trajectory (line and points) visible
                if past_positions_valid.shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/past_path",
                           rr.LineStrips3D(
                               [past_positions_valid.tolist()],
                               colors=[[0.6, 0.6, 1, 0.7]], # Lighter Blue
                               radii=line_width
                           ))
                    rr.log(f"{vis_path}/trajectory/past_points",
                           rr.Points3D(
                               positions=past_positions_valid,
                               colors=[0.3, 0.3, 1, 0.5], # Solid Blue
                               radii=point_size
                           ))
                
                # Keep GT trajectory (line and points) visible if it existed
                if full_gt_path_list:
                     rr.log(f"{vis_path}/trajectory/gt_path",
                           rr.LineStrips3D(
                               [full_gt_path_list],
                               colors=[[0.6, 1, 0.6, 0.7]], # Lighter Green
                               radii=line_width
                           ))
                     if future_positions_gt_valid is not None and future_positions_gt_valid.shape[0] > 0:
                         rr.log(f"{vis_path}/trajectory/gt_points",
                                rr.Points3D(
                                    positions=future_positions_gt_valid,
                                    colors=[0.3, 1, 0.3, 0.5], # Solid Green
                                    radii=point_size
                                ))

                # Construct current segment of Pred path (line)
                current_pred_segment_list = future_positions_pred_valid[0:t+1].tolist()

                if connection_point_np is not None:
                    points_to_log_pred = [connection_point_np.tolist()] + current_pred_segment_list
                else:
                    points_to_log_pred = current_pred_segment_list
                
                if points_to_log_pred: # Ensure there are points to log
                    rr.log(f"{vis_path}/trajectory/pred_path",
                           rr.LineStrips3D(
                               [points_to_log_pred],
                               colors=[pred_colors['path_color']], # Use configurable prediction color
                               radii=line_width
                           ))
                
                # Log growing Pred points
                if future_positions_pred_valid[0:t+1].shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/pred_points",
                           rr.Points3D(
                               positions=future_positions_pred_valid[0:t+1],
                               colors=pred_colors['point_color'], # Use configurable prediction color
                               radii=point_size
                           ))
                
                # Log orientation arrows for predicted future trajectory
                if show_arrows and future_orientations_pred_valid is not None and t < future_orientations_pred_valid.shape[0]:
                    current_position = future_positions_pred_valid[t]
                    current_orientation = future_orientations_pred_valid[t]
                    direction_vector = convert_6d_rotation_to_direction(current_orientation, arrow_length)
                    
                    rr.log(f"{vis_path}/arrows/pred_arrow_{t}",
                           rr.Arrows3D(
                               origins=[current_position],
                               vectors=[direction_vector],
                               colors=pred_colors['arrow_color']  # Use configurable prediction color
                           ))
                
                current_animation_step += 1
            
    except Exception as e:
        print(f"Error in Rerun visualization: {e}")
        return False
        
    return True

def save_rerun_recording(output_path, vis_path=None):
    """
    Save a Rerun recording to a file.
    
    NOTE: In modern Rerun versions (e.g., 0.9.0+), if Rerun was initialized with the
    `to_file` parameter (as done in `initialize_rerun` when `output_dir` is provided),
    the recording is ALREADY being saved to the specified .rrd file. Calling `rr.save()`
    explicitly might be redundant or could even cause issues in such cases.
    This function is retained for potential backward compatibility or for scenarios
    where Rerun was initialized without `to_file` (e.g., in-memory recording) and
    now needs to be saved. The evaluation script relies on the `to_file`
    mechanism in `initialize_rerun` and does NOT call this `save_rerun_recording` function.
    
    Args:
        output_path: Path to save the recording (e.g., "my_recording.rrd").
        vis_path: Specific visualization path to save. If None, attempts to save
                  the entire current Rerun recording. This argument is often not
                  directly supported by `rr.save()` in the way one might expect
                  for partial saves; `rr.save()` typically saves the whole recording.
                  The Rerun CLI `save` command also saves the whole recording.
        
    Returns:
        bool: Whether the recording was successfully saved or likely already saved.
    """
    if not HAS_RERUN:
        print("Warning: rerun module not available. Cannot save recording.")
        return False
    
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory for output_path {output_path}: {e}")
        return False

    try:
        if hasattr(rr, 'save') and callable(rr.save):
            rr.save(output_path)
            if os.path.exists(output_path):
                return True
        else:
            print("rr.save() function not available in this Rerun version or context.")
            
    except Exception as e_py_save:
        print(f"Error or issue using Python rr.save('{output_path}'): {e_py_save}")
        print("This might be expected if 'to_file' was used during rr.init().")

    if os.path.exists(output_path):
        print(f"Confirmed: Rerun recording file {output_path} exists (likely saved during initialization).")
        return True
        
    print(f"Attempting to use Rerun command-line tool to save recording as a fallback for {output_path}...")
    try:
        rerun_cmd = "rerun"
        result = subprocess.run([rerun_cmd, "--version"], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"Found rerun CLI: {result.stdout.strip()}")
            save_args = [rerun_cmd, "save", output_path]
            if vis_path:
                print(f"Warning: 'vis_path' argument ('{vis_path}') is not directly used by 'rerun save' CLI command.")
            
            print(f"Executing Rerun CLI save: {' '.join(save_args)}")
            save_result = subprocess.run(save_args, capture_output=True, text=True, check=False)
            
            if save_result.returncode == 0:
                print(f"Successfully saved recording to {output_path} using rerun CLI (stdout: {save_result.stdout.strip()})")
                return True
            else:
                print(f"Error using rerun CLI 'save' (return code {save_result.returncode}):")
                print(f"  Stdout: {save_result.stdout.strip()}")
                print(f"  Stderr: {save_result.stderr.strip()}")
        else:
            print("rerun CLI not available or --version command returned error.")
            
    except FileNotFoundError:
        print("rerun CLI command not found. Ensure Rerun is installed and in PATH for CLI fallback.")
    except Exception as cmd_e:
        print(f"Error during Rerun CLI save attempt: {cmd_e}")
            
    if not os.path.exists(output_path):
        print(f"Warning: Could not save or confirm Rerun recording at {output_path} after all attempts.")
        print("If 'output_dir' was specified during 'initialize_rerun', the recording should have been saved there directly.")
        print(f"Please check if '{output_path}' (or a similar file in the init 'output_dir') was created during 'rr.init()'.")
        return False
    
    return os.path.exists(output_path) 