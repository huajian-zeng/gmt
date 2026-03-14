"""Dataset class for loading single Aria Digital Twin sequence."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import traceback
from typing import Optional
import pickle
import hashlib
import time
from scipy.spatial import cKDTree
import logging

from utils.geometry_utils import rotation_matrix_to_6d
from .clip_feature_manager import CLIPFeatureManager

logger = logging.getLogger(__name__)


class ADTSingleDataset(Dataset):
    """Dataset for loading trajectories from a single Aria Digital Twin sequence."""

    def __init__(
        self,
        sequence_path: str,
        config=None,  # Add config parameter
        trajectory_length: int = 100,
        skip_frames: int = 5,
        max_objects: Optional[int] = None,
        device_num: int = 0,
        transform=None,
        load_pointcloud: bool = True,
        pointcloud_subsample: int = 1,  # Default changed to 1 to use full resolution
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        min_motion_threshold: float = 0.0,  # Minimum motion threshold (in meters)
        min_motion_percentile: float = 0.0,  # Filter trajectories below this percentile of motion
        use_displacements: bool = False,  # Whether to use displacements instead of absolute positions
        detect_motion_segments: bool = True,  # Whether to detect and extract active motion segments
        motion_velocity_threshold: float = 0.05,  # Threshold in m/s for detecting active motion
        min_segment_frames: int = 5,  # Minimum number of frames for a valid motion segment
        max_stationary_frames: int = 3,   # Maximum consecutive stationary frames allowed in a motion segment
        normalize_data: bool = True,      # Whether to normalize data using scene bounds
        trajectory_pointcloud_radius: float = 0.5,  # Radius around trajectory to collect points (meters)
        force_use_cache: bool = False,    # Force use any compatible cache file, ignoring parameter mismatches
        max_bboxes: int = 400,  # Maximum number of bounding boxes to use
        # CLIP feature caching parameters
        use_clip_cache: bool = False,  # Whether to cache CLIP features
        clip_cache_dir: Optional[str] = None,  # Directory for CLIP cache
        clip_model_name: str = "ViT-B/32",  # CLIP model to use
    ):
        """
        Initialize the ADT trajectory dataset.

        Args:
            sequence_path: Path to the ADT sequence
            config: Optional configuration object with dataset parameters
            trajectory_length: Length of trajectories to extract (if config not provided)
            skip_frames: Number of frames to skip between samples (if config not provided)
            max_objects: Maximum number of objects to include (None for all)
            device_num: Device number for ADT data provider
            transform: Optional transform to apply to trajectories
            load_pointcloud: Whether to load the MPS pointcloud (if config not provided)
            pointcloud_subsample: Subsample factor for pointcloud (if config not provided)
            use_cache: Whether to use caching for trajectories (if config not provided)
            cache_dir: Directory to store cached data (if config not provided)
            min_motion_threshold: Minimum motion threshold in meters (if config not provided)
            min_motion_percentile: Filter trajectories below this percentile of motion (if config not provided)
            use_displacements: Whether to use displacements instead of absolute positions (if config not provided)
            detect_motion_segments: Whether to detect and extract active motion segments (if config not provided)
            motion_velocity_threshold: Threshold in m/s for detecting active motion (if config not provided)
            min_segment_frames: Minimum number of frames for a valid motion segment (if config not provided)
            max_stationary_frames: Maximum consecutive stationary frames allowed in a motion segment (if config not provided)
            normalize_data: Whether to normalize data using scene bounds (if config not provided)
            trajectory_pointcloud_radius: Radius around trajectory to collect points (meters)
            force_use_cache: Force use any compatible cache file, ignoring parameter mismatches
            max_bboxes: Maximum number of bounding boxes to use for conditioning (for consistent batching)
            use_clip_cache: Whether to cache CLIP features (if config not provided)
            clip_cache_dir: Directory for CLIP cache (if config not provided)
            clip_model_name: CLIP model to use (if config not provided)
        """
        from projectaria_tools.projects.adt import (
            AriaDigitalTwinDataProvider,
            AriaDigitalTwinDataPathsProvider,
            MotionType
        )

        self.sequence_path = sequence_path
        
        # Use config values if provided, otherwise use the explicitly passed parameters
        if config is not None:
            self.trajectory_length = getattr(config, 'trajectory_length', trajectory_length)
            self.skip_frames = getattr(config, 'skip_frames', skip_frames)
            self.load_pointcloud = getattr(config, 'load_pointcloud', load_pointcloud)
            self.pointcloud_subsample = getattr(config, 'pointcloud_subsample', pointcloud_subsample)
            self.use_cache = getattr(config, 'use_cache', use_cache)
            self.min_motion_threshold = getattr(config, 'min_motion_threshold', min_motion_threshold)
            self.min_motion_percentile = getattr(config, 'min_motion_percentile', min_motion_percentile)
            self.use_displacements = getattr(config, 'use_displacements', use_displacements)
            self.detect_motion_segments = getattr(config, 'detect_motion_segments', detect_motion_segments)
            self.motion_velocity_threshold = getattr(config, 'motion_velocity_threshold', motion_velocity_threshold)
            self.min_segment_frames = getattr(config, 'min_segment_frames', min_segment_frames)
            self.max_stationary_frames = getattr(config, 'max_stationary_frames', max_stationary_frames)
            self.normalize_data = getattr(config, 'normalize_data', normalize_data) # Get normalize_data from config
            self.trajectory_pointcloud_radius = getattr(config, 'trajectory_pointcloud_radius', trajectory_pointcloud_radius)
            self.force_use_cache = getattr(config, 'force_use_cache', force_use_cache) # Get force_use_cache from config
            self.max_bboxes = getattr(config, 'max_bboxes', max_bboxes)
            # CLIP caching parameters
            self.use_clip_cache = getattr(config, 'use_clip_cache', use_clip_cache)
            self.clip_cache_dir = getattr(config, 'clip_cache_dir', clip_cache_dir)
            self.clip_model_name = getattr(config, 'clip_model_name', clip_model_name)
            # For cache_dir, don't use config directly - it might be set based on save_path
        else:
            self.trajectory_length = trajectory_length
            self.skip_frames = skip_frames
            self.load_pointcloud = load_pointcloud
            self.pointcloud_subsample = pointcloud_subsample
            self.use_cache = use_cache
            self.min_motion_threshold = min_motion_threshold
            self.min_motion_percentile = min_motion_percentile
            self.use_displacements = use_displacements
            self.detect_motion_segments = detect_motion_segments
            self.motion_velocity_threshold = motion_velocity_threshold
            self.min_segment_frames = min_segment_frames
            self.max_stationary_frames = max_stationary_frames
            self.normalize_data = normalize_data # Use passed parameter if no config
            self.trajectory_pointcloud_radius = trajectory_pointcloud_radius
            self.force_use_cache = force_use_cache # Use passed parameter if no config
            self.max_bboxes = max_bboxes
            # CLIP caching parameters
            self.use_clip_cache = use_clip_cache
            self.clip_cache_dir = clip_cache_dir
            self.clip_model_name = clip_model_name
        
        # Force load_pointcloud to True when normalize_data is True for maximum precision
        self.load_pointcloud = self.load_pointcloud or self.normalize_data

        # These parameters don't typically come from config
        self.max_objects = max_objects
        self.transform = transform
        self.pointcloud = None
        self.trajectories = []
        
        # Scene bbox caching attributes
        self._scene_bboxes = None
        self._scene_bboxes_mask = None
        self._scene_bboxes_categories = None
        
        # Initialize recent valid bboxes cache attributes
        self._recent_valid_bboxes = None
        self._recent_valid_bbox_mask = None
        self._recent_valid_bbox_categories = None
        self._recent_valid_bboxes_timestamp = None
        self._max_bbox_time_delta_ns = 100_000_000  # 100ms
        
        self.cache_dir = cache_dir if cache_dir is not None else './trajectory_cache'

        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_cache and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
            except Exception as e:
                print(f"Warning: Could not create cache directory: {e}")
                self.use_cache = False

        # Initialize CLIP feature manager if CLIP caching is enabled
        self.clip_feature_manager = None
        if self.use_clip_cache:
            try:
                self.clip_feature_manager = CLIPFeatureManager(
                    cache_dir=None,
                    clip_model_name=self.clip_model_name,
                    device="cpu"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize CLIPFeatureManager: {e}")
                self.use_clip_cache = False

        # Initialize scene bounds
        self.scene_min = np.zeros(3, dtype=np.float32)
        self.scene_max = np.ones(3, dtype=np.float32)
        self.scene_scale = np.ones(3, dtype=np.float32)

        # Check for cached data first
        if self.use_cache and self._load_from_cache():
            return
            
        # Create data provider
        try:
            paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
            data_paths = paths_provider.get_datapaths(False)
            if data_paths is None:
                raise ValueError(f"Failed to get data paths from {sequence_path}")

            self.adt_provider = AriaDigitalTwinDataProvider(data_paths)

            # Load pointcloud if requested (or if needed for normalization)
            if self.load_pointcloud:
                self._load_pointcloud()
            elif self.normalize_data:
                self._load_pointcloud()
                if self.pointcloud is None:
                    self._set_default_scene_bounds()

            # Extract trajectories
            self._extract_trajectories()

            # Generate trajectory-specific pointclouds
            if self.pointcloud is not None:
                self._generate_trajectory_specific_pointclouds()

            # Cache the scene bounding boxes
            self._scene_bboxes, self._scene_bboxes_mask, self._scene_bboxes_categories = self.get_all_scene_bboxes()

            # Generate trajectory-specific bboxes
            if self._scene_bboxes is not None:
                self._generate_trajectory_specific_bboxes()

            # Save to cache if caching is enabled
            if self.use_cache:
                self._save_to_cache()

        except Exception as e:
            print(f"Error loading ADT sequence: {e}")
            traceback.print_exc()
            raise
    
    def _filter_points_by_trajectory(self, trajectory_positions, radius=None):
        """Filter point cloud to include only points within radius of trajectory."""
        if self.pointcloud is None or len(self.pointcloud) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if trajectory_positions is None or len(trajectory_positions) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if radius is None:
            radius = self.trajectory_pointcloud_radius

        try:
            if isinstance(trajectory_positions, torch.Tensor):
                trajectory_positions = trajectory_positions.detach().cpu().numpy()

            if self.normalize_data:
                trajectory_positions = self.denormalize_position(trajectory_positions)

            trajectory_tree = cKDTree(trajectory_positions)
            distances, _ = trajectory_tree.query(self.pointcloud, k=1)
            mask = distances <= radius
            return self.pointcloud[mask]

        except Exception as e:
            logger.warning(f"Error filtering point cloud: {e}")
            return np.zeros((0, 3), dtype=np.float32)
    
    def _generate_trajectory_specific_pointclouds(self):
        """Generate trajectory-specific pointclouds for each trajectory."""
        if self.pointcloud is None or len(self.pointcloud) == 0:
            return

        for i, traj_item in enumerate(self.trajectories):
            try:
                poses_tensor = traj_item['trajectory_data']['poses']
                positions_tensor = poses_tensor[:, :3]
                mask = traj_item['trajectory_data']['attention_mask']

                if isinstance(mask, torch.Tensor):
                    valid_indices = torch.where(mask > 0.5)[0]
                    valid_positions = positions_tensor[valid_indices]
                else:
                    valid_indices = np.where(np.array(mask) > 0.5)[0]
                    valid_positions = positions_tensor[valid_indices]

                if self.normalize_data and isinstance(valid_positions, torch.Tensor):
                    positions_np = valid_positions.detach().cpu().numpy()
                    valid_positions = self.denormalize_position(positions_np)
                elif self.normalize_data:
                    valid_positions = self.denormalize_position(valid_positions)

                filtered_pc = self._filter_points_by_trajectory(valid_positions)
                if 'metadata' not in traj_item:
                    traj_item['metadata'] = {}
                traj_item['trajectory_specific_pointcloud'] = filtered_pc

            except Exception as e:
                logger.warning(f"Error generating trajectory-specific point cloud for trajectory {i}: {e}")
                traj_item['trajectory_specific_pointcloud'] = np.zeros((0, 3), dtype=np.float32)
    
    def _get_cache_filename(self):
        """Generate a unique filename for the cache based on the dataset parameters."""
        # Get sequence name instead of full path for more portable caches
        sequence_name = os.path.basename(self.sequence_path)
        
        # When force_use_cache is enabled, don't include parameters in the hash
        # This will create a simpler filename that can be matched across different parameter configurations
        if self.force_use_cache:
            # Just use sequence name with a simple file extension
            return os.path.join(self.cache_dir, f"{sequence_name}_cache.pkl")
        
        # Create a unique identifier based on relevant parameters, excluding the full path
        params = {
            'sequence_name': sequence_name,  # Use sequence name, not full path
            'trajectory_length': self.trajectory_length,
            'skip_frames': self.skip_frames,
            'max_objects': self.max_objects,
            'use_displacements': self.use_displacements,
            'detect_motion_segments': self.detect_motion_segments,
            'motion_velocity_threshold': self.motion_velocity_threshold,
            'min_segment_frames': self.min_segment_frames,
            'max_stationary_frames': self.max_stationary_frames,
            'min_motion_threshold': self.min_motion_threshold,
            'min_motion_percentile': self.min_motion_percentile,
            'pointcloud_subsample': self.pointcloud_subsample,
            'normalize_data': self.normalize_data, # Add normalize_data to cache key
            'trajectory_pointcloud_radius': self.trajectory_pointcloud_radius, # Add new parameter to cache key
            'include_bbox_corners': True, # New parameter for bbox corners
            'max_bboxes': self.max_bboxes,  # Add max_bboxes to cache key
            'include_trajectory_specific_bboxes': True,  # New parameter for trajectory-specific bboxes
        }
        
        # Generate hash from parameters
        params_str = str(params)
        hash_obj = hashlib.md5(params_str.encode())
        hash_str = hash_obj.hexdigest()
        
        # Create filename using sequence name and hash
        return os.path.join(self.cache_dir, f"{sequence_name}_{hash_str}.pkl")
    
    def _save_to_cache(self):
        """Save trajectories and scene features to cache."""
        if not self.use_cache:
            return False
            
        cache_file = self._get_cache_filename()
        
        try:
            # Get sequence name for consistency
            sequence_name = os.path.basename(self.sequence_path)
            
            # Prepare data to save
            cache_data = {
                'trajectories': self.trajectories,
                'pointcloud': self.pointcloud,
                # Add normalization parameters to cache
                'scene_min': self.scene_min,
                'scene_max': self.scene_max,
                'scene_scale': self.scene_scale,
                # Add scene bbox data to cache
                'scene_bboxes': self._scene_bboxes,
                'scene_bboxes_mask': self._scene_bboxes_mask,
                'scene_bboxes_categories': self._scene_bboxes_categories,
                'params': {
                    'sequence_name': sequence_name,  # Store sequence name, not full path
                    'trajectory_length': self.trajectory_length,
                    'skip_frames': self.skip_frames,
                    'max_objects': self.max_objects,
                    'use_displacements': self.use_displacements,
                    'detect_motion_segments': self.detect_motion_segments,
                    'motion_velocity_threshold': self.motion_velocity_threshold,
                    'min_segment_frames': self.min_segment_frames,
                    'max_stationary_frames': self.max_stationary_frames,
                    'min_motion_threshold': self.min_motion_threshold,
                    'min_motion_percentile': self.min_motion_percentile,
                    'pointcloud_subsample': self.pointcloud_subsample,
                    'normalize_data': self.normalize_data, # Save normalize_data state
                    'trajectory_pointcloud_radius': self.trajectory_pointcloud_radius, # Save new parameter
                    'include_bbox_corners': True, # Save new parameter state
                    'max_bboxes': self.max_bboxes,  # Save max_bboxes state
                    'include_trajectory_specific_bboxes': True,  # Save new parameter state
                    'clip_model_name': self.clip_model_name if self.use_clip_cache else None,  # Include CLIP model info
                }
            }
            
            # Compute and add CLIP features if enabled
            if self.use_clip_cache and self.clip_feature_manager:
                logger.info(f"[CLIP CACHE] Computing CLIP features for sequence {sequence_name}")
                
                # Extract unique object categories from trajectories
                object_categories = []
                scene_bbox_categories_list = []
                
                for traj in self.trajectories:
                    # Object category
                    obj_cat = traj.get('object_category', traj.get('object_name', 'unknown'))
                    object_categories.append(obj_cat)
                    
                    # Scene bbox categories (if available)
                    if 'trajectory_specific_bbox_categories' in traj:
                        scene_bbox_categories_list.append(traj['trajectory_specific_bbox_categories'])
                
                # Encode object categories
                if object_categories:
                    logger.info(f"[CLIP CACHE] Encoding {len(object_categories)} object categories")
                    obj_features = self.clip_feature_manager.encode_texts(object_categories)
                    
                    clip_features = {
                        'object_category': obj_features,
                    }
                    
                    # TODO: Add scene bbox category encoding if needed
                    # This would require flattening the nested list structure
                    
                    cache_data['clip_features'] = clip_features
                    logger.info(f"[CLIP CACHE] Added CLIP features to cache for sequence {sequence_name}")
                else:
                    logger.warning(f"[CLIP CACHE] No object categories found to encode")
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            return True
        except Exception as e:
            logger.warning(f"Error saving trajectory cache: {e}")
            return False
    
    def _load_from_cache(self):
        """Load trajectories and scene features from cache if available."""
        if not self.use_cache:
            return False
            
        # Get current cache file path with hash
        cache_file = self._get_cache_filename()
        
        # Get sequence name for potential fallback search
        sequence_name = os.path.basename(self.sequence_path)
        
        # Try loading the exact cache file first
        if not os.path.exists(cache_file):
            # Fallback: try to find any cache file for this sequence
            potential_files = []
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(f"{sequence_name}_") and filename.endswith(".pkl"):
                        potential_files.append(os.path.join(self.cache_dir, filename))

            if potential_files:
                potential_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                cache_file = potential_files[0]
            else:
                return False
            
        try:
            # Load data from file
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Get sequence name for comparison (not full path)
            sequence_name = os.path.basename(self.sequence_path)
            
            # Print the actual parameters for debugging
            stored_params = cache_data['params']
            current_params = {
                'sequence_name': sequence_name,
                'trajectory_length': self.trajectory_length,
                'skip_frames': self.skip_frames,
                'max_objects': self.max_objects,
                'use_displacements': self.use_displacements,
                'detect_motion_segments': self.detect_motion_segments,
                'motion_velocity_threshold': self.motion_velocity_threshold,
                'min_segment_frames': self.min_segment_frames,
                'max_stationary_frames': self.max_stationary_frames,
                'min_motion_threshold': self.min_motion_threshold,
                'min_motion_percentile': self.min_motion_percentile,
                'pointcloud_subsample': self.pointcloud_subsample,
                'normalize_data': self.normalize_data,
                'trajectory_pointcloud_radius': self.trajectory_pointcloud_radius,
                'include_bbox_corners': True, # Current setting for bbox corners
                'max_bboxes': self.max_bboxes,  # Current setting for max_bboxes
                'include_trajectory_specific_bboxes': True,  # Current setting for trajectory-specific bboxes
            }
            
            # Verify parameters match
            stored_sequence = stored_params.get('sequence_name', os.path.basename(stored_params.get('sequence_path', '')))
            if stored_sequence != sequence_name:
                return False

            # Check critical parameters (skip if force_use_cache is enabled)
            if not self.force_use_cache:
                critical_params = ['trajectory_length', 'skip_frames', 'use_displacements',
                                   'normalize_data', 'detect_motion_segments', 'motion_velocity_threshold',
                                   'min_segment_frames', 'max_stationary_frames',
                                   'include_bbox_corners', 'max_bboxes', 'include_trajectory_specific_bboxes']
                for param in critical_params:
                    stored_value = stored_params.get(param, None)
                    current_value = current_params.get(param, None)
                    if stored_value != current_value:
                        return False
            
            # Load data
            self.trajectories = cache_data['trajectories']
            self.pointcloud = cache_data['pointcloud']
            # Load normalization parameters from cache, with fallback for older caches
            self.scene_min = cache_data.get('scene_min', np.zeros(3, dtype=np.float32))
            self.scene_max = cache_data.get('scene_max', np.ones(3, dtype=np.float32))
            self.scene_scale = cache_data.get('scene_scale', np.ones(3, dtype=np.float32))
            # Load scene bbox data from cache
            self._scene_bboxes = cache_data.get('scene_bboxes', None)
            self._scene_bboxes_mask = cache_data.get('scene_bboxes_mask', None)
            self._scene_bboxes_categories = cache_data.get('scene_bboxes_categories', None)
            # Ensure normalize_data flag is set correctly based on cache
            self.normalize_data = stored_params.get('normalize_data', True)
            # Bbox corners are already part of self.trajectories, no separate loading needed here
            # if they were saved correctly. The check for 'include_bbox_corners' ensures compatibility.
            
            # Load CLIP features if available
            if self.use_clip_cache and 'clip_features' in cache_data:
                logger.info(f"[CLIP CACHE] Loading pre-computed CLIP features for sequence {sequence_name}")
                clip_features = cache_data['clip_features']
                
                # Add CLIP features to each trajectory
                if 'object_category' in clip_features:
                    for i, traj in enumerate(self.trajectories):
                        if i < len(clip_features['object_category']):
                            traj['object_category_clip'] = clip_features['object_category'][i]
                            logger.debug(f"[CLIP CACHE] Added CLIP feature for trajectory {i}")
                    logger.info(f"[CLIP CACHE] Loaded CLIP features for {len(clip_features['object_category'])} trajectories")
                
            return True
        except Exception as e:
            logger.warning(f"Error loading trajectory cache: {e}")
            return False

    def _load_pointcloud(self):
        """Load the MPS pointcloud from the sequence."""
        if not os.path.exists(self.sequence_path):
            self._set_default_scene_bounds()
            return

        try:
            import projectaria_tools.core.mps as aria_mps
            from projectaria_tools.core.mps.utils import filter_points_from_confidence

            mps_path = os.path.join(self.sequence_path, "mps")
            if not os.path.exists(mps_path):
                self._set_default_scene_bounds()
                return
                
            paths_provider = aria_mps.MpsDataPathsProvider(mps_path)
            data_paths = paths_provider.get_data_paths()
            if not data_paths:
                self._set_default_scene_bounds()
                return

            mps_provider = aria_mps.MpsDataProvider(data_paths)
            if not mps_provider.has_semidense_point_cloud():
                self._set_default_scene_bounds()
                return

            point_cloud = mps_provider.get_semidense_point_cloud()
            if not point_cloud:
                self._set_default_scene_bounds()
                return

            # Filter the point cloud using thresholds
            point_cloud = filter_points_from_confidence(point_cloud, 0.001, 0.15)
            points = np.array([point.position_world for point in point_cloud])

            # Subsample points if needed
            if self.pointcloud_subsample > 1 and len(points) > 0:
                indices = np.random.choice(len(points), size=len(points) // self.pointcloud_subsample, replace=False)
                points = points[indices]

            self.pointcloud = points

            # Calculate scene bounds for normalization
            if self.normalize_data and len(self.pointcloud) > 0:
                self.scene_min = np.min(self.pointcloud, axis=0)
                self.scene_max = np.max(self.pointcloud, axis=0)
                padding = (self.scene_max - self.scene_min) * 0.05
                self.scene_min -= padding
                self.scene_max += padding
                self.scene_scale = self.scene_max - self.scene_min
                self.scene_scale[self.scene_scale < 1e-6] = 1.0
            elif self.normalize_data:
                self._set_default_scene_bounds()

        except Exception as e:
            logger.warning(f"Error loading MPS point cloud: {e}")
            self._set_default_scene_bounds()
    
    def _set_default_scene_bounds(self):
        """Set default scene bounds if point cloud loading fails."""
        if self.normalize_data:
            self.scene_min = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
            self.scene_max = np.array([10.0, 10.0, 10.0], dtype=np.float32)
            self.scene_scale = self.scene_max - self.scene_min
        else:
            self.scene_min = np.zeros(3, dtype=np.float32)
            self.scene_max = np.ones(3, dtype=np.float32)
            self.scene_scale = np.ones(3, dtype=np.float32)

    def normalize_position(self, position):
        """Normalize position from world space to [-1, 1] range using scene bounds."""
        if not self.normalize_data:
            return np.asarray(position, dtype=np.float32)  # Return original if not normalizing
        
        position_array = np.asarray(position, dtype=np.float32)
        # Convert from world space to [0, 1] range
        normalized = (position_array - self.scene_min) / self.scene_scale
        # Convert from [0, 1] to [-1, 1] range
        normalized = normalized * 2.0 - 1.0
        return normalized

    def denormalize_position(self, normalized_position):
        """Denormalize position from [-1, 1] range back to world space using scene bounds."""
        if not self.normalize_data:
            return np.asarray(normalized_position, dtype=np.float32)  # Return as is if not normalized
        
        normalized_array = np.asarray(normalized_position, dtype=np.float32)
        # Convert from [-1, 1] to [0, 1] range
        denormalized = (normalized_array + 1.0) / 2.0
        # Convert from [0, 1] range to world space
        denormalized = denormalized * self.scene_scale + self.scene_min
        return denormalized

    def normalize_size(self, size):
        """Normalize size/dimension using scene scale."""
        if not self.normalize_data:
            return np.asarray(size, dtype=np.float32)  # Return original if not normalizing
        
        size_array = np.asarray(size, dtype=np.float32)
        # Normalize size based on scene scale - result will be in [0, 1] range approximately
        normalized = size_array / self.scene_scale
        # Scale to [-1, 1] range for consistency with positions
        normalized = normalized * 2.0
        return normalized

    def denormalize_size(self, normalized_size):
        """Denormalize size from normalized range back to world space."""
        if not self.normalize_data:
            return np.asarray(normalized_size, dtype=np.float32)  # Return as is if not normalized
        
        normalized_array = np.asarray(normalized_size, dtype=np.float32)
        # Scale back from [-1, 1] range-friendly format
        denormalized = normalized_array / 2.0
        # Convert back to world space scale
        denormalized = denormalized * self.scene_scale
        return denormalized
    
    def _calculate_trajectory_motion(self, positions, timestamps=None):
        """
        Calculate the total motion/displacement of a trajectory.
        
        Args:
            positions: List of position vectors [x,y,z] (extracted from poses if necessary)
            timestamps: Optional list of timestamps corresponding to positions
            
        Returns:
            total_path_length: Sum of distances between consecutive points (absolute distance moved)
            direct_displacement: Direct displacement between first and last points
            max_displacement: Maximum displacement between any point and the start point
            avg_velocity: Average velocity in m/s (with timestamps) or per-frame (without)
        """
        if len(positions) < 2:
            return 0.0, 0.0, 0.0, 0.0  # Return all four values: total_path_length, direct_displacement, max_displacement, avg_velocity
        
        # Convert positions to numpy array if needed
        if isinstance(positions, torch.Tensor):
            positions_np = positions.detach().cpu().numpy()
        elif isinstance(positions, list):
            # If positions is a list of position vectors, convert to numpy array
            # Ensure each position is only [x,y,z] if extracted from [x,y,z,roll,pitch,yaw]
            if len(positions[0]) > 3:
                positions_np = np.array([pos[:3] for pos in positions])
            else:
                positions_np = np.array(positions)
        else:
            positions_np = positions
            
        # Calculate total path length (sum of all point-to-point movements)
        # This represents the absolute distance moved along the trajectory
        diffs = np.diff(positions_np, axis=0)
        segment_distances = np.linalg.norm(diffs, axis=1)
        total_path_length = np.sum(segment_distances)
        
        # Calculate direct start-to-end displacement
        # This is just the straight-line distance between first and last points
        if len(positions_np) > 1:
            direct_displacement = np.linalg.norm(positions_np[-1] - positions_np[0])
        else:
            direct_displacement = 0.0
            
        # Calculate maximum displacement from start
        distances_from_start = np.linalg.norm(positions_np - positions_np[0], axis=1)
        max_displacement = np.max(distances_from_start)
        
        # Calculate velocity using timestamps if available
        if timestamps is not None and len(timestamps) == len(positions_np):
            # Calculate full trajectory duration in seconds
            trajectory_duration_ns = timestamps[-1] - timestamps[0]
            trajectory_duration_s = trajectory_duration_ns / 1e9
            
            if trajectory_duration_s > 0:
                # Average velocity over the entire trajectory (m/s)
                avg_velocity = total_path_length / trajectory_duration_s
                
                # Additional metrics using timestamps
                # Calculate instantaneous velocities
                instant_velocities = []
                for i in range(1, len(positions_np)):
                    dt_ns = timestamps[i] - timestamps[i-1]
                    dt_s = dt_ns / 1e9
                    if dt_s > 0:
                        vel = segment_distances[i-1] / dt_s
                        instant_velocities.append(vel)
                
                if instant_velocities:
                    # These are useful metrics but we'll just store avg_velocity for now
                    max_velocity = np.max(instant_velocities)
                    min_velocity = np.min(instant_velocities)
                    median_velocity = np.median(instant_velocities)
            else:
                avg_velocity = 0.0
        else:
            # Fall back to frame-based velocity if timestamps not available
            avg_velocity = total_path_length / (len(positions_np) - 1) if len(positions_np) > 1 else 0.0
        
        return total_path_length, direct_displacement, max_displacement, avg_velocity

    def _extract_trajectories(self):
        """Extract trajectories from the ADT sequence."""
        # Get timestamps from a reliable stream (usually RGB camera)
        from projectaria_tools.core.stream_id import StreamId
        from projectaria_tools.projects.adt import MotionType
        
        # Use the RGB camera stream ID directly
        rgb_stream_id = StreamId("214-1")  # This is the standard RGB camera ID in projectaria_tools
        
        # Get all timestamps
        all_timestamps = self.adt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)
        if not all_timestamps:
            raise ValueError("No timestamps found in dataset")
            
        # Get valid time range
        start_time = self.adt_provider.get_start_time_ns()
        end_time = self.adt_provider.get_end_time_ns()
        
        # Filter timestamps to valid range
        timestamps = [ts for ts in all_timestamps if start_time <= ts <= end_time]
        if not timestamps:
            raise ValueError("No valid timestamps found in valid time range")

        # Get all dynamic objects
        dynamic_objects = []
        all_objects = []

        instance_ids = self.adt_provider.get_instance_ids()
        for instance_id in instance_ids:
            try:
                instance_info = self.adt_provider.get_instance_info_by_id(instance_id)
                all_objects.append((instance_id, instance_info))
                if instance_info.motion_type == MotionType.DYNAMIC:
                    dynamic_objects.append((instance_id, instance_info))
            except Exception:
                continue

        if not dynamic_objects:
            dynamic_objects = all_objects

        if not dynamic_objects:
            raise ValueError("No objects found in sequence")

        if self.max_objects is not None and len(dynamic_objects) > self.max_objects:
            dynamic_objects = dynamic_objects[:self.max_objects]
        
        trajectories_with_motion = []

        # Track each object
        for obj_id, instance_info in dynamic_objects:
            try:
                segments = self._track_object_motion(obj_id, timestamps)
                for segment_idx, (poses, tracked_timestamps, bbox_corners_sequence) in enumerate(segments):
                    if len(poses) < 2:
                        continue
                    self._process_trajectory_segment(
                        poses, tracked_timestamps, instance_info, obj_id,
                        trajectories_with_motion, segment_idx=segment_idx,
                        bbox_corners_sequence=bbox_corners_sequence
                    )
            except Exception as e:
                logger.warning(f"Error processing object {obj_id}: {e}")

        # Apply motion filtering if threshold is set
        if self.min_motion_threshold > 0 or self.min_motion_percentile > 0:
            sorted_trajectories = sorted(trajectories_with_motion, key=lambda x: x['motion_value'])

            if self.min_motion_threshold > 0:
                filtered_trajectories = [t for t in trajectories_with_motion if t['motion_value'] >= self.min_motion_threshold]
                if len(filtered_trajectories) == 0 and len(trajectories_with_motion) > 0:
                    filtered_trajectories = [sorted_trajectories[-1]]
                trajectories_with_motion = filtered_trajectories

            if self.min_motion_percentile > 0 and len(sorted_trajectories) > 0:
                cutoff_idx = int(len(sorted_trajectories) * (self.min_motion_percentile / 100.0))
                filtered_trajectories = sorted_trajectories[cutoff_idx:]
                if len(filtered_trajectories) == 0:
                    filtered_trajectories = [sorted_trajectories[-1]]
                trajectories_with_motion = filtered_trajectories

            trajectories_with_motion.sort(key=lambda x: x['motion_value'], reverse=True)
        
        # Store final trajectories (remove the motion_value field)
        self.trajectories = [
            {'trajectory_data': t['trajectory_data'], 'metadata': t['metadata']}
            for t in trajectories_with_motion
        ]
    
    def _process_trajectory_segment(self, poses, tracked_timestamps, instance_info, obj_id, trajectories_with_motion, segment_idx=None, bbox_corners_sequence=None):
        """
        Process a single trajectory segment and add it to the trajectories list.
        
        Args:
            poses: List of pose vectors [x, y, z, roll, pitch, yaw]
            tracked_timestamps: List of timestamps for each pose
            instance_info: Object instance information
            obj_id: Object ID
            trajectories_with_motion: List to store processed trajectories
            segment_idx: Optional index of the segment (for multi-segment objects)
            bbox_corners_sequence: List of 8 corners of the Oriented Bounding Box (OBB) for each timestamp
        """
        # Extract positions (first 3 elements of each pose) for motion metrics
        positions = [pose[:3] for pose in poses]
        
        # Calculate trajectory motion metrics (using original positions before normalization)
        total_path_length, direct_displacement, max_displacement, avg_velocity = self._calculate_trajectory_motion(positions, tracked_timestamps)
        
        # Convert to numpy array and ensure consistent 2D shape [N, 9]
        poses_array = np.array(poses)
        if poses_array.ndim > 2:
            poses_array = poses_array.reshape(-1, 9)  # Reshape to [N, 9] for [x,y,z,r6d_1,r6d_2,r6d_3,r6d_4,r6d_5,r6d_6]
        
        # Split positions and rotations for separate normalization
        positions_array = poses_array[:, :3]  # [N, 3] positions
        rotations_array = poses_array[:, 3:9]  # [N, 6] rotations (6D representation)
        
        # Normalize positions if normalize_data is True, leave rotations unchanged
        if self.normalize_data:
            positions_array = self.normalize_position(positions_array)

        # Recombine normalized positions with rotations
        normalized_poses_array = np.concatenate([positions_array, rotations_array], axis=1)  # [N, 9]

        # Convert normalized (or original) poses to tensor
        poses_tensor = torch.tensor(normalized_poses_array, dtype=torch.float32)
        real_length = min(len(poses_tensor), self.trajectory_length)
        
        # Modified displacement calculation (applied on normalized positions if enabled)
        if self.use_displacements and real_length > 1:
            # Create a new tensor to store the modified poses with displacements for positions
            modified_tensor = torch.zeros_like(poses_tensor)
            
            # Keep the first pose as absolute coordinates and rotations
            modified_tensor[0] = poses_tensor[0]
            
            # Calculate displacements for positions (first 3 elements) after the first
            modified_tensor[1:real_length, :3] = poses_tensor[1:real_length, :3] - poses_tensor[:real_length-1, :3]
            
            # Keep absolute rotations (last 6 elements) for all poses
            modified_tensor[1:real_length, 3:] = poses_tensor[1:real_length, 3:]
            
            # Replace poses with this new representation
            poses_tensor = modified_tensor
            
            # No need to store first_position separately in metadata
            metadata_first_position = None
        else:
            metadata_first_position = None
        
        # Create attention mask (ones for real data, zeros for padding)
        attention_mask = torch.zeros(self.trajectory_length, dtype=torch.float32)
        attention_mask[:real_length] = 1.0
        
        # Pad the poses tensor if needed
        if len(poses_tensor) < self.trajectory_length:
            # Make sure padding has the same shape [M, 9]
            # Pad with zeros in the normalized space if normalizing, otherwise world space zeros
            padding = torch.zeros(self.trajectory_length - len(poses_tensor), 9, dtype=torch.float32)
            poses_tensor = torch.cat([poses_tensor, padding], dim=0)
        else:
            poses_tensor = poses_tensor[:self.trajectory_length]
        
        # Create trajectory data dict
        trajectory_data = {
            'poses': poses_tensor,  # Now contains [x, y, z, r6d_1, r6d_2, r6d_3, r6d_4, r6d_5, r6d_6]
            'attention_mask': attention_mask,
        }
        
        # Add bbox_corners_sequence to trajectory_data after processing
        if bbox_corners_sequence is not None and len(bbox_corners_sequence) > 0:
            # Convert list of [8,3] arrays to a single numpy array [N, 8, 3]
            bbox_corners_array = np.array(bbox_corners_sequence, dtype=np.float32)
            
            # Normalize bbox corners if normalize_data is True
            if self.normalize_data:
                # Reshape for normalization: [N*8, 3]
                original_shape = bbox_corners_array.shape
                bbox_corners_flat = bbox_corners_array.reshape(-1, 3)
                normalized_corners_flat = self.normalize_position(bbox_corners_flat)
                bbox_corners_array = normalized_corners_flat.reshape(original_shape)

            bbox_corners_tensor = torch.tensor(bbox_corners_array, dtype=torch.float32)
            
            # Pad or truncate bbox_corners_tensor to self.trajectory_length
            # Real length is determined by the poses/attention_mask
            current_len_bbox = bbox_corners_tensor.shape[0]
            if current_len_bbox < self.trajectory_length:
                padding_bbox = torch.zeros(self.trajectory_length - current_len_bbox, 8, 3, dtype=torch.float32)
                padded_bbox_corners_tensor = torch.cat([bbox_corners_tensor, padding_bbox], dim=0)
            else:
                padded_bbox_corners_tensor = bbox_corners_tensor[:self.trajectory_length, :, :]
            
            trajectory_data['bbox_corners'] = padded_bbox_corners_tensor
        else:
            # If no bbox data, add zeros as placeholder
            trajectory_data['bbox_corners'] = torch.zeros(self.trajectory_length, 8, 3, dtype=torch.float32)

        # Create metadata with only essential information
        metadata = {
            'name': instance_info.name,
            'prototype_name': instance_info.name.split('_')[0] if '_' in instance_info.name else 'unknown',
            'id': int(obj_id),
            'sequence': self.sequence_path.split('/')[-1],
            'total_path_length': float(total_path_length), # Motion metrics always in original scale
            'direct_displacement': float(direct_displacement),
            'max_displacement': float(max_displacement),
            'avg_velocity': float(avg_velocity),
            'is_displacement': self.use_displacements,
            'active_motion_segment': True,  # Flag indicating this is an active motion segment
            'segment_duration_s': float((tracked_timestamps[-1] - tracked_timestamps[0]) / 1e9) if len(tracked_timestamps) > 1 else 0.0,
            'original_length': len(poses),  # Store the length of the active segment
            'start_timestamp_ns': int(tracked_timestamps[0]) if tracked_timestamps else None,  # Store segment start timestamp
            # Store normalization parameters for denormalization later
            'normalization': {
                'is_normalized': self.normalize_data,
                'scene_min': self.scene_min.tolist(),
                'scene_max': self.scene_max.tolist(),
                'scene_scale': self.scene_scale.tolist()
            }
        }
        
        # Add segment index if provided
        if segment_idx is not None:
            metadata['segment_idx'] = segment_idx
        
        # Extract category and subcategory if available
        metadata['category'] = getattr(instance_info, 'category', 'unknown')
        metadata['subcategory'] = getattr(instance_info, 'subcategory', 'unknown')
        
        # Create complete trajectory item
        trajectory_item = {
            'trajectory_data': trajectory_data,
            'metadata': metadata,
            'motion_value': total_path_length  # Store total path length for filtering (absolute distance moved)
        }
        
        # Add bbox_corners_sequence to metadata
        if bbox_corners_sequence is not None:
            metadata['bbox_corners_sequence'] = bbox_corners_sequence
        
        trajectories_with_motion.append(trajectory_item)
    
    def _track_object_motion(self, object_id, timestamps):
        """
        Track an object's motion across a sequence.
        
        Args:
            object_id: ID of the object to track
            timestamps: List of timestamps to track at
            
        Returns:
            list: List of (poses, timestamps, bbox_corners_sequence) tuples for all active motion segments,
                  where poses contains both position and orientation [x, y, z, roll, pitch, yaw]
                  or a list with a single tuple of the entire trajectory if motion detection is disabled
        """
        from collections import deque
        import numpy as np
        
        # Lists to store trajectory data
        poses = []  # Will store [x, y, z, roll, pitch, yaw]
        tracked_timestamps = []
        bbox_corners_sequence = [] # New list to store 8 corners of OBB for each timestamp
        
        # Track object at each timestamp
        time_idx = 0  # For debug print limiting
        
        try:
            # Use skip_frames parameter to skip frames
            for i in range(0, len(timestamps), self.skip_frames):
                ts = timestamps[i]
                time_idx += 1
                try:
                    # Get 3D bounding box for the dynamic object
                    # Get all bounding boxes at this timestamp
                    bbox3d_with_dt = self.adt_provider.get_object_3d_boundingboxes_by_timestamp_ns(ts)
                    
                    if not bbox3d_with_dt.is_valid():
                        # Skip if we don't have valid bounding boxes at this timestamp
                        continue
                        
                    # Get the bounding box data
                    bboxes3d = bbox3d_with_dt.data()
                    
                    # Check if this object exists in the bounding boxes
                    if object_id not in bboxes3d:
                        # Skip this timestamp if object doesn't have a bounding box
                        continue
                    
                    # Get the specific bounding box for this object
                    bbox = bboxes3d[object_id]
                    
                    # Extract the center position from the transformation
                    center = bbox.transform_scene_object.translation()[0]
                    pos = [center[0], center[1], center[2]]
                    
                    # Get the dimensions from the AABB to calculate proper geometric center
                    aabb = bbox.aabb
                    height = aabb[3] - aabb[2]  # ymax - ymin (height in Y-axis)
                    
                    # Adjust center from bottom center to geometric center by moving up by half height
                    pos[1] = pos[1] + height / 2.0  # Move center up by half height in Y-axis
                    
                    # Extract rotation matrix from transform_scene_object
                    rotation_matrix = bbox.transform_scene_object.rotation().to_matrix()
                    
                    # Convert rotation matrix to 6D representation instead of Euler angles
                    rotation_6d = rotation_matrix_to_6d(rotation_matrix)
                    
                    # Create combined pose [x, y, z, r6d_1, r6d_2, r6d_3, r6d_4, r6d_5, r6d_6]
                    pose = pos + rotation_6d.tolist()
                    
                    # Store the combined pose and timestamp
                    poses.append(pose)
                    tracked_timestamps.append(ts)

                    # --- Extract 8 corners of the Oriented Bounding Box (OBB) ---
                    # sx = bbox.extent_x # Causes AttributeError
                    # sy = bbox.extent_y # Causes AttributeError
                    # sz = bbox.extent_z # Causes AttributeError

                    # Use dimensions from AABB as a fallback, acknowledging this is not the true OBB extent if rotated
                    aabb = bbox.aabb
                    sx = aabb[1] - aabb[0]  # width from world AABB
                    sy = aabb[3] - aabb[2]  # height from world AABB
                    sz = aabb[5] - aabb[4]  # depth from world AABB


                    local_corners = np.array([
                        [-sx/2, -sy/2, -sz/2], [sx/2, -sy/2, -sz/2], [sx/2, sy/2, -sz/2], [-sx/2, sy/2, -sz/2],
                        [-sx/2, -sy/2, sz/2], [sx/2, -sy/2, sz/2], [sx/2, sy/2, sz/2], [-sx/2, sy/2, sz/2]
                    ])

                    # Get the 4x4 transformation matrix - FIX: SE3 may not have matrix() method
                    try:
                        # Try the original method first
                        T_scene_object = bbox.transform_scene_object.matrix()
                        # Adjust the translation to account for geometric center instead of bottom center
                        T_scene_object[1, 3] = T_scene_object[1, 3] + sy / 2.0
                    except AttributeError:
                        # If matrix() not available, construct the matrix manually
                        rotation = bbox.transform_scene_object.rotation().to_matrix()
                        translation = bbox.transform_scene_object.translation()[0]
                        
                        # Construct 4x4 transformation matrix
                        T_scene_object = np.eye(4)
                        T_scene_object[:3, :3] = rotation
                        T_scene_object[:3, 3] = translation
                        
                        # Adjust the translation to account for geometric center instead of bottom center
                        # We need to move the center up by half height in the Y-axis
                        T_scene_object[1, 3] = T_scene_object[1, 3] + sy / 2.0

                    # Transform local corners to world coordinates
                    local_corners_h = np.hstack((local_corners, np.ones((8, 1)))) # Homogeneous coordinates
                    world_corners_h = (T_scene_object @ local_corners_h.T).T
                    world_corners = world_corners_h[:, :3] # De-homogenize

                    bbox_corners_sequence.append(world_corners)
                    # --------------------------------------------------------------

                except Exception as e:
                    if time_idx % 100 == 0 or time_idx == 1:
                        print(f"Warning: Could not get pose for object {object_id} at time {ts}: {e}")
                    # Skip this timestamp
                    continue
        except Exception as e:
            print(f"Error tracking object {object_id}: {e}")
            traceback.print_exc()
        
        # If we didn't collect enough points or motion detection is disabled, return raw trajectory as a single segment
        if len(poses) < self.min_segment_frames or not self.detect_motion_segments:
            return [(poses, tracked_timestamps, bbox_corners_sequence)]  # Return as a list with one segment
        
        # Calculate velocity between each pair of points - only considering position component (not rotation)
        velocities = []
        time_differences = []
        
        for i in range(1, len(poses)):
            # Calculate time difference in seconds
            dt = (tracked_timestamps[i] - tracked_timestamps[i-1]) / 1e9  # Convert ns to seconds
            
            if dt <= 0:
                # Skip invalid time differences
                continue
                
            # Calculate displacement vector (just for position part - first 3 elements)
            dx = np.array(poses[i][:3]) - np.array(poses[i-1][:3])
            
            # Calculate velocity in m/s
            velocity = np.linalg.norm(dx) / dt
            
            velocities.append(velocity)
            time_differences.append(dt)
        
        # Initialize segment tracking
        active_segments = []
        current_segment = []
        current_segment_timestamps = []
        stationary_count = 0
        
        # Process each position with its velocity
        for i in range(len(poses)):
            # Handle first point
            if i == 0:
                current_segment.append(poses[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                continue
                
            # Get velocity for this point (velocity at index i-1 corresponds to the motion from i-1 to i)
            if i-1 < len(velocities):
                velocity = velocities[i-1]
            else:
                # No velocity data available, add point and continue
                current_segment.append(poses[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                continue
            
            # Check if this point is moving
            is_moving = velocity >= self.motion_velocity_threshold
            
            if is_moving:
                # Add point to current segment and reset stationary counter
                current_segment.append(poses[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                stationary_count = 0
            else:
                # Point is stationary
                stationary_count += 1
                
                # Still include stationary points up to max_stationary_frames
                if stationary_count <= self.max_stationary_frames:
                    current_segment.append(poses[i])
                    current_segment_timestamps.append(tracked_timestamps[i])
                else:
                    # Too many stationary frames - end current segment if long enough
                    if len(current_segment) >= self.min_segment_frames:
                        # Store current segment
                        # 修复可能出现的IndexError，更安全地从bbox_corners_sequence获取数据
                        current_bbox_segment = []
                        for t in current_segment_timestamps:
                            if t in tracked_timestamps:
                                try:
                                    idx = tracked_timestamps.index(t)
                                    if 0 <= idx < len(bbox_corners_sequence):
                                        current_bbox_segment.append(bbox_corners_sequence[idx])
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Failed to get bbox corners for timestamp {t}: {e}")
                        
                        active_segments.append((current_segment.copy(), current_segment_timestamps.copy(), current_bbox_segment.copy()))
                    
                    # Start a new segment with this point
                    current_segment = [poses[i]]
                    current_segment_timestamps = [tracked_timestamps[i]]
                    stationary_count = 0
        
        # Add the final segment if it's long enough
        if len(current_segment) >= self.min_segment_frames:
            current_bbox_segment = []
            for t in current_segment_timestamps:
                if t in tracked_timestamps:
                    try:
                        idx = tracked_timestamps.index(t)
                        if 0 <= idx < len(bbox_corners_sequence):
                            current_bbox_segment.append(bbox_corners_sequence[idx])
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to get bbox corners for timestamp {t}: {e}")
            
            active_segments.append((current_segment, current_segment_timestamps, current_bbox_segment))
        
        # If no segments found or all too short, return the original trajectory
        if not active_segments:
            return [(poses, tracked_timestamps, bbox_corners_sequence)]
        
        # Return all active segments in their original chronological order
        return active_segments
    
    def get_scene_pointcloud(self):
        """
        Return the scene pointcloud if available.
        
        Returns:
            Numpy array of point coordinates [N, 3] or None if not available
        """
        return self.pointcloud

    def get_all_scene_bboxes(self, timestamp_ns=None, use_recent_fallback=True):
        """
        Extract bounding box information for all objects in the scene at a specific timestamp.
        
        Args:
            timestamp_ns: Optional timestamp for which to extract bounding boxes.
                         If None, a valid timestamp will be automatically selected.
            use_recent_fallback: Whether to use recently cached bounding boxes if retrieval fails
            
        Returns:
            Tuple of (
                torch.Tensor: Bounding box information [max_bboxes, bbox_dim],
                torch.Tensor: Mask indicating which boxes are real (1) vs padding (0) [max_bboxes],
                List[str]: Categories for each bounding box [max_bboxes]
            ) or (None, None, None) if no bounding box information is available
        """
        try:
            if not hasattr(self, 'adt_provider') or self.adt_provider is None:
                print("Warning: ADT provider not available for bounding box retrieval.")
                # Try to use recently cached bounding boxes as fallback
                if use_recent_fallback and hasattr(self, '_recent_valid_bboxes') and self._recent_valid_bboxes is not None:
                    return self._recent_valid_bboxes, self._recent_valid_bbox_mask, self._recent_valid_bbox_categories
                return None, None, None
            
            # Get timestamps from RGB stream if we need to find the first timestamp
            if timestamp_ns is None:
                from projectaria_tools.core.stream_id import StreamId
                rgb_stream_id = StreamId("214-1")  # Standard RGB camera ID
                
                # Get valid timestamps
                all_timestamps = self.adt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)
                if not all_timestamps:
                    print("Warning: No timestamps found in the scene.")
                    # Try to use recent bounding boxes as fallback
                    if use_recent_fallback and self._recent_valid_bboxes is not None:
                        return self._recent_valid_bboxes, self._recent_valid_bbox_mask, self._recent_valid_bbox_categories
                    return None, None, None
                    
                # Filter timestamps to valid range
                start_time = self.adt_provider.get_start_time_ns()
                end_time = self.adt_provider.get_end_time_ns()
                valid_timestamps = [ts for ts in all_timestamps if start_time <= ts <= end_time]
                
                if not valid_timestamps:
                    print("Warning: No valid timestamps found.")
                    # Try to use recent bounding boxes as fallback
                    if use_recent_fallback and self._recent_valid_bboxes is not None:
                        return self._recent_valid_bboxes, self._recent_valid_bbox_mask, self._recent_valid_bbox_categories
                    return None, None, None
                    
                # Use the first valid timestamp
                timestamp_ns = valid_timestamps[0]
                
            # Get 3D bounding boxes at this timestamp
            bbox3d_with_dt = self.adt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamp_ns)
            
            if not bbox3d_with_dt.is_valid():
                print(f"Warning: No valid bounding boxes at timestamp {timestamp_ns}.")
                # Try to use recent bounding boxes as fallback
                if use_recent_fallback and self._recent_valid_bboxes is not None:
                    # Check if the recent bboxes are within acceptable time range
                    if self._recent_valid_bboxes_timestamp is not None and abs(timestamp_ns - self._recent_valid_bboxes_timestamp) < self._max_bbox_time_delta_ns:
                        print(f"Using recent bounding boxes from timestamp {self._recent_valid_bboxes_timestamp}")
                        return self._recent_valid_bboxes, self._recent_valid_bbox_mask, self._recent_valid_bbox_categories
                return None, None, None
                
            # Get the bounding box data
            bboxes3d = bbox3d_with_dt.data()
            
            # Extract center positions, dimensions, and orientation from all bounding boxes
            bbox_info = []
            bbox_categories = []  # New list to store categories
            
            for obj_id, bbox in bboxes3d.items():
                try:
                    # Get the center position through the translation method
                    center = bbox.transform_scene_object.translation()[0]
                    
                    # Get the dimensions from the AABB
                    # AABB format is [xmin, xmax, ymin, ymax, zmin, zmax]
                    aabb = bbox.aabb
                    width = aabb[1] - aabb[0]   # xmax - xmin
                    height = aabb[3] - aabb[2]  # ymax - ymin
                    depth = aabb[5] - aabb[4]   # zmax - zmin
                    
                    # Adjust center from bottom center to geometric center by moving up by half height
                    # ADT bbox center is at the bottom, we need to move it to geometric center
                    center = [center[0], center[1] + height / 2.0, center[2]]
                    
                    # Extract rotation matrix from transform_scene_object
                    rotation_matrix = bbox.transform_scene_object.rotation().to_matrix()
                    
                    # Use 6D rotation representation
                    rotation_rep = rotation_matrix_to_6d(rotation_matrix)
                    
                    # Normalize center and dimensions if normalize_data is True
                    if self.normalize_data:
                        center = self.normalize_position(center)
                        dimensions = self.normalize_size(np.array([width, height, depth]))
                        width, height, depth = dimensions
                    
                    # Add the normalized center, dimensions, and rotation representation to our list
                    # 6D rotation format: [cx, cy, cz, width, height, depth, r1, r2, r3, r4, r5, r6]
                    bbox_info_entry = [
                        center[0], center[1], center[2],
                        width, height, depth,
                        rotation_rep[0], rotation_rep[1], rotation_rep[2], 
                        rotation_rep[3], rotation_rep[4], rotation_rep[5]
                    ]
                    
                    bbox_info.append(bbox_info_entry)
                    
                    # Get object category if available
                    category = "unknown"
                    try:
                        # Get instance info directly - this is more reliable for categories
                        instance_info = self.adt_provider.get_instance_info_by_id(obj_id)
                        if instance_info:
                            # First try the direct category attribute (like in _process_trajectory_segment)
                            category = getattr(instance_info, 'category', 'unknown')
                            
                            # If still unknown, try other potential category fields
                            if category == "unknown" and hasattr(instance_info, 'semantic_category'):
                                category = instance_info.semantic_category
                            elif category == "unknown" and hasattr(instance_info, 'object_type'):
                                category = instance_info.object_type
                            elif category == "unknown" and hasattr(instance_info, 'type'):
                                category = instance_info.type
                            
                            # If still unknown, try to get the prototype_name
                            if category == "unknown" and hasattr(instance_info, 'prototype_name'):
                                category = instance_info.prototype_name
                        
                    except Exception as e:
                        # Log error but continue
                        print(f"Warning: Failed to get category for object {obj_id}: {e}")
                    
                    bbox_categories.append(category)
                    
                except Exception as e:
                    print(f"Warning: Failed to get bbox info for object {obj_id}: {e}")
                    continue
                    
            if not bbox_info:
                print(f"Warning: No valid bounding box info extracted at timestamp {timestamp_ns}.")
                # Try to use recent bounding boxes as fallback
                if use_recent_fallback and self._recent_valid_bboxes is not None:
                    # Check if the recent bboxes are within acceptable time range
                    if self._recent_valid_bboxes_timestamp is not None and abs(timestamp_ns - self._recent_valid_bboxes_timestamp) < self._max_bbox_time_delta_ns:
                        print(f"Using recent bounding boxes from timestamp {self._recent_valid_bboxes_timestamp}")
                        return self._recent_valid_bboxes, self._recent_valid_bbox_mask, self._recent_valid_bbox_categories
                return None, None, None
                
            # Convert to tensor
            bbox_tensor = torch.tensor(bbox_info, dtype=torch.float32)
            
            # Count actual bounding boxes
            num_actual_bboxes = len(bbox_info)
            # print(f"Found {num_actual_bboxes} bounding boxes in scene at timestamp {timestamp_ns}")
            
            # Create mask for real vs. padding boxes (1 for real, 0 for padding)
            mask = torch.zeros(self.max_bboxes, dtype=torch.float32)
            
            # Create padded category list - fill with empties for padding boxes
            padded_categories = ["unknown"] * self.max_bboxes
            
            # If we have more boxes than max, truncate
            if num_actual_bboxes > self.max_bboxes:
                print(f"Truncating from {num_actual_bboxes} to {self.max_bboxes} bounding boxes")
                bbox_tensor = bbox_tensor[:self.max_bboxes]
                mask[:self.max_bboxes] = 1.0
                # Also truncate the categories
                bbox_categories = bbox_categories[:self.max_bboxes]
                for i, cat in enumerate(bbox_categories):
                    padded_categories[i] = cat
            else:
                # Create padded tensor to fill with zeros
                padded_boxes = torch.zeros(self.max_bboxes, 12, dtype=torch.float32)
                padded_boxes[:num_actual_bboxes] = bbox_tensor
                # Set 6D representation to identity rotation for padding boxes
                # First 3 components (first column): [1,0,0]
                padded_boxes[num_actual_bboxes:, 6] = 1.0  # rx1 = 1
                padded_boxes[num_actual_bboxes:, 7] = 0.0  # ry1 = 0
                padded_boxes[num_actual_bboxes:, 8] = 0.0  # rz1 = 0
                # Next 3 components (second column): [0,1,0]
                padded_boxes[num_actual_bboxes:, 9] = 0.0   # rx2 = 0
                padded_boxes[num_actual_bboxes:, 10] = 1.0  # ry2 = 1
                padded_boxes[num_actual_bboxes:, 11] = 0.0  # rz2 = 0
                
                bbox_tensor = padded_boxes
                mask[:num_actual_bboxes] = 1.0
                # Add categories to padded list
                for i, cat in enumerate(bbox_categories):
                    padded_categories[i] = cat
            
            # Cache these bounding boxes for future use
            self._recent_valid_bboxes = bbox_tensor
            self._recent_valid_bboxes_timestamp = timestamp_ns
            self._recent_valid_bbox_mask = mask
            self._recent_valid_bbox_categories = padded_categories
                
            return bbox_tensor, mask, padded_categories
            
        except Exception as e:
            print(f"Error extracting bounding boxes at timestamp {timestamp_ns}: {e}")
            traceback.print_exc()
            # Try to use recent bounding boxes as fallback
            if use_recent_fallback and self._recent_valid_bboxes is not None:
                print(f"Using recent bounding boxes from timestamp {self._recent_valid_bboxes_timestamp} after error")
                if hasattr(self, '_recent_valid_bbox_categories'):
                    return self._recent_valid_bboxes, self._recent_valid_bbox_mask, self._recent_valid_bbox_categories
                else:
                    # Create default categories if not previously stored
                    default_categories = ["unknown"] * self.max_bboxes
                    return self._recent_valid_bboxes, self._recent_valid_bbox_mask, default_categories
            return None, None, None

    def get_scene_bboxes(self, batch_size=1):
        """
        Get scene bounding boxes for training or sampling.
        
        This method provides bounding boxes for the initial timestamp of the sequence.
        Note that for individual trajectories accessed via __getitem__, each motion segment
        uses bounding boxes from its specific start timestamp for more accurate scene context.
        
        Args:
            batch_size: Number of copies of the bounding box information to return
        
        Returns:
            Tuple of (
                torch.Tensor: Bounding box information [batch_size, max_bboxes, bbox_dim],
                torch.Tensor: Mask indicating which boxes are real (1) vs padding (0) [batch_size, max_bboxes],
                List[List[str]]: Categories for each bounding box [batch_size, max_bboxes]
            ) or (None, None, None) if no bounding box information is available
        """
        # Get bounding boxes (load or compute if not already available)
        if not hasattr(self, '_scene_bboxes') or self._scene_bboxes is None:
            self._scene_bboxes, self._scene_bboxes_mask, self._scene_bboxes_categories = self.get_all_scene_bboxes()
        
        if self._scene_bboxes is None:
            return None, None, None
        
        # Repeat for batch size
        if batch_size > 1:
            bboxes = self._scene_bboxes.unsqueeze(0).repeat(batch_size, 1, 1)
            mask = self._scene_bboxes_mask.unsqueeze(0).repeat(batch_size, 1)
            # For categories, we need to repeat the list
            categories = [self._scene_bboxes_categories for _ in range(batch_size)]
            return bboxes, mask, categories
        else:
            return self._scene_bboxes.unsqueeze(0), self._scene_bboxes_mask.unsqueeze(0), [self._scene_bboxes_categories]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Get a trajectory sample.
        
        Args:
            idx: Index of the trajectory
        
        Returns:
            dict: Trajectory data including poses (positions+orientations), attention mask, and basic metadata
        """
        item = self.trajectories[idx]
        trajectory_data = item['trajectory_data']
        metadata = item['metadata']
        
        # Apply transforms if any
        if self.transform:
            trajectory_data = self.transform(trajectory_data)
        
        # Create the result with minimal metadata and split trajectories
        result = {
            'poses': trajectory_data['poses'],  # [N, 9] tensor with [x,y,z,r6d_1,r6d_2,r6d_3,r6d_4,r6d_5,r6d_6]
            'attention_mask': trajectory_data['attention_mask'],
            'object_type': metadata['prototype_name'],
            'object_name': metadata['name'],
            'object_id': metadata['id'],
            'object_category': metadata['category'],
            'object_subcategory': metadata['subcategory'],
            'sequence': metadata['sequence'],
            'is_displacement': metadata.get('is_displacement', False)
        }
        
        # Add pre-computed CLIP features if available
        if self.use_clip_cache and 'object_category_clip' in item:
            result['object_category_clip'] = item['object_category_clip']
            logger.debug(f"[CLIP CACHE] Using pre-computed CLIP features for trajectory {idx}")
        elif self.use_clip_cache and self.clip_feature_manager:
            # Compute CLIP features on-the-fly if not cached
            obj_cat = metadata.get('category', metadata.get('name', 'unknown'))
            clip_feature = self.clip_feature_manager.get_feature_for_text(obj_cat)
            result['object_category_clip'] = clip_feature
            logger.debug(f"[CLIP CACHE] Computed CLIP features on-the-fly for trajectory {idx}")
        
        # TODO: Add scene bbox CLIP features handling if needed
        
        # Include trajectory-specific point cloud if available
        if 'trajectory_specific_pointcloud' in item:
            result['trajectory_specific_pointcloud'] = item['trajectory_specific_pointcloud']
        
        # Add normalization parameters to the sample
        if 'normalization' in metadata:
             # Convert normalization numpy arrays to tensors for collation
             norm_info = metadata['normalization']
             result['normalization'] = {
                 'is_normalized': norm_info['is_normalized'],
                 'scene_min': torch.tensor(norm_info.get('scene_min', [0.0, 0.0, 0.0]), dtype=torch.float32),
                 'scene_max': torch.tensor(norm_info.get('scene_max', [1.0, 1.0, 1.0]), dtype=torch.float32),
                 'scene_scale': torch.tensor(norm_info.get('scene_scale', [1.0, 1.0, 1.0]), dtype=torch.float32)
             }
        else:
             # Include default normalization parameters if not found in metadata
             result['normalization'] = {
                 'is_normalized': self.normalize_data,
                 'scene_min': torch.tensor(self.scene_min, dtype=torch.float32),
                 'scene_max': torch.tensor(self.scene_max, dtype=torch.float32),
                 'scene_scale': torch.tensor(self.scene_scale, dtype=torch.float32)
             }

        # Add segment index if available in metadata
        if 'segment_idx' in metadata:
            result['segment_idx'] = metadata['segment_idx']

        # Retrieve bbox_corners from trajectory_data and add to result
        if 'bbox_corners' in trajectory_data:
            result['bbox_corners'] = trajectory_data['bbox_corners']
        else:
            # Fallback if not present, though it should be due to _process_trajectory_segment
            result['bbox_corners'] = torch.zeros(self.trajectory_length, 8, 3, dtype=torch.float32)

        # Add scene-level bbox information - prioritize trajectory-specific bboxes
        if 'trajectory_specific_bbox_info' in item:
            # Use trajectory-specific bboxes (filtered by trajectory range)
            result['scene_bbox_info'] = item['trajectory_specific_bbox_info']
            result['scene_bbox_mask'] = item['trajectory_specific_bbox_mask']
            result['scene_bbox_categories'] = item['trajectory_specific_bbox_categories']
        elif hasattr(self, '_scene_bboxes') and self._scene_bboxes is not None:
            # Fallback to full scene bboxes if trajectory-specific ones are not available
            result['scene_bbox_info'] = self._scene_bboxes
            result['scene_bbox_mask'] = self._scene_bboxes_mask
            result['scene_bbox_categories'] = self._scene_bboxes_categories
        else:
            # Create empty scene bbox data if not available
            result['scene_bbox_info'] = torch.zeros(self.max_bboxes, 12, dtype=torch.float32)
            # Set identity rotation in 6D representation (first two columns of identity matrix)
            result['scene_bbox_info'][:, 6] = 1.0   # rx1 = 1
            result['scene_bbox_info'][:, 10] = 1.0  # ry2 = 1
            result['scene_bbox_mask'] = torch.zeros(self.max_bboxes, dtype=torch.float32)
            result['scene_bbox_categories'] = ["unknown"] * self.max_bboxes

        return result
    
    def get_trajectory_specific_pointcloud(self, idx):
        """
        Get the trajectory-specific pointcloud for a specific trajectory.
        
        Args:
            idx: Index of the trajectory
            
        Returns:
            Numpy array of point cloud coordinates or None if not available
        """
        if idx < 0 or idx >= len(self.trajectories):
            print(f"Error: Index {idx} out of range [0, {len(self.trajectories)-1}]")
            return None
            
        item = self.trajectories[idx]
        if 'trajectory_specific_pointcloud' in item:
            return item['trajectory_specific_pointcloud']
        else:
            print(f"Warning: No trajectory-specific pointcloud found for trajectory {idx}")
            return None

    def get_trajectory_specific_bboxes(self, idx):
        """
        Get the trajectory-specific bboxes for a specific trajectory.
        
        Args:
            idx: Index of the trajectory
            
        Returns:
            Tuple of (
                torch.Tensor: Filtered bounding box information [max_bboxes, 12],
                torch.Tensor: Mask indicating which boxes are real (1) vs padding (0) [max_bboxes],
                List[str]: Categories for each bounding box [max_bboxes]
            ) or (None, None, None) if not available
        """
        if idx < 0 or idx >= len(self.trajectories):
            print(f"Error: Index {idx} out of range [0, {len(self.trajectories)-1}]")
            return None, None, None
            
        item = self.trajectories[idx]
        if ('trajectory_specific_bbox_info' in item and 
            'trajectory_specific_bbox_mask' in item and 
            'trajectory_specific_bbox_categories' in item):
            return (item['trajectory_specific_bbox_info'], 
                    item['trajectory_specific_bbox_mask'], 
                    item['trajectory_specific_bbox_categories'])
        else:
            print(f"Warning: No trajectory-specific bboxes found for trajectory {idx}")
            return None, None, None

    def _filter_bboxes_by_trajectory(self, trajectory_positions, radius=None, timestamp_ns=None):
        """
        Filter scene bounding boxes to include only those within a specified radius of any trajectory point.
        
        Args:
            trajectory_positions: Numpy array of shape [N, 3] containing trajectory positions
            radius: Radius in meters around each trajectory point to include scene bboxes (defaults to self.trajectory_pointcloud_radius)
            timestamp_ns: Optional timestamp for which to extract bounding boxes
            
        Returns:
            Tuple of (
                torch.Tensor: Filtered bounding box information [M, 12] where M <= max_bboxes,
                torch.Tensor: Mask indicating which boxes are real (1) vs padding (0) [max_bboxes],
                List[str]: Categories for each bounding box [max_bboxes]
            ) or (None, None, None) if no bounding box information is available
        """
        if trajectory_positions is None or len(trajectory_positions) == 0:
            print("Warning: Empty trajectory provided for bbox filtering")
            return None, None, None
            
        # Use instance default radius if not specified
        if radius is None:
            radius = self.trajectory_pointcloud_radius
        
        try:
            # Get all scene bboxes at the specified timestamp
            all_bboxes, all_mask, all_categories = self.get_all_scene_bboxes(timestamp_ns=timestamp_ns, use_recent_fallback=True)
            
            if all_bboxes is None:
                return None, None, None
            
            # Convert trajectory positions to numpy if they're tensors
            if isinstance(trajectory_positions, torch.Tensor):
                trajectory_positions = trajectory_positions.detach().cpu().numpy()
                
            # If normalized, denormalize trajectory positions for consistent distance calculation
            if self.normalize_data:
                trajectory_positions = self.denormalize_position(trajectory_positions)
            
            # Extract bbox centers (first 3 dimensions) from all_bboxes
            bbox_centers = all_bboxes[:, :3].numpy()  # [max_bboxes, 3]
            
            # If bboxes were normalized, denormalize them for distance calculation
            if self.normalize_data:
                bbox_centers = self.denormalize_position(bbox_centers)
            
            # Use KDTree for efficient distance calculation
            from scipy.spatial import cKDTree
            trajectory_tree = cKDTree(trajectory_positions)
            
            # Find distances from each bbox center to the nearest trajectory point
            distances, _ = trajectory_tree.query(bbox_centers, k=1)
            
            # Create boolean mask for bboxes within radius and also valid (non-padding)
            within_radius_mask = distances <= radius
            valid_bbox_mask = all_mask.numpy() > 0.5  # Only consider non-padding bboxes
            final_mask = within_radius_mask & valid_bbox_mask
            
            # Get indices of bboxes that are within range
            valid_indices = np.where(final_mask)[0]
            
            if len(valid_indices) == 0:
                print(f"Warning: No bboxes found within {radius}m of trajectory")
                # Return empty but properly formatted result
                empty_bboxes = torch.zeros(self.max_bboxes, 12, dtype=torch.float32)
                empty_bboxes[:, 6] = 1.0   # rx1 = 1 (identity rotation)
                empty_bboxes[:, 10] = 1.0  # ry2 = 1 (identity rotation)
                empty_mask = torch.zeros(self.max_bboxes, dtype=torch.float32)
                empty_categories = ["unknown"] * self.max_bboxes
                return empty_bboxes, empty_mask, empty_categories
            
            # Create filtered results
            num_filtered_bboxes = len(valid_indices)
            
            # Create new tensors for filtered results
            filtered_bboxes = torch.zeros(self.max_bboxes, 12, dtype=torch.float32)
            filtered_mask = torch.zeros(self.max_bboxes, dtype=torch.float32)
            filtered_categories = ["unknown"] * self.max_bboxes
            
            # Set identity rotation for padding boxes
            filtered_bboxes[:, 6] = 1.0   # rx1 = 1
            filtered_bboxes[:, 10] = 1.0  # ry2 = 1
            
            # Copy filtered bboxes to the result (limit to max_bboxes)
            copy_count = min(num_filtered_bboxes, self.max_bboxes)
            for i in range(copy_count):
                src_idx = valid_indices[i]
                filtered_bboxes[i] = all_bboxes[src_idx]
                filtered_mask[i] = 1.0
                filtered_categories[i] = all_categories[src_idx]
            
            print(f"Filtered bboxes: {copy_count} boxes within {radius}m of trajectory (from {int(valid_bbox_mask.sum())} total valid boxes)")
            
            return filtered_bboxes, filtered_mask, filtered_categories
            
        except Exception as e:
            print(f"Error filtering bboxes by trajectory: {e}")
            traceback.print_exc()
            return None, None, None
    
    def _generate_trajectory_specific_bboxes(self):
        """
        Generate and store trajectory-specific bboxes for each trajectory in the dataset.
        These filtered bboxes only include boxes within trajectory_pointcloud_radius
        of any point in the trajectory.
        """
        print(f"Generating trajectory-specific bounding boxes for {len(self.trajectories)} trajectories...")
        
        for i, traj_item in enumerate(self.trajectories):
            try:
                # Get trajectory poses and extract positions (first 3 dimensions)
                poses_tensor = traj_item['trajectory_data']['poses']
                positions_tensor = poses_tensor[:, :3]  # Extract first 3 dimensions (x, y, z)
                
                # Get attention mask to filter out padding
                mask = traj_item['trajectory_data']['attention_mask']
                
                # Extract only valid positions (where mask is 1)
                if isinstance(mask, torch.Tensor):
                    valid_indices = torch.where(mask > 0.5)[0]
                    valid_positions = positions_tensor[valid_indices]
                else:
                    # Fallback if mask isn't a tensor
                    valid_indices = np.where(np.array(mask) > 0.5)[0]
                    valid_positions = positions_tensor[valid_indices]
                
                # If normalized, denormalize positions for accurate distance calculation
                if self.normalize_data and isinstance(valid_positions, torch.Tensor):
                    # Convert to numpy, denormalize, then back to tensor if needed
                    positions_np = valid_positions.detach().cpu().numpy()
                    positions_np = self.denormalize_position(positions_np)
                    valid_positions = positions_np
                elif self.normalize_data:
                    valid_positions = self.denormalize_position(valid_positions)
                
                # Get the trajectory's start timestamp for bbox context
                start_timestamp = traj_item['metadata'].get('start_timestamp_ns')
                
                # Filter the bboxes
                filtered_bboxes, filtered_mask, filtered_categories = self._filter_bboxes_by_trajectory(
                    valid_positions, 
                    radius=self.trajectory_pointcloud_radius,
                    timestamp_ns=start_timestamp
                )
                
                # Store the filtered bboxes with the trajectory
                if filtered_bboxes is not None:
                    traj_item['trajectory_specific_bbox_info'] = filtered_bboxes
                    traj_item['trajectory_specific_bbox_mask'] = filtered_mask
                    traj_item['trajectory_specific_bbox_categories'] = filtered_categories
                else:
                    # Create empty bbox data if filtering failed
                    empty_bboxes = torch.zeros(self.max_bboxes, 12, dtype=torch.float32)
                    empty_bboxes[:, 6] = 1.0   # rx1 = 1 (identity rotation)
                    empty_bboxes[:, 10] = 1.0  # ry2 = 1 (identity rotation)
                    empty_mask = torch.zeros(self.max_bboxes, dtype=torch.float32)
                    empty_categories = ["unknown"] * self.max_bboxes
                    
                    traj_item['trajectory_specific_bbox_info'] = empty_bboxes
                    traj_item['trajectory_specific_bbox_mask'] = empty_mask
                    traj_item['trajectory_specific_bbox_categories'] = empty_categories
                
                # For progress reporting
                if (i+1) % 10 == 0 or i == 0 or i == len(self.trajectories)-1:
                    print(f"  Processed {i+1}/{len(self.trajectories)} trajectories for bbox filtering")
                
            except Exception as e:
                print(f"Error generating trajectory-specific bboxes for trajectory {i}: {e}")
                traceback.print_exc()
                # Set empty bbox data as fallback
                empty_bboxes = torch.zeros(self.max_bboxes, 12, dtype=torch.float32)
                empty_bboxes[:, 6] = 1.0   # rx1 = 1 (identity rotation)
                empty_bboxes[:, 10] = 1.0  # ry2 = 1 (identity rotation)
                empty_mask = torch.zeros(self.max_bboxes, dtype=torch.float32)
                empty_categories = ["unknown"] * self.max_bboxes
                
                traj_item['trajectory_specific_bbox_info'] = empty_bboxes
                traj_item['trajectory_specific_bbox_mask'] = empty_mask
                traj_item['trajectory_specific_bbox_categories'] = empty_categories
        
        print(f"Completed generating trajectory-specific bounding boxes")
