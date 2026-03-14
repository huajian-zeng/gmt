#!/usr/bin/env python3
"""Evaluation script for trajectory prediction model."""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import time
import logging
from functools import partial

from config.config import Config
from model.trajectory_model import TrajectoryModel
from dataset.adt_dataset import ADTDataset
from utils.visualization import visualize_trajectory, visualize_prediction
from utils.metrics_utils import (
    transform_coords_for_visualization,
    compute_metrics_for_sample,
    compute_bbox_trajectory_collision,
    check_gt_trajectory_collision,
    eval_collate_fn
)

try:
    from utils.adt_sequence_utils import find_adt_sequences
    HAS_SEQ_UTILS = True
except ImportError:
    HAS_SEQ_UTILS = False

try:
    from utils.rerun_visualization import (
        initialize_rerun, visualize_trajectory_rerun, save_rerun_recording,
        downsample_point_cloud, extract_trajectory_specific_point_cloud, HAS_RERUN
    )
    import rerun as rr
except ImportError:
    HAS_RERUN = False


def get_coord_mapping_for_dataset(config):
    """Get coordinate mapping based on dataset type."""
    return ['x', '-z', 'y']


def setup_logger(output_dir):
    """Set up logger for console and file output."""
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(output_dir, 'evaluation.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trajectory prediction model")

    # Model and output
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")

    # Dataset
    parser.add_argument("--adt_dataroot", type=str, default=None, help="Override ADT dataroot")
    parser.add_argument("--test_split_file", type=str, default=None, help="Override test split file")
    parser.add_argument("--global_cache_dir", type=str, default=None, help="Directory with pre-computed cache files")
    parser.add_argument("--force_use_cache", action="store_true", help="Force use cache files without parameter validation")
    parser.add_argument("--sequences", type=str, nargs='+', default=None, help="Specific sequence names to evaluate (e.g., Apartment_release_work_seq136_M1292)")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Data loading workers")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Visualization
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--num_vis_samples", type=int, default=100, help="Samples to visualize")
    parser.add_argument("--use_rerun", action="store_true", help="Enable Rerun visualization")

    # Rerun visualization options
    parser.add_argument("--pointcloud_downsample_factor", type=int, default=1, help="Factor for downsampling point clouds in Rerun visualization")
    parser.add_argument("--use_trajectory_pointcloud", type=float, default=None, metavar='RADIUS', help="Extract trajectory-specific point cloud with specified radius")
    parser.add_argument("--rerun_line_width", type=float, default=0.02, help="Width of lines in Rerun visualization")
    parser.add_argument("--rerun_point_size", type=float, default=0.03, help="Size of points in Rerun visualization")
    parser.add_argument("--rerun_show_arrows", action="store_true", default=True, help="Show orientation arrows in Rerun visualization")
    parser.add_argument("--rerun_arrow_length", type=float, default=0.2, help="Length of orientation arrows in Rerun visualization")

    # Model overrides
    parser.add_argument("--use_first_frame_only", action="store_true", help="Use only first frame as input")

    # Dataset parameter overrides
    parser.add_argument("--detect_motion_segments", action="store_true", default=None, help="Enable motion segment detection")
    parser.add_argument("--no_detect_motion_segments", action="store_true", help="Disable motion segment detection")

    return parser.parse_args()


def load_model_and_config(checkpoint_path, device, logger):
    """Load model and config from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config'")

    config = argparse.Namespace(**checkpoint['config'])
    model = TrajectoryModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_epoch = checkpoint.get('epoch', 0)
    logger.info(f"Loaded model from epoch {best_epoch}")

    return model, config, best_epoch


def prepare_batch(batch, config, device):
    """Prepare batch data for model forward pass (same as train.py)."""
    full_trajectory = batch['full_poses'].float().to(device)
    point_cloud = batch['point_cloud'].float().to(device)
    attention_mask = batch['full_attention_mask'].to(device)

    bbox_corners = None
    if not config.no_bbox:
        bbox_corners = batch['bbox_corners'].float().to(device)

    actual_lengths = attention_mask.sum(dim=1).int()

    # Determine history length
    if config.use_first_frame_only:
        hist_len = 1 if torch.any(actual_lengths > 0) else 0
    else:
        hist_lengths = (actual_lengths.float() * config.history_fraction).floor().long()
        hist_lengths = torch.clamp(hist_lengths, min=1)
        hist_lengths = torch.min(hist_lengths, actual_lengths)
        hist_len = min(hist_lengths.max().item(), full_trajectory.shape[1])

    input_trajectory = full_trajectory[:, :hist_len, :]
    bbox_input = bbox_corners[:, :hist_len, :, :] if bbox_corners is not None else None

    # Object category
    object_category = None
    if not config.no_text_embedding:
        object_category = batch.get('object_category_clip', batch.get('object_category'))
        if isinstance(object_category, torch.Tensor):
            object_category = object_category.to(device)

    # Semantic bbox
    semantic_bbox_info, semantic_bbox_mask = None, None
    if not config.no_semantic_bbox:
        semantic_bbox_info = batch['scene_bbox_info'].float().to(device)
        semantic_bbox_mask = batch['scene_bbox_mask'].float().to(device)

    # Semantic text
    semantic_text = None
    if not getattr(config, 'no_semantic_text', False):
        semantic_text = batch.get('scene_bbox_categories_clip', batch.get('scene_bbox_categories'))
        if isinstance(semantic_text, torch.Tensor):
            semantic_text = semantic_text.to(device)

    # End pose
    end_pose = None
    if not getattr(config, 'no_end_pose', False):
        end_poses = []
        for i in range(full_trajectory.shape[0]):
            actual_len = attention_mask[i].sum().int().item()
            if actual_len > 0:
                end_poses.append(full_trajectory[i, actual_len-1:actual_len, :])
            else:
                end_poses.append(torch.zeros(1, full_trajectory.shape[2], device=device))
        end_pose = torch.cat(end_poses, dim=0)

    return {
        'input_trajectory': input_trajectory,
        'point_cloud': point_cloud,
        'bbox_corners': bbox_input,
        'object_category': object_category,
        'semantic_bbox_info': semantic_bbox_info,
        'semantic_bbox_mask': semantic_bbox_mask,
        'semantic_text': semantic_text,
        'end_pose': end_pose,
        'full_trajectory': full_trajectory,
        'attention_mask': attention_mask,
        'hist_len': hist_len,
    }


def evaluate(model, config, args, logger):
    """Run evaluation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    coord_mapping = get_coord_mapping_for_dataset(config)

    # Setup dataset
    logger.info("Setting up test dataset...")
    checkpoint_dir = os.path.dirname(args.model_path)
    val_split_path = os.path.join(checkpoint_dir, 'val_sequences.txt')

    test_sequences = None

    # Priority 1: Use --sequences argument if provided (sequence names only, for cache-only mode)
    if args.sequences:
        test_sequences = args.sequences
        logger.info(f"Using specified sequences: {test_sequences}")
    # Priority 2: Load from val_sequences.txt in checkpoint dir
    elif os.path.exists(val_split_path):
        with open(val_split_path, 'r') as f:
            test_sequences = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(test_sequences)} test sequences from checkpoint")
    # Priority 3: Use adt_dataroot
    elif args.adt_dataroot or hasattr(config, 'adt_dataroot'):
        dataroot = args.adt_dataroot or config.adt_dataroot
        if os.path.isdir(dataroot) and HAS_SEQ_UTILS:
            test_sequences = find_adt_sequences(dataroot)
        elif os.path.isdir(dataroot):
            test_sequences = [os.path.join(dataroot, d) for d in os.listdir(dataroot)
                            if os.path.isdir(os.path.join(dataroot, d))]
        else:
            test_sequences = [dataroot]
        logger.info(f"Found {len(test_sequences)} test sequences")

    if not test_sequences:
        logger.error("No test sequences found")
        return None

    # Use global_cache_dir if provided, otherwise use output_dir/cache
    if args.global_cache_dir:
        cache_dir = args.global_cache_dir
        logger.info(f"Using global cache directory: {cache_dir}")
    else:
        cache_dir = os.path.join(args.output_dir, 'cache')
        logger.info(f"Using evaluation-specific cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    test_dataset = ADTDataset(sequence_paths=test_sequences, config=config, cache_dir=cache_dir)

    if len(test_dataset) == 0:
        logger.error("Test dataset is empty")
        return None

    eval_collate = partial(eval_collate_fn, dataset=test_dataset, num_sample_points=config.sample_points)
    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=eval_collate)
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Metrics
    total_l1, total_rmse, total_fde, total_frechet, total_angular = 0.0, 0.0, 0.0, 0.0, 0.0
    total_collision = 0.0
    total_gt_collisions = 0  # Track GT trajectories with collisions
    total_valid_for_collision_eval = 0  # Track trajectories valid for collision evaluation
    total_samples = 0
    all_metrics = []

    # Visualization
    visualize = not args.no_visualize
    vis_dir = os.path.join(args.output_dir, "visualizations")
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    visualized_count = 0
    evaluated_samples = 0

    # Rerun visualization setup
    per_sample_rrd_basedir = None
    use_rerun = getattr(config, 'use_rerun', True) or args.use_rerun  # Default to True
    logger.info(f"Rerun visualization: use_rerun={use_rerun}, HAS_RERUN={HAS_RERUN}")
    if use_rerun and HAS_RERUN:
        per_sample_rrd_basedir = os.path.join(args.output_dir, "rerun_visualizations")
        os.makedirs(per_sample_rrd_basedir, exist_ok=True)
        logger.info(f"Rerun per-sample .rrd files will be saved to: {per_sample_rrd_basedir}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            if args.max_eval_samples and evaluated_samples >= args.max_eval_samples:
                break

            try:
                prepared = prepare_batch(batch, config, device)
            except Exception as e:
                logger.warning(f"Error preparing batch {batch_idx}: {e}")
                continue

            # Model inference
            predicted = model(
                prepared['input_trajectory'], prepared['point_cloud'],
                prepared['bbox_corners'], prepared['object_category'],
                prepared['semantic_bbox_info'], prepared['semantic_bbox_mask'],
                prepared['semantic_text'], prepared['end_pose']
            )

            # Process each sample
            batch_size = prepared['full_trajectory'].shape[0]
            for i in range(batch_size):
                gt_full = prepared['full_trajectory'][i]
                pred_full = predicted[i]
                mask = prepared['attention_mask'][i]
                actual_len = mask.sum().int().item()

                if actual_len < 2:
                    continue

                # Determine history length for this sample
                if config.use_first_frame_only:
                    hist_len = 1
                else:
                    hist_len = max(1, int(actual_len * config.history_fraction))

                # Extract position component
                gt_future = gt_full[hist_len:actual_len, :3]
                pred_future = pred_full[hist_len:actual_len, :3]
                future_mask = mask[hist_len:actual_len]

                if future_mask.sum() == 0:
                    continue

                # Compute metrics
                l1, rmse, fde, frechet, angular = compute_metrics_for_sample(
                    pred_future, gt_future, future_mask
                )

                # Collision detection (following GIMO methodology)
                # For GT collision detection: always use the full future trajectory based on history_fraction
                # This ensures consistent GT collision detection across different evaluation modes
                gt_collision_history_length = max(1, int(actual_len * config.history_fraction))
                gt_collision_history_length = min(gt_collision_history_length, actual_len - 1)
                gt_collision_future_mask = mask[gt_collision_history_length:actual_len]

                collision_rate = 0.0
                gt_has_collision = False
                if prepared['semantic_bbox_info'] is not None:
                    sample_bbox_info = batch['scene_bbox_info'][i].float().to(device)
                    sample_bbox_mask = batch['scene_bbox_mask'][i].float().to(device)

                    # First check if GT trajectory has collision
                    gt_collision_future = gt_full[gt_collision_history_length:actual_len, :3]
                    gt_has_collision, _ = check_gt_trajectory_collision(
                        gt_collision_future,
                        sample_bbox_info,
                        sample_bbox_mask,
                        gt_collision_future_mask
                    )

                    # Compute collision for predicted future trajectory (not full trajectory)
                    pred_collision, _ = compute_bbox_trajectory_collision(
                        pred_future,
                        sample_bbox_info,
                        sample_bbox_mask,
                        future_mask
                    )

                    # Only count this trajectory for collision evaluation if GT has no collision
                    if gt_has_collision:
                        collision_rate = -1.0  # Mark as excluded
                    else:
                        collision_rate = pred_collision.item()

                # Accumulate
                total_l1 += l1.item()
                total_rmse += rmse.item()
                total_fde += fde.item()
                total_frechet += frechet.item()
                total_angular += angular.item()

                # Only accumulate collision rate for valid trajectories (GT has no collision)
                if not gt_has_collision:
                    total_collision += collision_rate
                    total_valid_for_collision_eval += 1
                else:
                    total_gt_collisions += 1
                total_samples += 1

                all_metrics.append({
                    'batch_idx': batch_idx, 'sample_idx': i,
                    'l1': l1.item(), 'rmse': rmse.item(), 'fde': fde.item(),
                    'frechet': frechet.item(), 'angular': angular.item(),
                    'collision': collision_rate,
                    'gt_has_collision': gt_has_collision
                })

                # Visualization
                if visualize and visualized_count < args.num_vis_samples:
                    obj_name = batch.get('object_name', ['unknown'])[i] if 'object_name' in batch else f'obj{i}'
                    seg_idx = batch.get('segment_idx', [0])[i] if 'segment_idx' in batch else 0

                    gt_pos = transform_coords_for_visualization(gt_full[:actual_len, :3].cpu(), coord_mapping)
                    pred_pos = transform_coords_for_visualization(pred_full[:actual_len, :3].cpu(), coord_mapping)

                    save_path = os.path.join(vis_dir, f"{obj_name}_seg{seg_idx}_b{batch_idx}_s{i}.png")
                    visualize_prediction(
                        past_positions=gt_pos[:hist_len],
                        future_positions_gt=gt_pos[hist_len:],
                        future_positions_pred=pred_pos[hist_len:],
                        title=f"{obj_name} (FDE: {fde.item():.3f}m)",
                        save_path=save_path
                    )

                    # Rerun visualization - save .rrd file per sample
                    if use_rerun and HAS_RERUN and per_sample_rrd_basedir:
                        try:
                            filename_base = f"{obj_name}_seg{seg_idx}_b{batch_idx}_s{i}"

                            # Initialize rerun for this sample
                            rerun_initialized, rrd_path = initialize_rerun(
                                recording_name=filename_base,
                                spawn=False,
                                output_dir=per_sample_rrd_basedir
                            )

                            if rerun_initialized:
                                # Get point cloud with proper processing
                                pc_vis = None
                                if 'point_cloud' in batch and batch['point_cloud'] is not None:
                                    point_cloud_raw = batch['point_cloud'][i].cpu()

                                    # Downsample point cloud if requested
                                    if args.pointcloud_downsample_factor > 1:
                                        point_cloud_raw = downsample_point_cloud(
                                            point_cloud_raw.numpy(),
                                            args.pointcloud_downsample_factor
                                        )
                                        point_cloud_raw = torch.from_numpy(point_cloud_raw)

                                    # Extract trajectory-specific point cloud if requested
                                    if args.use_trajectory_pointcloud is not None:
                                        full_traj = gt_full[:actual_len, :3].cpu()
                                        point_cloud_raw = extract_trajectory_specific_point_cloud(
                                            point_cloud_raw.numpy(),
                                            full_traj.numpy(),
                                            radius=args.use_trajectory_pointcloud
                                        )
                                        if point_cloud_raw is not None:
                                            point_cloud_raw = torch.from_numpy(point_cloud_raw)

                                    if point_cloud_raw is not None and len(point_cloud_raw) > 0:
                                        pc_vis = transform_coords_for_visualization(point_cloud_raw, coord_mapping)

                                # Get orientation data (6D rotation)
                                gt_hist_rot = gt_full[:hist_len, 3:].cpu() if gt_full.shape[1] > 3 else None
                                gt_future_rot = gt_full[hist_len:actual_len, 3:].cpu() if gt_full.shape[1] > 3 else None
                                pred_future_rot = pred_full[hist_len:actual_len, 3:].cpu() if pred_full.shape[1] > 3 else None

                                # Get semantic bbox info
                                sem_bbox_info = batch.get('scene_bbox_info', [None])[i] if 'scene_bbox_info' in batch else None
                                sem_bbox_mask = batch.get('scene_bbox_mask', [None])[i] if 'scene_bbox_mask' in batch else None
                                sem_bbox_cats = batch.get('scene_bbox_categories', None)

                                sequence_name = batch.get('sequence_name', ['unknown'])[i] if 'sequence_name' in batch else 'unknown'

                                # Log debug info
                                logger.info(f"Rerun visualization for {filename_base}:")
                                logger.info(f"  - Past trajectory: {hist_len} points")
                                logger.info(f"  - Future GT: {actual_len - hist_len} points")
                                logger.info(f"  - Point cloud: {pc_vis.shape[0] if pc_vis is not None else 0} points")

                                # Visualize trajectory with rerun
                                visualize_trajectory_rerun(
                                    past_positions=gt_pos[:hist_len],
                                    future_positions_gt=gt_pos[hist_len:],
                                    future_positions_pred=pred_pos[hist_len:],
                                    past_mask=mask[:hist_len].cpu(),
                                    future_mask_gt=mask[hist_len:actual_len].cpu(),
                                    point_cloud=pc_vis,
                                    past_orientations=gt_hist_rot,
                                    future_orientations_gt=gt_future_rot,
                                    future_orientations_pred=pred_future_rot,
                                    semantic_bbox_info=sem_bbox_info.cpu() if sem_bbox_info is not None else None,
                                    semantic_bbox_mask=sem_bbox_mask.cpu() if sem_bbox_mask is not None else None,
                                    semantic_bbox_categories=sem_bbox_cats,
                                    object_name=obj_name,
                                    sequence_name=sequence_name,
                                    segment_idx=seg_idx,
                                    arrow_length=args.rerun_arrow_length,
                                    line_width=args.rerun_line_width,
                                    point_size=args.rerun_point_size,
                                    show_arrows=args.rerun_show_arrows,
                                    coord_mapping=coord_mapping
                                )

                                # Save the .rrd recording
                                if rrd_path:
                                    save_rerun_recording(output_path=rrd_path)
                                    # Verify file was created
                                    if os.path.exists(rrd_path):
                                        file_size = os.path.getsize(rrd_path)
                                        logger.info(f"  - Saved .rrd file: {file_size} bytes")

                                # Disconnect rerun session to avoid data leakage between samples
                                if hasattr(rr, 'disconnect') and callable(rr.disconnect):
                                    try:
                                        rr.disconnect()
                                    except Exception:
                                        pass

                        except Exception as rerun_e:
                            logger.warning(f"Rerun visualization error for {obj_name}: {rerun_e}")
                            import traceback
                            logger.warning(traceback.format_exc())

                    visualized_count += 1

            evaluated_samples += batch_size

    # Compute and log results
    if total_samples > 0:
        # Calculate collision rate only for trajectories where GT has no collision
        collision_rate = total_collision / total_valid_for_collision_eval if total_valid_for_collision_eval > 0 else 0.0

        results = {
            'l1_mean': total_l1 / total_samples,
            'rmse_ade': total_rmse / total_samples,
            'fde': total_fde / total_samples,
            'frechet': total_frechet / total_samples,
            'angular_cosine': total_angular / total_samples,
            'collision_rate': collision_rate,
            'num_samples': total_samples,
            'num_valid_for_collision': total_valid_for_collision_eval,
            'num_gt_collisions': total_gt_collisions
        }

        logger.info("\n=== Evaluation Results ===")
        logger.info(f"Samples: {total_samples}")
        logger.info(f"L1 Mean: {results['l1_mean']:.4f}")
        logger.info(f"RMSE/ADE: {results['rmse_ade']:.4f}")
        logger.info(f"FDE: {results['fde']:.4f}")
        logger.info(f"Frechet: {results['frechet']:.4f}")
        logger.info(f"Angular Cosine: {results['angular_cosine']:.4f}")
        logger.info(f"Collision Rate: {results['collision_rate']:.4f} (evaluated on {total_valid_for_collision_eval} samples, {total_gt_collisions} GT collisions excluded)")

        # Save results
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(args.output_dir, 'per_sample_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)

        return results

    logger.warning("No valid samples evaluated")
    return None


def main():
    args = parse_args()

    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.basename(os.path.dirname(os.path.normpath(args.model_path)))
    args.output_dir = f"{args.output_dir}/{timestamp}_{model_dir}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    logger = setup_logger(args.output_dir)
    logger.info(f"Evaluating: {args.model_path}")
    logger.info(f"Output: {args.output_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    try:
        model, config, best_epoch = load_model_and_config(args.model_path, device, logger)

        if args.use_first_frame_only:
            config.use_first_frame_only = True
            logger.info("Override: use_first_frame_only = True")

        # Default to detect_motion_segments=True unless explicitly disabled
        if args.no_detect_motion_segments:
            config.detect_motion_segments = False
            logger.info("Override: detect_motion_segments = False")
        else:
            config.detect_motion_segments = True
            if not getattr(config, 'detect_motion_segments', True):
                logger.info("Override: detect_motion_segments = True (default for eval)")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {total_params:,} parameters")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Run evaluation
    evaluate(model, config, args, logger)
    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
