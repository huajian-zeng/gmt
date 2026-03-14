#!/usr/bin/env python3
import torch
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
import wandb
from functools import partial

from config.config import Config
from dataset.adt_dataset import ADTDataset
from model.trajectory_model import TrajectoryModel
from torch.utils.data import DataLoader
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory
from utils.metrics_utils import (
    transform_coords_for_visualization,
    compute_metrics_for_sample,
    collate_fn
)

try:
    from utils.adt_sequence_utils import find_adt_sequences, create_train_test_split
    HAS_SEQ_UTILS = True
except ImportError:
    print("Warning: Could not import adt_sequence_utils.")
    HAS_SEQ_UTILS = False

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def get_coord_mapping_for_dataset(config):
    """Get coordinate mapping based on dataset type."""
    return ['x', '-z', 'y']


def log_metrics(epoch, title, metrics, logger_func):
    log_str = f"Epoch {epoch} {title}: Total Loss {metrics['total_loss']:.4f}"
    log_str += " | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'total_loss'])
    logger_func(log_str)


def prepare_batch(batch, config, device):
    """Prepare batch data for model forward pass."""
    full_trajectory_batch = batch['full_poses'].float().to(device)
    point_cloud_batch = batch['point_cloud'].float().to(device)

    bbox_corners_batch = None
    if not config.no_bbox:
        bbox_corners_batch = batch['bbox_corners'].float().to(device)

    batch['full_attention_mask'] = batch['full_attention_mask'].to(device)
    current_attention_mask_batch = batch['full_attention_mask']
    actual_lengths_batch = current_attention_mask_batch.sum(dim=1).int()

    # Determine input history length
    if config.use_first_frame_only:
        hist_len = 1 if torch.any(actual_lengths_batch > 0) else 0
    else:
        individual_hist_lengths = (actual_lengths_batch.float() * config.history_fraction).floor().long()
        individual_hist_lengths = torch.max(torch.ones_like(individual_hist_lengths), individual_hist_lengths)
        individual_hist_lengths = torch.min(individual_hist_lengths, actual_lengths_batch)
        individual_hist_lengths[actual_lengths_batch == 0] = 0
        hist_len = min(torch.max(individual_hist_lengths).item(), full_trajectory_batch.shape[1])

    input_trajectory_batch = full_trajectory_batch[:, :hist_len, :]
    bbox_corners_input_batch = bbox_corners_batch[:, :hist_len, :, :] if bbox_corners_batch is not None else None

    # Object category
    object_category_ids = None
    if not config.no_text_embedding:
        if 'object_category_clip' in batch:
            object_category_ids = batch['object_category_clip'].to(device)
        else:
            object_category_ids = batch['object_category']

    # Semantic bbox
    semantic_bbox_info = None
    semantic_bbox_mask = None
    if not config.no_semantic_bbox:
        semantic_bbox_info = batch['scene_bbox_info'].float().to(device)
        semantic_bbox_mask = batch['scene_bbox_mask'].float().to(device)

    # Semantic text
    semantic_text_categories = None
    if not getattr(config, 'no_semantic_text', False):
        if 'scene_bbox_categories_clip' in batch:
            semantic_text_categories = batch['scene_bbox_categories_clip'].to(device)
        else:
            semantic_text_categories = batch.get('scene_bbox_categories', None)

    # End pose
    end_pose_batch = None
    if not getattr(config, 'no_end_pose', False):
        end_pose_batch = []
        for i in range(full_trajectory_batch.shape[0]):
            actual_length = torch.sum(current_attention_mask_batch[i]).int().item()
            if actual_length > 0:
                end_pose = full_trajectory_batch[i, actual_length-1:actual_length, :]
                end_pose_batch.append(end_pose)
            else:
                end_pose_batch.append(torch.zeros(1, full_trajectory_batch.shape[2], device=device))
        end_pose_batch = torch.cat(end_pose_batch, dim=0)

    return {
        'input_trajectory': input_trajectory_batch,
        'point_cloud': point_cloud_batch,
        'bbox_corners': bbox_corners_input_batch,
        'object_category': object_category_ids,
        'semantic_bbox_info': semantic_bbox_info,
        'semantic_bbox_mask': semantic_bbox_mask,
        'semantic_text': semantic_text_categories,
        'end_pose': end_pose_batch,
        'full_trajectory': full_trajectory_batch,
        'bbox_corners_full': bbox_corners_batch,
    }


def validate(model, dataloader, device, config, epoch):
    """Run validation and return metrics."""
    model.eval()
    val_total_loss = 0.0
    val_loss_components = {}
    coord_mapping = get_coord_mapping_for_dataset(config)

    # Metrics tracking
    total_l1, total_rmse, total_fde, total_frechet, total_angular_cosine = 0.0, 0.0, 0.0, 0.0, 0.0
    total_valid_samples = 0

    # Visualization setup - only visualize every 10 epochs
    visualized_count = 0
    vis_limit = config.num_val_visualizations
    is_first_validation = epoch == config.val_fre
    should_visualize = (epoch % 10 == 0) or is_first_validation

    vis_output_dir = os.path.join(config.save_path, "val_visualizations", f"epoch_{epoch}")
    if vis_limit > 0 and should_visualize:
        os.makedirs(vis_output_dir, exist_ok=True)

    if is_first_validation:
        trajectory_vis_dir = os.path.join(config.save_path, "trajectory_visualizations")
        os.makedirs(trajectory_vis_dir, exist_ok=True)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                prepared = prepare_batch(batch, config, device)
            except (KeyError, Exception) as e:
                print(f"Error processing batch {batch_idx}: {e}. Skipping.")
                continue

            predicted = model(
                prepared['input_trajectory'], prepared['point_cloud'],
                prepared['bbox_corners'], prepared['object_category'],
                prepared['semantic_bbox_info'], prepared['semantic_bbox_mask'],
                prepared['semantic_text'], prepared['end_pose']
            )

            total_loss, loss_dict = model.compute_loss(predicted, batch)
            val_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    val_loss_components[key] = val_loss_components.get(key, 0.0) + value.item()

            progress_bar.set_postfix({'val_loss': f"{total_loss.item():.4f}"})

            # Compute metrics
            batch_size = prepared['full_trajectory'].shape[0]
            for i in range(batch_size):
                gt_full_poses = batch['full_poses'][i]
                gt_full_mask = batch['full_attention_mask'][i]
                pred_full = predicted[i]

                actual_length = torch.sum(gt_full_mask).int().item()
                if actual_length < 2:
                    continue

                if config.use_first_frame_only:
                    history_length = 1
                else:
                    history_length = max(1, min(int(actual_length * config.history_fraction), actual_length - 1))

                position_dim = config.object_position_dim
                gt_future = gt_full_poses[history_length:actual_length, :position_dim]
                pred_future = pred_full[history_length:actual_length, :position_dim]
                future_mask = gt_full_mask[history_length:actual_length]

                if future_mask.sum() == 0:
                    continue

                l1, rmse, fde, frechet, angular = compute_metrics_for_sample(pred_future, gt_future, future_mask)
                total_l1 += l1.item()
                total_rmse += rmse.item()
                total_fde += fde.item()
                total_frechet += frechet.item()
                total_angular_cosine += angular.item()
                total_valid_samples += 1

            # Visualization (only every 10 epochs)
            if should_visualize and visualized_count < vis_limit:
                for i in range(min(batch_size, vis_limit - visualized_count)):
                    if visualized_count >= vis_limit:
                        break

                    gt_full_poses = batch['full_poses'][i]
                    gt_full_mask = batch['full_attention_mask'][i]
                    pred_full = predicted[i]
                    sample_pc = prepared['point_cloud'][i].cpu()

                    actual_length = torch.sum(gt_full_mask).int().item() if gt_full_mask is not None else gt_full_poses.shape[0]
                    if config.use_first_frame_only:
                        hist_len = 1 if actual_length >= 1 else 0
                    else:
                        hist_len = max(1, min(int(actual_length * config.history_fraction), actual_length))

                    position_dim = 3
                    gt_positions = gt_full_poses[:, :position_dim]
                    pred_positions = pred_full[:, :position_dim]
                    gt_rotations = gt_full_poses[:, position_dim:]
                    pred_rotations = pred_full[:, position_dim:]

                    # Build filename
                    obj_name = batch['object_name'][i] if 'object_name' in batch else f"unknown_{i}"
                    segment_idx = batch['segment_idx'][i].item() if 'segment_idx' in batch and batch['segment_idx'][i].item() != -1 else None
                    seq_path = batch['sequence_path'][i]
                    seq_name = os.path.splitext(os.path.basename(seq_path))[0]
                    filename_base = f"{obj_name}_seq_{seq_name}_seg{segment_idx if segment_idx else 'NA'}_batch{batch_idx}"
                    vis_title = f"{obj_name} (Seq: {seq_name}, Seg: {segment_idx if segment_idx else 'NA'})"

                    if is_first_validation:
                        # Full trajectory visualization
                        gt_pos_vis = transform_coords_for_visualization(gt_positions.cpu(), coord_mapping)
                        pc_vis = transform_coords_for_visualization(sample_pc, coord_mapping)
                        bbox_vis = None
                        if not config.no_bbox and 'bbox_corners' in batch:
                            bbox_vis = transform_coords_for_visualization(prepared['bbox_corners_full'][i].cpu(), coord_mapping)

                        traj_bbox_info = batch['scene_bbox_info'][i].cpu() if 'scene_bbox_info' in batch else None
                        traj_bbox_mask = batch['scene_bbox_mask'][i].cpu() if 'scene_bbox_mask' in batch else None

                        visualize_full_trajectory(
                            positions=gt_pos_vis,
                            attention_mask=gt_full_mask.cpu() if gt_full_mask is not None else None,
                            point_cloud=pc_vis,
                            bbox_corners_sequence=bbox_vis,
                            trajectory_specific_bbox_info=traj_bbox_info,
                            trajectory_specific_bbox_mask=traj_bbox_mask,
                            title=f"Full GT - {vis_title}",
                            save_path=os.path.join(trajectory_vis_dir, f"{filename_base}_full_trajectory.png"),
                            segment_idx=segment_idx,
                            coord_mapping=coord_mapping
                        )

                        # Split visualization
                        past_pos_vis = transform_coords_for_visualization(gt_positions[:hist_len].cpu(), coord_mapping)
                        future_pos_vis = transform_coords_for_visualization(gt_positions[hist_len:actual_length].cpu(), coord_mapping)
                        visualize_trajectory(
                            past_positions=past_pos_vis,
                            future_positions=future_pos_vis,
                            past_mask=gt_full_mask[:hist_len].cpu() if gt_full_mask is not None else None,
                            future_mask=gt_full_mask[hist_len:actual_length].cpu() if gt_full_mask is not None else None,
                            title=f"Split GT - {vis_title}",
                            save_path=os.path.join(trajectory_vis_dir, f"{filename_base}_split.png"),
                            segment_idx=segment_idx
                        )

                    # Prediction visualization
                    past_pos_vis = transform_coords_for_visualization(gt_positions[:hist_len].cpu(), coord_mapping)
                    future_gt_vis = transform_coords_for_visualization(gt_positions[hist_len:actual_length].cpu(), coord_mapping)
                    future_pred_vis = transform_coords_for_visualization(pred_positions[hist_len:actual_length].cpu(), coord_mapping)

                    visualize_prediction(
                        past_positions=past_pos_vis,
                        future_positions_gt=future_gt_vis,
                        future_positions_pred=future_pred_vis,
                        past_mask=gt_full_mask[:hist_len].cpu() if gt_full_mask is not None else None,
                        future_mask_gt=gt_full_mask[hist_len:actual_length].cpu() if gt_full_mask is not None else None,
                        title=f"Pred vs GT - {vis_title} (Epoch {epoch})",
                        save_path=os.path.join(vis_output_dir, f"{filename_base}_pred_epoch{epoch}.png"),
                        segment_idx=segment_idx,
                        show_orientation=getattr(config, 'show_ori_arrows', False),
                        past_orientations=gt_rotations[:hist_len].cpu(),
                        future_orientations_gt=gt_rotations[hist_len:actual_length].cpu(),
                        future_orientations_pred=pred_rotations[hist_len:actual_length].cpu()
                    )
                    visualized_count += 1

    avg_val_loss = val_total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in val_loss_components.items()}
    avg_components['total_loss'] = avg_val_loss

    if total_valid_samples > 0:
        avg_components['l1_mean'] = total_l1 / total_valid_samples
        avg_components['rmse'] = total_rmse / total_valid_samples
        avg_components['fde'] = total_fde / total_valid_samples
        avg_components['frechet'] = total_frechet / total_valid_samples
        avg_components['angular_cosine'] = total_angular_cosine / total_valid_samples
        print(f"Validation Metrics - L1: {avg_components['l1_mean']:.4f}, RMSE: {avg_components['rmse']:.4f}, "
              f"FDE: {avg_components['fde']:.4f}, Frechet: {avg_components['frechet']:.4f}")

    return avg_components


def main():
    print("Loading configuration...")
    config = Config().get_configs()

    # Add timestamp and exp_name to save_path
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_path_base = config.save_path.rstrip('/')
    config.save_path = os.path.join(save_path_base, f"{timestamp}_{config.exp_name}")

    print(f"Save path: {config.save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logging setup
    os.makedirs(config.save_path, exist_ok=True)
    log_file = os.path.join(config.save_path, 'train_log.txt')
    def logger(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")

    logger(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger(f"Config: {vars(config)}\n")

    # WandB initialization
    if config.wandb_mode != 'disabled':
        try:
            run_name = f"GMT_{timestamp}_{os.path.basename(config.save_path)}"
            wandb.init(
                project=config.wandb_project,
                config=vars(config),
                name=run_name,
                entity=config.wandb_entity,
                mode=config.wandb_mode
            )
            logger(f"WandB initialized. Run name: {run_name}")
        except Exception as e:
            logger(f"Warning: Could not initialize WandB: {e}")
            config.wandb_mode = 'disabled'

    # Dataset setup
    logger("Setting up dataset...")
    train_sequences, val_sequences = [], []
    loaded_from_files = False

    # Load from split files if provided
    if config.train_split_file and config.val_split_file:
        if os.path.exists(config.train_split_file) and os.path.exists(config.val_split_file):
            with open(config.train_split_file, 'r') as f:
                train_sequences = [line.strip() for line in f if line.strip()]
            with open(config.val_split_file, 'r') as f:
                val_sequences = [line.strip() for line in f if line.strip()]
            if train_sequences and val_sequences:
                logger(f"Loaded {len(train_sequences)} train and {len(val_sequences)} val sequences from files.")
                loaded_from_files = True

    # Dynamic splitting fallback
    if not loaded_from_files:
        if not os.path.exists(config.adt_dataroot):
            logger(f"Error: adt_dataroot {config.adt_dataroot} does not exist.")
            return

        if os.path.isdir(config.adt_dataroot) and HAS_SEQ_UTILS:
            all_sequences = find_adt_sequences(config.adt_dataroot)
            if not all_sequences:
                logger(f"Error: No sequences found in {config.adt_dataroot}.")
                return
            train_sequences, val_sequences = create_train_test_split(
                all_sequences, train_ratio=config.train_ratio, random_seed=config.split_seed
            )
            if config.train_ratio >= 1.0:
                val_sequences = train_sequences
            logger(f"Split: {len(train_sequences)} train, {len(val_sequences)} val sequences.")
        elif os.path.isdir(config.adt_dataroot):
            train_sequences = [os.path.join(config.adt_dataroot, d) for d in os.listdir(config.adt_dataroot)
                             if os.path.isdir(os.path.join(config.adt_dataroot, d))]
            val_sequences = train_sequences
        else:
            train_sequences = [config.adt_dataroot]
            val_sequences = [config.adt_dataroot]

    if not train_sequences or not val_sequences:
        logger("Error: No sequences found.")
        return

    # Save sequence lists
    with open(os.path.join(config.save_path, 'train_sequences.txt'), 'w') as f:
        f.write('\n'.join(train_sequences))
    with open(os.path.join(config.save_path, 'val_sequences.txt'), 'w') as f:
        f.write('\n'.join(val_sequences))

    # Cache directory
    cache_dir = config.global_cache_dir if config.global_cache_dir else os.path.join(config.save_path, 'trajectory_cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger(f"Cache directory: {cache_dir}")

    # Create datasets
    logger("Using ADT dataset")
    train_dataset = ADTDataset(sequence_paths=train_sequences, config=config, cache_dir=cache_dir)
    val_dataset = ADTDataset(sequence_paths=val_sequences, config=config, cache_dir=cache_dir)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger("Error: Empty dataset.")
        return

    # DataLoaders
    train_collate = partial(collate_fn, dataset=train_dataset, num_sample_points=config.sample_points)
    val_collate = partial(collate_fn, dataset=val_dataset, num_sample_points=config.sample_points)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                             num_workers=config.num_workers, drop_last=True, collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=config.num_workers, drop_last=True, collate_fn=val_collate)
    logger(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Model
    logger("Initializing model...")
    model = TrajectoryModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    if config.wandb_mode != 'disabled':
        wandb.watch(model)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                           betas=(config.adam_beta1, config.adam_beta2),
                           eps=config.adam_eps, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    # Load checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    if config.load_model_dir and os.path.exists(config.load_model_dir):
        logger(f"Loading checkpoint: {config.load_model_dir}")
        checkpoint = torch.load(config.load_model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        logger(f"Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    # Training loop
    logger("\n--- Starting Training ---")
    for epoch in range(start_epoch, config.epoch + 1):
        logger(f"\nEpoch {epoch}/{config.epoch}")
        model.train()
        epoch_loss = 0.0
        epoch_components = {}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                prepared = prepare_batch(batch, config, device)
            except (KeyError, Exception) as e:
                print(f"Error in batch {batch_idx}: {e}. Skipping.")
                continue

            predicted = model(
                prepared['input_trajectory'], prepared['point_cloud'],
                prepared['bbox_corners'], prepared['object_category'],
                prepared['semantic_bbox_info'], prepared['semantic_bbox_mask'],
                prepared['semantic_text'], prepared['end_pose']
            )

            total_loss, loss_dict = model.compute_loss(predicted, batch)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    epoch_components[k] = epoch_components.get(k, 0.0) + v.item()

            progress_bar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        # Log training metrics
        avg_train = {k: v / len(train_loader) for k, v in epoch_components.items()}
        avg_train['total_loss'] = epoch_loss / len(train_loader)
        log_metrics(epoch, "Training", avg_train, logger)

        if config.wandb_mode != 'disabled':
            wandb.log({"train/" + k: v for k, v in avg_train.items()}, step=epoch)
            wandb.log({"learning_rate": scheduler.get_last_lr()[0]}, step=epoch)

        # Validation
        if epoch % config.val_fre == 0:
            val_metrics = validate(model, val_loader, device, config, epoch)
            log_metrics(epoch, "Validation", val_metrics, logger)

            if config.wandb_mode != 'disabled':
                wandb.log({"val/" + k: v for k, v in val_metrics.items()}, step=epoch)

            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                logger(f"New best val loss: {best_val_loss:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': vars(config)
                }, os.path.join(config.save_path, 'best_model.pth'))

        # Periodic checkpoint
        if epoch % config.save_fre == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(config)
            }, os.path.join(config.save_path, f'ckpt_epoch_{epoch}.pth'))

        scheduler.step()

    # Save final model
    logger("\n--- Training Finished ---")
    torch.save({
        'epoch': config.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': vars(config)
    }, os.path.join(config.save_path, 'final_model.pth'))

    if config.wandb_mode != 'disabled':
        wandb.finish()


if __name__ == '__main__':
    main()
