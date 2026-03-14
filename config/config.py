from argparse import ArgumentParser


class Config(ArgumentParser):
    def __init__(self):
        super().__init__()

        # === Input Configuration ===
        self.add_argument('--batch_size', default=1, type=int)
        self.add_argument('--num_workers', default=0, type=int)

        # === Dataset Configuration ===
        self.add_argument('--dataset_type', default='adt', type=str, choices=['adt'])

        # ADT dataset
        self.add_argument('--adt_dataroot', default='./data', type=str)
        self.add_argument('--train_split_file', default=None, type=str)
        self.add_argument('--val_split_file', default=None, type=str)
        self.add_argument('--train_ratio', default=0.9, type=float)
        self.add_argument('--split_seed', default=42, type=int)

        # Trajectory
        self.add_argument('--trajectory_length', default=200, type=int)
        self.add_argument('--history_fraction', default=0.3, type=float)
        self.add_argument('--object_position_dim', default=3, type=int)
        self.add_argument('--object_rotation_dim', default=6, type=int)
        self.add_argument('--object_motion_dim', default=9, type=int)
        self.add_argument('--skip_frames', default=5, type=int)
        self.add_argument('--use_first_frame_only', action='store_true', default=False)
        self.add_argument('--no_load_pointcloud', dest='load_pointcloud', action='store_false')
        self.set_defaults(load_pointcloud=True)
        self.add_argument('--pointcloud_subsample', default=1, type=int)
        self.add_argument('--trajectory_pointcloud_radius', default=1.0, type=float)
        self.add_argument('--min_motion_threshold', default=0.5, type=float)
        self.add_argument('--min_motion_percentile', default=0.0, type=float)
        self.add_argument('--no_use_cache', dest='use_cache', action='store_false')
        self.set_defaults(use_cache=True)
        self.add_argument('--no_detect_motion_segments', dest='detect_motion_segments', action='store_false')
        self.set_defaults(detect_motion_segments=True)
        self.add_argument('--motion_velocity_threshold', default=0.05, type=float)
        self.add_argument('--min_segment_frames', default=5, type=int)
        self.add_argument('--max_stationary_frames', default=3, type=int)
        self.add_argument('--normalize_data', action='store_true', default=False)
        self.add_argument('--global_cache_dir', type=str, default=None)
        self.add_argument('--force_use_cache', action='store_true', default=False)

        # CLIP caching
        self.add_argument('--no_clip_cache', dest='use_clip_cache', action='store_false')
        self.set_defaults(use_clip_cache=True)
        self.add_argument('--clip_cache_dir', type=str, default=None)

        # === Scene Configuration ===
        self.add_argument('--scene_feats_dim', default=256, type=int)
        self.add_argument('--sample_points', default=50000, type=int)
        self.add_argument('--no_bbox', action='store_true', default=False)
        self.add_argument('--no_scene', action='store_true', default=False)
        self.add_argument('--no_semantic_bbox', action='store_true', default=False)
        self.add_argument('--no_semantic_text', action='store_true', default=False)
        self.add_argument('--max_bboxes', default=400, type=int)
        self.add_argument('--semantic_bbox_embed_dim', default=256, type=int)
        self.add_argument('--semantic_text_embed_dim', default=256, type=int)
        self.add_argument('--semantic_bbox_hidden_dim', default=128, type=int)
        self.add_argument('--semantic_bbox_num_heads', default=4, type=int)
        self.add_argument('--no_semantic_bbox_attention', dest='semantic_bbox_use_attention', action='store_false')
        self.set_defaults(semantic_bbox_use_attention=True)

        # === Motion Transformer ===
        self.add_argument('--motion_hidden_dim', default=256, type=int)
        self.add_argument('--motion_latent_dim', default=256, type=int)
        self.add_argument('--motion_n_heads', default=8, type=int)
        self.add_argument('--motion_n_layers', default=3, type=int)
        self.add_argument('--dropout', default=0.0, type=float)

        # === Output Pathway ===
        self.add_argument('--output_latent_dim', default=256, type=int)
        self.add_argument('--output_n_heads', default=8, type=int)
        self.add_argument('--output_n_layers', default=3, type=int)

        # === Text/Category Embedding ===
        self.add_argument('--no_text_embedding', action='store_true', default=False)
        self.add_argument('--category_embed_dim', default=256, type=int)
        self.add_argument('--clip_model_name', default="ViT-B/32", type=str, choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"])

        # === End Pose ===
        self.add_argument('--no_end_pose', action='store_true', default=False)
        self.add_argument('--end_pose_embed_dim', default=256, type=int)

        # === Visualization ===
        self.add_argument('--use_rerun', action='store_true', default=True, help='Enable Rerun visualization and save .rrd files')

        # === Training ===
        self.add_argument('--exp_name', type=str, default='adt', help='Experiment name for checkpoint folder')
        self.add_argument('--save_path', type=str, default='checkpoints/')
        self.add_argument('--save_fre', type=int, default=10)
        self.add_argument('--val_fre', type=int, default=1)
        self.add_argument('--num_val_visualizations', type=int, default=1)
        self.add_argument('--load_model_dir', type=str, default=None)
        self.add_argument('--epoch', type=int, default=500)
        self.add_argument('--lr', type=float, default=1e-4)
        self.add_argument('--weight_decay', type=float, default=5e-4)
        self.add_argument('--gamma', type=float, default=0.99)
        self.add_argument('--adam_beta1', type=float, default=0.9)
        self.add_argument('--adam_beta2', type=float, default=0.999)
        self.add_argument('--adam_eps', type=float, default=1e-8)
        self.add_argument('--lambda_trans', type=float, default=1.0)
        self.add_argument('--lambda_ori', type=float, default=1.0)
        self.add_argument('--lambda_rec', type=float, default=1.0)

        # === WandB ===
        self.add_argument('--wandb_project', type=str, default="GMT")
        self.add_argument('--wandb_entity', type=str, default=None)
        self.add_argument('--wandb_mode', type=str, default="online", choices=["online", "offline", "disabled"])

    def get_configs(self):
        args = self.parse_args()

        # Validate object_motion_dim
        if args.object_motion_dim != args.object_position_dim + args.object_rotation_dim:
            args.object_motion_dim = args.object_position_dim + args.object_rotation_dim

        return args


if __name__ == '__main__':
    config = Config()
    print(config.get_configs())
