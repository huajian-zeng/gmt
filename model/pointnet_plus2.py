"""PointNet2 implementation borrowed from https://github.com/erikwijmans/Pointnet2_PyTorch"""

import torch
import torch.nn as nn

from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class MyFPModule(nn.Module):
    """Feature propagation module using 3-NN interpolation."""

    def __init__(self):
        super().__init__()

    def forward(self, unknown, known, known_feats):
        """
        Args:
            unknown: (B, N, 3) target points
            known: (B, M, 3) source points
            known_feats: (B, F, M) source features
        Returns:
            (B, F, N) interpolated features
        """
        dist, idx = pointnet2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        return interpolated_feats


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.InstanceNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        """Forward pass. Input: (B, N, 3+input_channels), Output: (B, feat_dim, N)"""
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class PointNet2SemSegSSGShape(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 256],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.InstanceNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, self.hparams['feat_dim']),
            nn.InstanceNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        """Forward pass. Returns per-point features and global features."""
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        bottleneck_feats = l_features[-1].squeeze(-1)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0]), self.fc_layer2(bottleneck_feats)


class PointNet(nn.Module):
    """Simple PointNet for feature aggregation."""

    def __init__(self, feat_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(feat_dim, feat_dim, 1)
        self.conv2 = nn.Conv1d(feat_dim, feat_dim, 1)
        self.conv3 = nn.Conv1d(feat_dim, feat_dim, 1)
        self.bn1 = nn.InstanceNorm1d(feat_dim)
        self.bn2 = nn.InstanceNorm1d(feat_dim)
        self.bn3 = nn.InstanceNorm1d(feat_dim)

    def forward(self, x):
        """Input: (B, F, N), Output: (B, F)"""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1)[0]
        return x
