import torch.nn as nn
import torch
from .pvcnn_generation import *


class Cond_PVCNN2Base(PVCNN2Base):

    def __init__(self, num_classes, embed_dim, use_att, dropout,
                 extra_feature_channels, width_multiplier, voxel_resolution_multiplier):
        super().__init__(num_classes, embed_dim, use_att, dropout,
                 extra_feature_channels, width_multiplier, voxel_resolution_multiplier)
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # only use extra features in the last fp module
        # sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, dropout, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)


    def forward(self, inputs, cond=None):
        coords, features = cond[:, :3, :].contiguous(), torch.cat([inputs, cond], dim=1)
        coords_list, in_features_list = [], []
        for i, sa_blocks  in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks ((features, coords))

        in_features_list[0] = cond.contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks  in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, in_features_list[-1-fp_idx]))

        out = self.classifier(features)

        return out

class Cond_PVCNN2(Cond_PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att, dropout, extra_feature_channels, width_multiplier,
                 voxel_resolution_multiplier):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )