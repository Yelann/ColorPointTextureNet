import functools

import torch.nn as nn
from ..modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule, Attention
# from ..modules import PVConv

# def _linear_gn_relu(in_channels, out_channels):
#     return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8,out_channels), Swish())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        layers.append(nn.Linear(in_channels, out_channels[-1]))
    else:
        layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    c = 0
    for k, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(r * out_channels)
        for p in range(num_blocks):
            attention = k % 2 == 0 and k > 0 and p == 0
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                          with_se=with_se, normalize=normalize, eps=eps)

            if c == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels+embed_dim, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
            c += 1
    return layers, in_channels, concat_channels


<<<<<<< Updated upstream
def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
=======
def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, use_att=False,
>>>>>>> Stashed changes
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                              dropout=dropout,
                                              with_se=with_se, with_se_relu=True,
                                              normalize=normalize, eps=eps)

<<<<<<< Updated upstream
                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k ==0:
                    sa_blocks.append(block(in_channels+embed_dim, out_channels))
=======
                if c == 0 or k == 0:
                    sa_blocks.append(block(in_channels, out_channels))
>>>>>>> Stashed changes
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
<<<<<<< Updated upstream
=======
        # (1024, 0.1, 32, (32, 64))
>>>>>>> Stashed changes
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
<<<<<<< Updated upstream
        sa_blocks.append(block(in_channels=extra_feature_channels+(embed_dim if k==0 else 0 ), out_channels=out_channels,
=======
        sa_blocks.append(block(in_channels=extra_feature_channels, out_channels=out_channels,
>>>>>>> Stashed changes
                               include_coordinates=True))
        c += 1
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

<<<<<<< Updated upstream
    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
=======
    # print("sa in channels", sa_in_channels)
    # print("================")
    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, use_att=False,
>>>>>>> Stashed changes
                                dropout=0.1,
                                with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
<<<<<<< Updated upstream
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim, out_channels=out_channels)
=======
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx], out_channels=out_channels)
>>>>>>> Stashed changes
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                              dropout=dropout,
                                              with_se=with_se, with_se_relu=True,
                                              normalize=normalize, eps=eps)
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels



class PVCNN2Base(nn.Module):

    def __init__(self, num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
<<<<<<< Updated upstream
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
=======
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, 
>>>>>>> Stashed changes
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
<<<<<<< Updated upstream
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True, embed_dim=embed_dim,
=======
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True, 
>>>>>>> Stashed changes
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)


        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, dropout, num_classes], # was 0.5
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

<<<<<<< Updated upstream
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

=======
>>>>>>> Stashed changes

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for i, sa_blocks  in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks ((features, coords))
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks  in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, in_features_list[-1-fp_idx]))

        return self.classifier(features)


