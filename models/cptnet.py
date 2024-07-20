from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Any, List
from torch_geometric.nn import fps

from models.diffusion_nets.coarse_stage.model.cond_pvcnn_generation import Cond_PVCNN2 as PVCNN


class ColorEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(ColorEncoder, self).__init__()
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_img = nn.Linear(256 * 32 * 32, embedding_dim)  # assuming input image size is 512x512

    def forward(self, uv_imgs, coord_imgs, mask_imgs):
        x = torch.cat((uv_imgs, coord_imgs, mask_imgs), dim=3).float().cuda()
        x = x.reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        embedding = F.relu(self.fc_img(x))
        return embedding


class PointEncoder(nn.Module):
    def __init__(self, pos_emb_dim=16):
        super(PointEncoder, self).__init__()
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, pos_emb_dim, 1)
        )

    def forward(self, x):
        out = self.point_encoder(x)
        return out


class PointDiffusionNet(nn.Module):
    def __init__(self, pos_emb_dim, color_emb_dim, num_points):
        super(PointDiffusionNet, self).__init__()
        self.num_points = num_points
        
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, pos_emb_dim, 1)
        )
        
        # self.color_encoder = nn.Sequential(
        #     nn.Conv1d(embedding_dim, 128, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, num_points, 1)
        # )
        
        self.diffusion = nn.Sequential(
            nn.Conv1d(pos_emb_dim+color_emb_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder for color prediction
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1),
            # nn.Sigmoid()
        )

        
    def forward(self, positions, color_embeddings):
        positions = positions.transpose(1, 2).float()  # [batch, 3, num_points]
        color_embeddings = color_embeddings.transpose(1, 2).float()  # [batch, color_emb_dim, num_points]
        
        encoded_positions = self.point_encoder(positions)  # [batch, pos_emb_dim, num_points]
        encoded_colors = color_embeddings   # [batch, color_emb_dim, num_points]
        combined_features = torch.cat((encoded_positions, encoded_colors), dim=1)  # [batch, pos_emb_dim+color_emb_dim, num_points]

        diffused_features = self.diffusion(combined_features)  # [batch, 128, num_points]
        
        predicted_colors = self.decoder(diffused_features)  # [batch, 3, num_points]
        predicted_colors = predicted_colors.transpose(1, 2)  # [batch, num_points, 3]

        return predicted_colors



class PointUNet(PVCNN):
    def __init__(self, num_classes=6, embed_dim=64, use_att=True, dropout=0.2, extra_feature_channels=9, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim, use_att=use_att, dropout=dropout, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


    def forward(self, point_embs, color_embs):
        cond = torch.cat([point_embs, color_embs], dim=1)
        print("Cond", cond.shape)

        coords, features = cond[:, :3, :].contiguous(), torch.cat([cond, cond], dim=1)
        coords_list, in_features_list = [], []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
            print()
        
        # if self.global_att is not None:
        #     features = self.global_att(features)
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1 - fp_idx], coords, features, in_features_list[-1 - fp_idx]))
        out = self.classifier(features)



class CPTNet(nn.Module):
    def __init__(self, args):
        super(CPTNet, self).__init__()
        self.sample_num = args.num_out_points
        self.color_emb_dim = args.num_emb_dim
        self.pos_emb_dim = 16

        self.color_encoder = ColorEncoder(self.color_emb_dim)
        self.point_encoder = PointEncoder(pos_emb_dim=self.pos_emb_dim)
        # self.diffusion = PointDiffusionNet(200, self.color_emb_dim, self.sample_num)
        self.diffusion = PointUNet(num_classes=6, embed_dim=64, use_att=True, dropout=0.2, extra_feature_channels=9, width_multiplier=1, voxel_resolution_multiplier=1)


    def forward(self, data):
        # uv_imgs, coord_imgs, masks, sample_points, sample_colors = data
        # batch_size = uv_imgs.shape[0]

        # # print("======ENCODER======")
        # color_embeddings = self.encoder(uv_imgs, coord_imgs, masks)  # [batch, emb_dim]
        # color_embeddings = torch.cat([color_embeddings.reshape(batch_size, self.color_emb_dim, 1)] * self.sample_num, dim=2)    # [batch, emb_dim x sample_num]

        # sample_points = sample_points.cuda()    # [batch, sample_num, 3]
        # color_embeddings = color_embeddings.reshape(batch_size, self.sample_num, -1)      # [batch, sample_num, emb_dim]

        # point_embeddings = self.point_encoder(sample_points)    # [batch, sample_num, 3]
        # # print("======DIFF======")
        # # pred_colors = self.diffusion(sample_points, color_embedding)
        # # return pred_colors

        # pred_colors = self.diffusion(point_embeddings, color_embeddings)

        # return pred_colors
        return data
    
