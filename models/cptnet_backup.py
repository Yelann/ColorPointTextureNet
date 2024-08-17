from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Any, List
from torch_geometric.nn import fps
from utuils.point_sample_gather import pc_to_uv
from utuils.sample_utuils import get_all_colors

from models.diffusion_nets.coarse_stage.model.cond_pvcnn_generation import Cond_PVCNN2 as PVCNN

mse_loss = nn.MSELoss()

class ColorEncoder(nn.Module):
    def __init__(self, embedding_dim, pixel_num):
        super(ColorEncoder, self).__init__()
        self.PIXEL_NUM = pixel_num
        self.conv1 = nn.Conv2d(10, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if pixel_num == 512:
            self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.fc_img = nn.Linear(128 * 32 * 32, embedding_dim)  # assuming input image size is 512x512
        else:
            self.fc_img = nn.Linear(256 * 8 * 8, embedding_dim)  # assuming input image size is 128x128

    def forward(self, uv_imgs, coord_imgs, normal_imgs, mask_imgs):
        mask_imgs = mask_imgs[:, :, :, 0].reshape(uv_imgs.shape[0], uv_imgs.shape[1], uv_imgs.shape[2], 1)
        x = torch.cat((uv_imgs, coord_imgs, normal_imgs, mask_imgs), dim=3).float().cuda()
        # x = torch.cat((uv_imgs, coord_imgs, normal_imgs, mask_imgs), dim=3).float().cuda()
        x = x.reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))

        x = x.view(x.size(0), -1)  # flatten
        embedding = F.leaky_relu(self.fc_img(x))
        return embedding


class PointEncoder(nn.Module):
    def __init__(self, pos_emb_dim=16):
        super(PointEncoder, self).__init__()
        self.point_encoder = nn.Sequential(
            # nn.Conv1d(3, 16, 1),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(16, 32, 1),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(32, pos_emb_dim, 1),
            # nn.ReLU(inplace=True),
            nn.Conv1d(3, pos_emb_dim, 1),
        )

    def forward(self, x):
        out = self.point_encoder(x)
        return out



class PointUNet(PVCNN):
    def __init__(self, num_classes=6, use_att=True, dropout=0.2, extra_feature_channels=64, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(num_classes=num_classes, embed_dim=0, use_att=use_att, dropout=dropout, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


    def forward(self, fps_pos, feature_embs):
        features = torch.cat([fps_pos, feature_embs], dim=1)    # 32, 32, 256
        coords = fps_pos  # 32, 3, 256

        coords_list, in_features_list = [], []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        # if self.global_att is not None:
        #     features = self.global_att(features)
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1 - fp_idx], coords, features, in_features_list[-1 - fp_idx]))
        out = self.classifier(features)
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(9, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 3, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, pred_results, sample_points):
        x = torch.cat((pred_results, sample_points), dim=2).transpose(1, 2)
        # x = pred_results.transpose(1, 2)
        y = self.layers(x).transpose(1, 2)
        return y



class CPTNet(nn.Module):
    def __init__(self, args):
        super(CPTNet, self).__init__()
        self.PIXEL_NUM = args.pixel_num
        self.SAMPLE_NUM = args.num_out_points
        self.COLOR_EMB_DIM = args.num_emb_dim
        self.POS_EMB_DIM = args.num_pos_dim
        self.BATCH_SIZE = args.batch_size

        self.color_encoder = ColorEncoder(self.COLOR_EMB_DIM, self.PIXEL_NUM)
        self.point_encoder = PointEncoder(pos_emb_dim=self.POS_EMB_DIM)
        self.diffusion = PointUNet(num_classes=6, use_att=True, dropout=0.2, extra_feature_channels=self.COLOR_EMB_DIM+self.POS_EMB_DIM, width_multiplier=1, voxel_resolution_multiplier=1)
        self.mlp = MLP()


    def forward(self, data):
        uv_imgs, coord_imgs, normal_imgs, masks, sample_points, sample_colors, sample_uvs = data
        sample_points = sample_points.float().cuda()    # [batch, sample_num, 3]
        BATCH_SIZE = uv_imgs.shape[0]

        # ======COLOR ENCODER======
        color_embedding = self.color_encoder(uv_imgs, coord_imgs, normal_imgs, masks)       # [batch, emb_dim]
        color_embeddings = torch.cat([color_embedding.reshape(BATCH_SIZE, self.COLOR_EMB_DIM, 1)] * self.SAMPLE_NUM, dim=2)    # [batch, emb_dim x sample_num]
        color_embeddings = color_embeddings.reshape(BATCH_SIZE, -1, self.SAMPLE_NUM,)       # [batch, emb_dim, sample_num]

        # ======POINT ENCODER======
        if self.POS_EMB_DIM > 0:
            point_embeddings = self.point_encoder(sample_points.reshape(BATCH_SIZE, 3, -1))    # [batch, pos_emb, sample_num]
            feature_embeddings = torch.cat([point_embeddings, color_embeddings], dim=1)
        else:
            feature_embeddings = color_embeddings
        
        # ======DIFF======
        pred_result = self.diffusion(sample_points.reshape(BATCH_SIZE, 3, -1), feature_embeddings).transpose(1, 2)   # 0...1 [batch, sample_num, 6]

        # ======MPL======
        pred_features = self.mlp(pred_result, sample_points)    #[batch, sample_num, 3]

        # to uv
        feature_input = pred_features.float().cuda()    # [batch, sample_num, 3]
        point_input = sample_points.float().cuda() # [batch, sample_num, 3]

        # pred_features = (pred_features + 1) / 2.
        uv_11 = uv_imgs * 2. - 1
        uv_input = uv_11.float().cuda()

        # pred_imgs = pc_to_uv(feature_input, point_input, uv_input, masks, self.PIXEL_NUM, self.PIXEL_NUM)    # [batch, 512, 512, 3]
        pred_imgs = pc_to_uv(feature_input, point_input, uv_input)    # [batch, 512, 512, 3]
        uv_input2 = uv_imgs.float().cuda()
        pred_imgs2 = pc_to_uv(feature_input, point_input, uv_input2)    # [batch, 512, 512, 3]
        pred_imgs = (pred_imgs + pred_imgs2) / 2
        pred_imgs = torch.clamp(pred_imgs, min=0, max=1)
        # coord_11 = coord_imgs
        # coord_input = coord_11.float().cuda()
        # pred_imgs = pc_to_uv(feature_input, point_input, coord_input)    # [batch, 512, 512, 3]
        # pred_imgs = (pred_imgs + 1) / 2.
        # pred_imgs = torch.clamp(pred_imgs, min=0, max=1)

        pred_colors = get_all_colors(sample_uvs, pred_imgs)     # [batch, sample_num, 3]
        pred_imgs = pred_imgs * masks   # [0, 1]

        # loss 4 and loss 5
        enable_pixels   = torch.sum(masks, dim=(1,2)).cuda()
        ori_color_mean  = torch.sum(masks - uv_imgs, dim=(1,2)) / enable_pixels
        ori_color_var   = torch.sum((masks - uv_imgs)**2, dim=(1,2)) / enable_pixels
        pred_color_mean = torch.sum(masks - pred_imgs, dim=(1,2)) / enable_pixels
        pred_color_var  = torch.sum((masks - pred_imgs)**2, dim=(1,2)) / enable_pixels

        # loss
        loss1 = 0
        for i in range(BATCH_SIZE):
            loss1 += mse_loss(pred_imgs[i], uv_imgs[i]) / torch.sum(masks[i])
        loss1 = loss1 * self.PIXEL_NUM**2*3 / self.BATCH_SIZE

        loss2 = mse_loss(torch.zeros(1), torch.zeros(1))
        loss3 = mse_loss(torch.zeros(1), torch.zeros(1))
        # loss2 = mse_loss(pred_colors.float().cuda(), ori_colors.float().cuda())
        # loss3 = mse_loss(pred_color_embedding.float().cuda(), color_embedding.float().cuda())
        loss4 = mse_loss(pred_color_mean, ori_color_mean)
        loss5 = mse_loss(pred_color_var, ori_color_var)

        return pred_colors, pred_imgs, loss1, loss2, loss3, loss4, loss5
