<<<<<<< Updated upstream
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
    
=======
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from typing import Any, List
from utuils.point_sample_gather import pc_to_uv
from utuils.sample_utuils import get_all_colors
from models.pvcnn.model.cond_pvcnn_generation import Cond_PVCNN2 as PVCNN

mse_loss = nn.MSELoss()

class PositionEncoder(nn.Module):
    def __init__(self, input_dim=3, encoding_dim=16):
        super(PositionEncoder, self).__init__()
        self.encoding_dim = encoding_dim
        
        self.freq_bands = torch.tensor(
            [1 / (10000 ** (2 * (i // 2) / encoding_dim)) for i in range(encoding_dim)]
        )

    def forward(self, sample_points):
        batch_size, num_points, _ = sample_points.size()    # [batch_size, num_points, 3]

        x = sample_points.unsqueeze(-1)  # [batch_size, num_points, input_dim, 1]
        freq_bands = self.freq_bands.unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, encoding_dim]
        angle = x * freq_bands  # [batch_size, num_points, input_dim, encoding_dim]

        encodings = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # [batch_size, num_points, input_dim, 2*encoding_dim]
        encodings = encodings.view(batch_size, -1, num_points)  # [batch_size, input_dim * 2 * encoding_dim, num_points]

        return encodings


class PointUNet(PVCNN):
    def __init__(self, num_classes=6, use_att=True, dropout=0.2, extra_feature_channels=64, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(num_classes=num_classes, embed_dim=0, use_att=use_att, dropout=dropout, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


    def forward(self, fps_pos, feature_embs):
        features = feature_embs    # 32, 32, 256
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
            # nn.Conv1d(6, 64, 1),
            nn.Conv1d(9, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, pred_results, sample_points):
        x = torch.cat((pred_results, sample_points), dim=2).transpose(1, 2)
        y = self.layers(x).transpose(1, 2)
        return y



class CPTNet(nn.Module):
    def __init__(self, args):
        super(CPTNet, self).__init__()
        self.PIXEL_NUM = args.pixel_num
        self.SAMPLE_NUM = args.num_out_points
        self.COLOR_EMB_DIM = args.num_emb_dim
        self.POS_EMB_DIM = args.num_pos_dim
        self.SHAPE_EMB_DIM = args.num_shape_dim
        self.BATCH_SIZE = args.batch_size

        self.point_encoder = PositionEncoder(encoding_dim=self.POS_EMB_DIM)
        self.unet = PointUNet(num_classes=6, use_att=True, dropout=0.2, extra_feature_channels=self.COLOR_EMB_DIM+self.SHAPE_EMB_DIM+6*self.POS_EMB_DIM-3, width_multiplier=1, voxel_resolution_multiplier=1)
        self.mlp = MLP()


    def forward(self, color_embeddings, shape_conditions, data):
        uv_imgs, coord_imgs, masks, sample_points = data
        sample_points = sample_points.float()    # [batch, sample_num, 3]
        BATCH_SIZE = coord_imgs.shape[0]


        # ======POINT ENCODER======
        position_embeddings = self.point_encoder(sample_points)
        
        # ======DIFF======
        if self.SHAPE_EMB_DIM == 0:
            feature_embeddings = color_embeddings
        else:
            feature_embeddings = torch.cat([color_embeddings, shape_conditions], dim=1)
        feature_embeddings = torch.cat([feature_embeddings.reshape(BATCH_SIZE, -1, 1)] * self.SAMPLE_NUM, dim=2).float()    # [32, 64, 256]
        feature_embeddings = torch.cat([feature_embeddings, position_embeddings], dim=1)
        pred_result = self.unet(sample_points.reshape(BATCH_SIZE, 3, -1), feature_embeddings).transpose(1, 2)   # 0...1 [batch, sample_num, 6]

        # ======MPL======
        pred_colors = self.mlp(pred_result, sample_points)    #[batch, sample_num, 3]
        pred_colors = torch.clamp(pred_colors, min=0, max=1)

        # to uv
        color_input = pred_colors.float()    # [batch, sample_num, 3]
        coord_input = coord_imgs.float()

        pred_imgs = pc_to_uv(color_input, sample_points, coord_input)    # [batch, 512, 512, 3]
        pred_imgs = torch.clamp(pred_imgs, min=0, max=1)

        # pred_colors = get_all_colors(sample_uvs, pred_imgs)     # [batch, sample_num, 3]
        pred_imgs = pred_imgs * masks   # [0, 1]

        # loss 2 and loss 3
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
        loss1 = loss1.cpu()
        loss2 = mse_loss(pred_color_mean, ori_color_mean).cpu()
        loss3 = mse_loss(pred_color_var, ori_color_var).cpu()

        return pred_colors, pred_imgs, loss1, loss2, loss3


    def predict(self, color_embeddings, shape_conditions, data):
        coord_imgs, masks, sample_points = data
        sample_points = sample_points.float()    # [batch, sample_num, 3]
        BATCH_SIZE = coord_imgs.shape[0]

        # ======POINT ENCODER======
        position_embeddings = self.point_encoder(sample_points)
        
        # ======DIFF======
        feature_embeddings = torch.cat([color_embeddings, shape_conditions], dim=1)
        feature_embeddings = torch.cat([feature_embeddings.reshape(BATCH_SIZE, -1, 1)] * self.SAMPLE_NUM, dim=2).float()    # [32, 64, 256]
        feature_embeddings = torch.cat([feature_embeddings, position_embeddings], dim=1)
        pred_result = self.unet(sample_points.reshape(BATCH_SIZE, 3, -1), feature_embeddings).transpose(1, 2)   # 0...1 [batch, sample_num, 6]

        # ======MPL======
        pred_colors = self.mlp(pred_result, sample_points)    #[batch, sample_num, 3]
        pred_colors = torch.clamp(pred_colors, min=0, max=1)

        # to uv
        color_input = pred_colors.float()    # [batch, sample_num, 3]
        coord_input = coord_imgs.float()

        pred_imgs = pc_to_uv(color_input, sample_points, coord_input)    # [batch, 512, 512, 3]
        pred_imgs = torch.clamp(pred_imgs, min=0, max=1)

        # pred_colors = get_all_colors(sample_coords, pred_imgs)
        pred_imgs_mask = pred_imgs * masks   # [0, 1]
        return pred_colors, pred_imgs, pred_imgs_mask
>>>>>>> Stashed changes
