# from torch_cluster import fps, knn
import torch
from torch_geometric.nn.unpool import knn_interpolate
import numpy as np

def pc_to_uv(pc_feature_s, pc_pos_s, uv_pos_s):
    '''
    does not support batch processing!!
    '''
    features = []
    for b in range(pc_pos_s.size()[0]):
        pc_feature, pc_pos, uv_pos = pc_feature_s[b], pc_pos_s[b], uv_pos_s[b]
        h, w, _ = uv_pos.size()
        uv_pos = uv_pos.reshape(h*w, 3)

        # node_feature (Nxf), nod_pos (Nxd), unsample_pos (Mxd)
        uv_scatter_feature = knn_interpolate(pc_feature.cuda(), pc_pos.cuda(), uv_pos.cuda(), k=3)
        n, c = pc_feature.size()

        uv_scatter_feature = uv_scatter_feature.reshape(h, w, c)
        features.append(uv_scatter_feature)
    features = torch.stack(features, dim=0)
    return features


# def pc_to_uv(pc_feature_s, pc_pos_s, uv_pos_s, masks, h, w):
#     '''
#     does not support batch processing!!
#     '''
#     features = []
#     for b in range(pc_pos_s.size()[0]):
#         pc_feature, pc_pos, uv_pos, mask = pc_feature_s[b], pc_pos_s[b], uv_pos_s[b], masks[b]
#         uv_pos = uv_pos.reshape(-1, 3)
#         mask = mask[:, :, 0].reshape(-1)
#         mask_indices = mask == 1
#         uv_pos = uv_pos[mask_indices]

#         # h, w, _ = uv_pos.size()
#         # uv_pos = uv_pos.reshape(h*w, 3)

#         # node_feature (Nxf), nod_pos (Nxd), unsample_pos (Mxd)
#         uv_scatter_feature = knn_interpolate(pc_feature.cuda(), pc_pos.cuda(), uv_pos.cuda(), k=3)

#         uv_scatter_feature = uv_scatter_feature.reshape(-1, 3)
#         uv_img = torch.zeros(h * w, 3).cuda()
#         uv_img[mask_indices] = uv_scatter_feature
#         uv_img = uv_img.reshape(h, w, 3)
#         features.append(uv_img)
#         # uv_scatter_feature = uv_scatter_feature.reshape(h, w, c)
#         # features.append(uv_scatter_feature)
#     features = torch.stack(features, dim=0)
#     return features
def find_uv_from_pos(points_list, coords_list, pixel_num=128):
  
    uv_list = []
    for points, coords in zip(points_list, coords_list):
        uvs = []
        for point in points:
            distances = np.linalg.norm(coords - point, axis=-1)
            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            v, u = min_index
            v = pixel_num - v - 1
            min_index = (u, v)
            uvs.append(min_index)

        uvs = np.stack(uvs)
        uv_list.append(uvs)
    uv_list = np.stack(uv_list)
    uv_list = np.clip(uv_list, a_min=0, a_max=pixel_num-1) / pixel_num
    return uv_list
