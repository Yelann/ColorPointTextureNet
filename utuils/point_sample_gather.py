# from torch_cluster import fps, knn
import torch
from torch_geometric.nn.unpool import knn_interpolate

def pc_to_uv(pc_feature_s, pc_pos_s, uv_pos_s):
    '''
    does not support batch processing!!
    '''
    features = []
    for b in range(pc_pos_s.size()[0]):
        pc_feature, pc_pos, uv_pos = pc_feature_s[b], pc_pos_s[b], uv_pos_s[b]
        h, w, _ = uv_pos.size()
        uv_pos = uv_pos.reshape(h*w, 3)

        uv_scatter_feature = knn_interpolate(pc_feature.cuda(), pc_pos.cuda(), uv_pos.cuda(), k=3)
        n, c = pc_feature.size()

        uv_scatter_feature = uv_scatter_feature.reshape(h, w, c)
        features.append(uv_scatter_feature)
    features = torch.stack(features, dim=0)
    return features