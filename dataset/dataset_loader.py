import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import trimesh
import tqdm
import cv2
from torch_geometric.nn import fps
from utuils.sample_utuils import get_all_v_colors

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root='/mnt/d/Project/Dataset/PointUVDiff',
                 folder_num='04379243',
                 npoints=1600,
                 spoints=256,
                 split='train',
                ):
        self.npoints = npoints
        self.spoints = spoints
        self.root = root
        self.folder_num = folder_num
        
        if split == 'train':
            # split_file = f"{root}/final_split_files/{folder_num}/{npoints}/train.lst"
            split_file = f"{root}/final_split_files/{folder_num}/train.lst"
        elif split == 'test':
            # split_file = f"{root}/final_split_files/{folder_num}/{npoints}/test.lst"
            split_file = f"{root}/final_split_files/{folder_num}/test.lst"

        gt_dir = f"{root}/uv_model_512/{folder_num}/"
        gt_img_save_dir = f"{root}/evaluation/gt"
        os.makedirs(gt_img_save_dir, exist_ok=True)
        self.datapath = []

        with open(split_file, 'r') as f:
            for line in f:
                name = line.strip().split()[0]
                model_path = os.path.join(gt_dir, name, 'uv_texture_512.obj')
                gt_img_path = os.path.join(gt_dir, name, 'uv_texture_512.png')
                # coord_path = os.path.join(gt_dir, name, 'uv_position_512.npz')
                mask_path = os.path.join(gt_dir, name, "uv_mask_512.png")
                coord_path = os.path.join(gt_dir, name, "uv_coord_512.png")
                # mtl_path = os.path.join(gt_dir, name, "uv_texture_512.mlt")
                s_path = os.path.join(gt_dir, name, f"{spoints}")
                s_point_path = os.path.join(gt_dir, name, f"{spoints}/uv_s_points_{spoints}.npy")
                s_color_path = os.path.join(gt_dir, name, f"{spoints}/uv_s_colors_{spoints}.npy")
                s_uv_path = os.path.join(gt_dir, name, f"{spoints}/uv_s_uvs_{spoints}.npy")
                self.datapath.append((model_path, gt_img_path, coord_path, mask_path, s_path, s_point_path, s_color_path, s_uv_path))


    def __getitem__(self, index):
        model_path, uv_path, coord_path, mask_path, s_path, s_point_path, s_color_path, s_uv_path = self.datapath[index]

        uv_img = cv2.imread(uv_path)
        uv_img = (uv_img) / 255. * 2 - 1    # normalized to [-1,1]
        coord_img = cv2.imread(coord_path)
        coord_img = (coord_img) / 255. * 2 - 1   # normalized to [-1,1]
        mask = cv2.imread(mask_path)
        mask = (mask) / 255.    # normalized to [0,1]

        if os.path.exists(s_point_path):
            sample_points = np.load(s_point_path)
            sample_color = np.load(s_color_path)
            sample_uv = np.load(s_uv_path)
        else:
            mesh = trimesh.load(model_path)
            vertices = np.array(mesh.vertices)
            points = np.vstack(vertices)
            tex_coords = mesh.visual.uv

            # sample
            points = torch.from_numpy(points)
            ratio = self.spoints / points.shape[0]
            indices = fps(points, ratio=ratio)
            indices = indices[:self.spoints]

            uv_colors, uv_coords = get_all_v_colors(tex_coords, uv_img)

            sample_points = points[indices]
            sample_color = uv_colors[indices]
            sample_uv = uv_coords[indices]
            os.makedirs(s_path, exist_ok=True)
            np.save(s_point_path, sample_points)
            np.save(s_color_path, sample_color)
            np.save(s_uv_path, sample_uv)
        
        return uv_img.copy(), coord_img.copy(), mask.copy(), sample_points, sample_color, sample_uv

    def __len__(self):
        return len(self.datapath)
