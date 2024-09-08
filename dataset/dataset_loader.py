<<<<<<< Updated upstream
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
=======
import os
import os.path

import numpy as np
import torch.utils.data as data
import random
import torch 

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root='/data2/ShapeNetCoreColor',
                 folder_num='04379243',
                 pixel_num=256,
                 in_points = 50000,
                 sample_points=256,
                 shape_dim=16,
                 split='train',
                 sampler='uniform'
                ):
        self.PIXEL_NUM = pixel_num
        self.IN_POINTS = in_points
        self.S_POINTS = sample_points
        self.SAMPLER = sampler
        self.SHAPE_DIM = shape_dim
        
        split_file = f"{root}/final_split_files/{folder_num}/{split}.lst"

        gt_dir = f"{root}/uv_model_512/{folder_num}/"
        self.datapath = []

        with open(split_file, 'r') as f:
            for line in f:
                name = line.strip().split()[0]
                # model_path = os.path.join(gt_dir, name, f"uv_texture_512.obj")
                data_path = os.path.join(gt_dir, name, f'data_{pixel_num}.npz')
                if sampler == 'samplenet':
                    path = os.path.join(gt_dir, name, f'uniform_{in_points}.npz')
                elif sampler == 'fps':
                    path = os.path.join(gt_dir, name, f'fps_{sample_points}_')
                elif sampler == 'uniform':
                    path = os.path.join(gt_dir, name, f'uniform_{sample_points}.npz')
                else:
                    path = os.path.join(gt_dir, name, f'uniform_{in_points}.npz')
                shape_path = os.path.join(gt_dir, name, f"shape_condition.npz")

                self.datapath.append((path, data_path, shape_path))
                # self.datapath.append((model_path, init_path, uniform_path, fps_path, data_path, shape_path))
                # self.datapath.append((uniform_path, fps_path, data_path, shape_path))


    def __getitem__(self, index):
        path, data_path, shape_path = self.datapath[index]
        # uniform_path, fps_path, data_path, shape_path = self.datapath[index]

        data = np.load(data_path)
        uv_img      = data['texture']
        # normal_img  = data['normal']    # 0...1
        mask        = data['mask']
        coord_img   = data['coord']

        if self.SHAPE_DIM == 0:
            shape_conditions = np.ndarray(0)
        else:
            shape_conditions = np.load(shape_path)['dim' + str(self.SHAPE_DIM)]
        
        if self.SAMPLER == 'fps':
            # uniform_init_data = np.load(init_path)
            # vertices    = uniform_init_data['points']
            # colors      = uniform_init_data['colors']
            # uvs         = uniform_init_data['texs']

            # ratio = self.S_POINTS / vertices.shape[0]
            # indices = fps(torch.tensor(vertices), ratio=ratio)
            # indices = indices[:self.S_POINTS]
            # sample_points = vertices[indices]
            # sample_colors = colors[indices]
            # sample_coords = uvs[indices]

            fps_index = random.randrange(15)
            fps_data = np.load(path + f'{fps_index}.npz')
            sample_points = fps_data['points']
            sample_colors = fps_data['colors']
            sample_coords = fps_data['coords']

            return uv_img, coord_img, mask, shape_conditions, sample_points.copy(), sample_colors.copy(), sample_coords.copy()

        elif self.SAMPLER == 'uniform':
            uniform_init_data = np.load(path)
            sample_points = uniform_init_data['points']
            sample_colors = uniform_init_data['colors']
            sample_coords = uniform_init_data['texs']

            # mesh = trimesh.load(model_path)
            # mesh.update_faces(mesh.unique_faces())
            # vertices = np.array(mesh.vertices)
            # uvs = np.array(mesh.visual.uv)

            # sample_points, sample_coords = uniformSample(self.S_POINTS, torch.tensor(vertices), torch.tensor(mesh.faces).int().T, torch.from_numpy(uvs))
            # sample_colors = get_all_v_colors(sample_coords, uv_img)
            # sample_points = sample_points.numpy()

            return uv_img, coord_img, mask, shape_conditions, sample_points.copy(), sample_colors.copy(), sample_coords.copy()
        
        elif self.SAMPLER == 'samplenet':
            uniform_init_data = np.load(path)
            
            points = uniform_init_data['points']
            mean = np.expand_dims(np.mean(points, axis=0), 0)
            points = points - mean  # center
            dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
            points = points / dist  # scale

            choice = np.random.choice(len(points), 2048, replace=True)
            points = points[choice, :]

            coord_img = ((coord_img - mean) / dist) * mask

            return uv_img, coord_img, mask, shape_conditions, points.copy()
        
        elif self.SAMPLER == 'random':
            uniform_init_data = np.load(path)
            
            points = uniform_init_data['points']
            choice = np.random.choice(len(points), self.S_POINTS, replace=True)
            sample_points = uniform_init_data['points'][choice, :]
            sample_colors = uniform_init_data['colors'][choice, :]
            sample_coords = uniform_init_data['texs'][choice, :]

            return uv_img, coord_img, mask, shape_conditions, sample_points.copy(), sample_colors.copy(), sample_coords.copy()

        # return uv_img, coord_img, normal_img, mask, shape_conditions, sample_points.copy(), sample_colors.copy(), sample_coords.copy()
        

    def __len__(self):
        return len(self.datapath)


class ShapeNetVAEDataset(data.Dataset):
    def __init__(self,
                 root='/data2/ShapeNetCoreColor',
                 folder_num='04379243',
                 pixel_num=256,
                 split='train',
                 sampler='uniform'
                ):
        self.PIXEL_NUM = pixel_num
        self.SAMPLER = sampler
        
        if split == 'train':
            split_file = f"{root}/final_split_files/{folder_num}/train.lst"
        elif split == 'test':
            split_file = f"{root}/final_split_files/{folder_num}/test.lst"

        gt_dir = f"{root}/uv_model_512/{folder_num}/"
        self.datapath = []
        with open(split_file, 'r') as f:
            for line in f:
                name = line.strip().split()[0]
                data_path = os.path.join(gt_dir, name, f'data_{pixel_num}.npz')
                self.datapath.append((data_path))


    def __getitem__(self, index):
        data_path = self.datapath[index]

        data = np.load(data_path)
        uv_img      = data['texture']
        normal_img  = data['normal']    # 0...1
        mask        = data['mask']
        coord_img   = data['coord']
        
        return uv_img.copy(), coord_img.copy(), normal_img.copy(), mask.copy()

    def __len__(self):
        return len(self.datapath)
>>>>>>> Stashed changes
