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
