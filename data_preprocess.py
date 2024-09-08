<<<<<<< Updated upstream
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import trimesh
import torch
from torch_geometric.nn import fps
from utuils.sample_utuils import get_all_v_colors, uniformSample
from models.global_shape_condition import get_shape_condition

def save_coord(file_path):
    read_path = os.path.join(file_path, 'uv_position_512.npz')
    coord_path = os.path.join(file_path, 'uv_coord_512.png')
    normal_path = os.path.join(file_path, 'uv_normal_512.png')
    mask_path = os.path.join(file_path, 'uv_mask_512.png')

    pointcloud_dict = np.load(read_path)
    points = pointcloud_dict['points'].astype(np.float32)
    normals = pointcloud_dict['normals'].astype(np.float32)

    points[np.isnan(points)] = 0
    normals[np.isnan(normals)] = 0

    coord = points.reshape(512, 512, 3)
    coord = np.flipud(coord)
    min_val = coord.min()
    max_val = coord.max()

    nor_img = normals.reshape(512, 512, 3)  # -1...1
    nor_img = (nor_img + 1) / 2
    nor_img = np.flipud(nor_img)

    mask = cv2.imread(mask_path)
    mask = mask / 255

    coord = (coord - min_val) / (max_val - min_val)
    coord = coord * mask
    nor_img = nor_img * mask

    plt.imsave(coord_path, coord)
    plt.imsave(normal_path, nor_img)


def save_uniform(model_path, file_path, sample_num, uv_img):
    mesh = trimesh.load(model_path)
    face_set = torch.from_numpy(mesh.faces).int().T
    point_set = torch.from_numpy(mesh.vertices)
    tex_coords = torch.from_numpy(mesh.visual.uv)
    sample_points, sample_coords = uniformSample(sample_num, point_set, face_set, tex_coords)
    sample_colors = get_all_v_colors(sample_coords, uv_img)
    sample_points = sample_points.numpy()

    save_path = os.path.join(file_path, f'uniform_{sample_num}.npz')
    np.savez(save_path, points=sample_points, colors=sample_colors, texs=sample_coords)


def get_paths(gt_dir, name):
    model_path = os.path.join(gt_dir, name, 'uv_texture_512.obj')
    uv_path = os.path.join(gt_dir, name, 'uv_texture_512.png')
    normal_path = os.path.join(gt_dir, name, 'uv_normal_512.png')
    coord_path = os.path.join(gt_dir, name, 'uv_position_512.npz')
    mask_path = os.path.join(gt_dir, name, "uv_mask_512.png")
    file_path = os.path.join(gt_dir, name)
    return model_path, uv_path, normal_path, coord_path, mask_path, file_path


def save_condition(model_path, dim=10):
    mesh = trimesh.load(model_path)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    shape_conditions = get_shape_condition(vertices, faces, dim)
    save_path = os.path.join(file_path, f'shape_condition.npz')
    key = 'dim' + str(dim)
    if os.path.exists(save_path):
        data = np.load(save_path)
        existing_data = {key: data[key] for key in data}
        existing_data[key] = shape_conditions
        np.savez(save_path, **existing_data)
    else:
        np.savez(save_path, key=shape_conditions)


def read_condition(condition_path, dim):
    shape_conditions = np.load(condition_path)['dim']
    return shape_conditions.min(), shape_conditions.max()


def fps_preprocess(file_path, sample_num, num):
    uniform_path = os.path.join(gt_dir, name, f'uniform_{in_points}.npz')
    

    uniform_init_data = np.load(uniform_path)
    vertices = uniform_init_data['points']
    colors = uniform_init_data['colors']
    uvs = uniform_init_data['texs']
    point_set = torch.tensor(vertices)

    ratio = sample_num / point_set.shape[0]
    for i in range(num):
        indices = fps(point_set.cuda(), ratio=ratio).cpu()
        indices = indices[:sample_num]
        sample_points = vertices[indices]
        sample_colors = colors[indices]
        sample_coords = uvs[indices]
        save_path = os.path.join(file_path, f'fps_{sample_num}_{i}.npz')
        np.savez(save_path, points=sample_points, colors=sample_colors, coords=sample_coords)


def pack_imgs(file_path, pixel_num=128):
    save_path = os.path.join(file_path, f'data_{pixel_num}.npz')

    uv_path = os.path.join(file_path, f"uv_texture_{pixel_num}.png")
    normal_path = os.path.join(file_path, f"uv_normal_{pixel_num}.png")
    mask_path = os.path.join(file_path, f"uv_mask_{pixel_num}.png")
    coord_path = os.path.join(file_path, f'uv_position_{pixel_num}.npz')

    uv_img = cv2.imread(uv_path) / 255.
    normal_img = cv2.imread(normal_path) / 255.    # 0...1
    mask = cv2.imread(mask_path) / 255.
    coord_img = np.load(coord_path)['points'].reshape(pixel_num, pixel_num, 3)

    np.savez(save_path, texture=uv_img, normal=normal_img, mask=mask, coord=coord_img)


root = "/data2/ShapeNetCoreColor"
# folder_num = '03001627'
folder_num = '04379243'
train_split_file = f"{root}/final_split_files/{folder_num}/train.lst"
test_split_file = f"{root}/final_split_files/{folder_num}/test.lst"

gt_dir = f"{root}/uv_model_512/{folder_num}/"
datapath = []
in_points = 50000
sample_num = 256

with open(train_split_file, 'r') as f:
    index = 0
    # total = len(f.readlines())
    total = 7276
    for line in f:
        if index > -1:
            name = line.strip().split()[0]
            model_path, uv_path, normal_path, coord_path, mask_path, file_path = get_paths(gt_dir, name)
            # save_coord(file_path)
            # uv_img = cv2.imread(uv_path) / 255.
            # save_uniform(model_path, file_path, 2048, uv_img)
            # condition_path = os.path.join(file_path, f'shape_condition.npz')
            # save_condition(model_path, 10)
            pack_imgs(file_path, 64)
            # fps_preprocess(file_path, sample_num, 15)
               
            
            if index % 1 == 0:
                print(f"Train: {index}/{total} {index/total*100}%")
        index += 1

with open(test_split_file, 'r') as f:
    index = 0
    total = 1000
    for line in f:
        if index > -1:
            name = line.strip().split()[0]
            model_path, uv_path, normal_path, coord_path, mask_path, file_path = get_paths(gt_dir, name)
            
            # save_coord(file_path)
            # uv_img = cv2.imread(uv_path) / 255.
            # save_uniform(model_path, file_path, 2048, uv_img)
            # condition_path = os.path.join(file_path, f'shape_condition.npz')
            # print(name)
            # save_condition(model_path, 10)
            # fps_preprocess(file_path, sample_num, 15)
            pack_imgs(file_path, 64)
            
            if index % 10 == 0:
                print(f"Test: {index}/{total} {index/total*100}%")

        index += 1



=======
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import trimesh
import torch
from torch_geometric.nn import fps
from utuils.sample_utuils import get_all_v_colors, uniformSample
from models.global_shape_condition import get_shape_condition

def save_coord(file_path):
    read_path = os.path.join(file_path, 'uv_position_512.npz')
    coord_path = os.path.join(file_path, 'uv_coord_512.png')
    normal_path = os.path.join(file_path, 'uv_normal_512.png')
    mask_path = os.path.join(file_path, 'uv_mask_512.png')

    pointcloud_dict = np.load(read_path)
    points = pointcloud_dict['points'].astype(np.float32)
    normals = pointcloud_dict['normals'].astype(np.float32)

    points[np.isnan(points)] = 0
    normals[np.isnan(normals)] = 0

    coord = points.reshape(512, 512, 3)
    coord = np.flipud(coord)
    min_val = coord.min()
    max_val = coord.max()

    nor_img = normals.reshape(512, 512, 3)  # -1...1
    nor_img = (nor_img + 1) / 2
    nor_img = np.flipud(nor_img)

    mask = cv2.imread(mask_path)
    mask = mask / 255

    coord = (coord - min_val) / (max_val - min_val)
    coord = coord * mask
    nor_img = nor_img * mask

    plt.imsave(coord_path, coord)
    plt.imsave(normal_path, nor_img)


def save_uniform(model_path, file_path, sample_num, uv_img):
    mesh = trimesh.load(model_path)
    face_set = torch.from_numpy(mesh.faces).int().T
    point_set = torch.from_numpy(mesh.vertices)
    tex_coords = torch.from_numpy(mesh.visual.uv)
    sample_points, sample_coords = uniformSample(sample_num, point_set, face_set, tex_coords)
    sample_colors = get_all_v_colors(sample_coords, uv_img)
    sample_points = sample_points.numpy()

    save_path = os.path.join(file_path, f'uniform_{sample_num}.npz')
    np.savez(save_path, points=sample_points, colors=sample_colors, texs=sample_coords)


def get_paths(gt_dir, name):
    model_path = os.path.join(gt_dir, name, 'uv_texture_512.obj')
    uv_path = os.path.join(gt_dir, name, 'uv_texture_512.png')
    normal_path = os.path.join(gt_dir, name, 'uv_normal_512.png')
    coord_path = os.path.join(gt_dir, name, 'uv_position_512.npz')
    mask_path = os.path.join(gt_dir, name, "uv_mask_512.png")
    file_path = os.path.join(gt_dir, name)
    return model_path, uv_path, normal_path, coord_path, mask_path, file_path


def save_condition(model_path, dim=10):
    mesh = trimesh.load(model_path)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    shape_conditions = get_shape_condition(vertices, faces, dim)
    save_path = os.path.join(file_path, f'shape_condition.npz')
    key = 'dim' + str(dim)
    if os.path.exists(save_path):
        data = np.load(save_path)
        existing_data = {key: data[key] for key in data}
        existing_data[key] = shape_conditions
        np.savez(save_path, **existing_data)
    else:
        np.savez(save_path, key=shape_conditions)


def read_condition(condition_path, dim):
    shape_conditions = np.load(condition_path)['dim']
    return shape_conditions.min(), shape_conditions.max()


def fps_preprocess(file_path, sample_num, num):
    uniform_path = os.path.join(gt_dir, name, f'uniform_{in_points}.npz')
    

    uniform_init_data = np.load(uniform_path)
    vertices = uniform_init_data['points']
    colors = uniform_init_data['colors']
    uvs = uniform_init_data['texs']
    point_set = torch.tensor(vertices)

    ratio = sample_num / point_set.shape[0]
    for i in range(num):
        indices = fps(point_set.cuda(), ratio=ratio).cpu()
        indices = indices[:sample_num]
        sample_points = vertices[indices]
        sample_colors = colors[indices]
        sample_coords = uvs[indices]
        save_path = os.path.join(file_path, f'fps_{sample_num}_{i}.npz')
        np.savez(save_path, points=sample_points, colors=sample_colors, coords=sample_coords)


def pack_imgs(file_path, pixel_num=128):
    save_path = os.path.join(file_path, f'data_{pixel_num}.npz')

    uv_path = os.path.join(file_path, f"uv_texture_{pixel_num}.png")
    normal_path = os.path.join(file_path, f"uv_normal_{pixel_num}.png")
    mask_path = os.path.join(file_path, f"uv_mask_{pixel_num}.png")
    coord_path = os.path.join(file_path, f'uv_position_{pixel_num}.npz')

    uv_img = cv2.imread(uv_path) / 255.
    normal_img = cv2.imread(normal_path) / 255.    # 0...1
    mask = cv2.imread(mask_path) / 255.
    coord_img = np.load(coord_path)['points'].reshape(pixel_num, pixel_num, 3)

    np.savez(save_path, texture=uv_img, normal=normal_img, mask=mask, coord=coord_img)


root = "/data2/ShapeNetCoreColor"
# folder_num = '03001627'
folder_num = '04379243'
train_split_file = f"{root}/final_split_files/{folder_num}/train.lst"
test_split_file = f"{root}/final_split_files/{folder_num}/test.lst"

gt_dir = f"{root}/uv_model_512/{folder_num}/"
datapath = []
in_points = 50000
sample_num = 256

with open(train_split_file, 'r') as f:
    index = 0
    # total = len(f.readlines())
    total = 7276
    for line in f:
        if index > -1:
            name = line.strip().split()[0]
            model_path, uv_path, normal_path, coord_path, mask_path, file_path = get_paths(gt_dir, name)
            # save_coord(file_path)
            # uv_img = cv2.imread(uv_path) / 255.
            # save_uniform(model_path, file_path, 2048, uv_img)
            # condition_path = os.path.join(file_path, f'shape_condition.npz')
            # save_condition(model_path, 10)
            pack_imgs(file_path, 64)
            # fps_preprocess(file_path, sample_num, 15)
               
            
            if index % 1 == 0:
                print(f"Train: {index}/{total} {index/total*100}%")
        index += 1

with open(test_split_file, 'r') as f:
    index = 0
    total = 1000
    for line in f:
        if index > -1:
            name = line.strip().split()[0]
            model_path, uv_path, normal_path, coord_path, mask_path, file_path = get_paths(gt_dir, name)
            
            # save_coord(file_path)
            # uv_img = cv2.imread(uv_path) / 255.
            # save_uniform(model_path, file_path, 2048, uv_img)
            # condition_path = os.path.join(file_path, f'shape_condition.npz')
            # print(name)
            # save_condition(model_path, 10)
            # fps_preprocess(file_path, sample_num, 15)
            pack_imgs(file_path, 64)
            
            if index % 10 == 0:
                print(f"Test: {index}/{total} {index/total*100}%")

        index += 1



>>>>>>> Stashed changes
