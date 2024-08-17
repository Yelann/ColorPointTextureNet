import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def save_coord(file_path):
    read_path = os.path.join(file_path, 'uv_position_512.npz')
    save_path = os.path.join(file_path, 'uv_coord_512.png')
    mask_path = os.path.join(file_path, 'uv_mask_512.png')

    pointcloud_dict = np.load(read_path)
    points = pointcloud_dict['points'].astype(np.float32)
    # print("Points", points.shape)
    normal = pointcloud_dict['normals'].astype(np.float32)
    # print("Normal", normal.shape)
    # color = pointcloud_dict['colors'].astype(np.float32)
    # print("Color", color.shape)

    normal[np.isnan(normal)] = 0

    coord = points.reshape(512, 512, 3)
    coord = np.flipud(coord)
    min_val = coord.min()
    max_val = coord.max()

    mask = cv2.imread(mask_path)
    # print(mask.shape)
    mask = mask / 255
    # print(mask.min(), mask.max())

    coord = (coord - min_val) / (max_val - min_val)
    coord = coord * mask
    plt.imsave(save_path, coord)


root = "/mnt/d/Project/Dataset/PointUVDiff"
folder_num = '04379243'
train_split_file = f"{root}/final_split_files/{folder_num}/train.lst"
test_split_file = f"{root}/final_split_files/{folder_num}/test.lst"

gt_dir = f"{root}/uv_model_512/{folder_num}/"
datapath = []

with open(train_split_file, 'r') as f:
    index = 0
    # total = len(f.readlines())
    total = 7279
    for line in f:
        name = line.strip().split()[0]
        # file_path = os.path.join(gt_dir, name, 'uv_texture_512.obj')
        # gt_img_path = os.path.join(gt_dir, name, 'uv_texture_512.png')
        # coord_path = os.path.join(gt_dir, name, 'uv_position_512.npz')
        # mask_path = os.path.join(gt_dir, name, "uv_mask_512.png")
        file_path = os.path.join(gt_dir, name)
        save_coord(file_path)
        index += 1
        if index % 100 == 0:
            print(f"Train: {index}/{total} {index/total*100}%")

with open(test_split_file, 'r') as f:
    index = 0
    total = 1000
    for line in f:
        name = line.strip().split()[0]
        # file_path = os.path.join(gt_dir, name, 'uv_texture_512.obj')
        # gt_img_path = os.path.join(gt_dir, name, 'uv_texture_512.png')
        # coord_path = os.path.join(gt_dir, name, 'uv_position_512.npz')
        # mask_path = os.path.join(gt_dir, name, "uv_mask_512.png")
        file_path = os.path.join(gt_dir, name)
        save_coord(file_path)
        index += 1
        if index % 100 == 0:
            print(f"Test: {index}/{total} {index/total*100}%")

