import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def read_and_save_img(folder_path):
    ori_pixel_num = 512
    ori_coord_path = os.path.join(folder_path, f"uv_coord_{ori_pixel_num}.png")
    ori_uv_path = os.path.join(folder_path, f"uv_texture_{ori_pixel_num}.png")
    ori_mask_path = os.path.join(folder_path, f"uv_mask_{ori_pixel_num}.png")
    ori_normal_path = os.path.join(folder_path, f"uv_normal_{ori_pixel_num}.png")
    ori_coord_path = os.path.join(folder_path, f"uv_position_{ori_pixel_num}.npz")
    

    pixel_num = 64
    coord_img_path = os.path.join(folder_path, f"uv_coord_{pixel_num}.png")
    uv_path = os.path.join(folder_path, f"uv_texture_{pixel_num}.png")
    mask_path = os.path.join(folder_path, f"uv_mask_{pixel_num}.png")
    normal_path = os.path.join(folder_path, f"uv_normal_{pixel_num}.png")
    coord_path = os.path.join(folder_path, f"uv_position_{pixel_num}.npz")


    coord = np.load(ori_coord_path)
    coord = coord['points'].astype(np.float32)
    coord[np.isnan(coord)] = 0
    coord = coord.reshape(ori_pixel_num, ori_pixel_num, 3)
    coord = np.flipud(coord)
    rate = (coord.max() - coord.min())
    offset = (coord.max() - coord.min()) / 2

    coord = (coord + offset) / rate
    coord = cv2.resize(coord, (pixel_num, pixel_num), interpolation=cv2.INTER_NEAREST)
    mask = cv2.imread(mask_path)
    cv2.imwrite(coord_img_path, coord*mask)
    coord = coord * rate - offset
    coord.reshape(-1, 3)
    np.savez(coord_path, points=coord)

    # coord = cv2.imread(ori_coord_path)
    # coord = cv2.resize(coord, (pixel_num, pixel_num))

    uv = cv2.imread(ori_uv_path)
    uv = cv2.resize(uv, (pixel_num, pixel_num), interpolation=cv2.INTER_NEAREST)

    mask = cv2.imread(ori_mask_path)
    mask = cv2.resize(mask, (pixel_num, pixel_num), interpolation=cv2.INTER_NEAREST)

    normal = cv2.imread(ori_normal_path)
    normal = cv2.resize(normal, (pixel_num, pixel_num), interpolation=cv2.INTER_NEAREST)

    # cv2.imwrite(coord_path, coord)
    cv2.imwrite(uv_path, uv)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(normal_path, normal)



root = "/data2/ShapeNetCoreColor"
folder_num = '04379243'
train_split_file = f"{root}/final_split_files/{folder_num}/train.lst"
test_split_file = f"{root}/final_split_files/{folder_num}/test.lst"

folder_path = f"{root}/uv_model_512/{folder_num}/"



with open(train_split_file, 'r') as f:
    index = 0
    # total = len(f.readlines())
    total = 7279
    for line in f:
        name = line.strip().split()[0]
        read_and_save_img(os.path.join(folder_path, name))
        index += 1
        if index % 100 == 0:
            print(f"Train: {index}/{total} {index/total*100}%")

with open(test_split_file, 'r') as f:
    index = 0
    total = 968
    for line in f:
        name = line.strip().split()[0]
        read_and_save_img(os.path.join(folder_path, name))
        index += 1
        if index % 100 == 0:
            print(f"Test: {index}/{total} {index/total*100}%")
