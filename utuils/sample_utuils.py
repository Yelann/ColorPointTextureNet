import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def parse_mtl(file_path):
    materials = {}
    current_material = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('newmtl'):
                current_material = line.split()[1]
                materials[current_material] = {}
            elif line.startswith('Ka'):
                materials[current_material]['ambient'] = list(map(float, line.split()[1:]))
            elif line.startswith('Kd'):
                materials[current_material]['diffuse'] = list(map(float, line.split()[1:]))
            elif line.startswith('Ks'):
                materials[current_material]['specular'] = list(map(float, line.split()[1:]))
            elif line.startswith('d'):
                materials[current_material]['dissolve'] = float(line.split()[1])
            elif line.startswith('illum'):
                materials[current_material]['illum'] = int(line.split()[1])
            elif line.startswith('map_Kd'):
                materials[current_material]['texture'] = line.split()[1]
    
    return materials


def parse_obj(file_path):
    vertices = []
    tex_coords = []
    faces = []
    face_textures = []
    material_name = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('vt '):
                tex_coords.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):
                face = []
                face_texture = []
                for v in line.split()[1:]:
                    v_split = v.split('/')
                    face.append(int(v_split[0]) - 1)
                    if len(v_split) > 1 and v_split[1]:
                        face_texture.append(int(v_split[1]) - 1)
                faces.append(face)
                face_textures.append(face_texture)
            elif line.startswith('usemtl'):
                material_name = line.split()[1]

    return np.array(vertices), np.array(tex_coords), np.array(faces), np.array(face_textures), material_name


def get_all_v_colors(tex_coords, uv_img):
    # 获取顶点的纹理颜色
    vertex_colors = []
    uv_coords = []
    for tex_coord in tex_coords:
        u = int(tex_coord[0] * (uv_img.shape[1] - 1))
        v = int((1 - tex_coord[1]) * (uv_img.shape[0] - 1))  # 纹理坐标的V轴需要反转
        vertex_colors.append(uv_img[v, u])
        uv_coords.append([u, v])

    vertex_colors = np.array(vertex_colors)
    uv_coords = np.array(uv_coords)
    return vertex_colors, uv_coords


def save_sample_tex_img(name, uvs, colors, mask):
    img = torch.full((512, 512, 3), 1).float() * mask.cpu()
    for uv, color in zip(uvs, colors):
        u, v = uv
        color = torch.tensor([color[2], color[1], color[0]])
        for ui in range(3):
            for vi in range(3):
                img[max(0,v-vi), max(0, u-ui)] = color
                img[max(0,v-vi), min(511, u+ui)] = color
                img[min(511,v+vi), min(511, u+ui)] = color
                img[min(511,v+vi), max(0, u-ui)] = color
    plt.imsave(f'/mnt/d/Project/ColorPointTextureNet/output/save_sample/{name}.png', img.numpy())