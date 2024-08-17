import numpy as np
import torch

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
    vertex_colors = []
    for tex_coord in tex_coords:
        u = int(tex_coord[0] * (uv_img.shape[1] - 1))
        v = int((1 - tex_coord[1]) * (uv_img.shape[0] - 1))  # 纹理坐标的V轴需要反转
        vertex_colors.append(uv_img[v, u])

    vertex_colors = np.array(vertex_colors)
    return vertex_colors


def create_sample_tex_img(coords, colors, mask, pixel_num):
    img = torch.full((pixel_num, pixel_num, 3), 1).float() * mask.cpu()
    for coord, color in zip(coords, colors):
        u = int(coord[0] * (pixel_num - 1))
        v = int((1 - coord[1]) * (pixel_num - 1))  # 纹理坐标的V轴需要反转
        for ui in range(3):
            for vi in range(3):
                img[max(0,v-vi), max(0, u-ui)] = color
                img[max(0,v-vi), min(pixel_num-1, u+ui)] = color
                img[min(pixel_num-1,v+vi), min(pixel_num-1, u+ui)] = color
                img[min(pixel_num-1,v+vi), max(0, u-ui)] = color
    return img


def get_all_colors(uv_coords, uv_imgs):
    color_list = []
    for coords, uv_img in zip(uv_coords, uv_imgs):
        colors = []
        for coord in coords:
            u = int(coord[0] * (uv_img.shape[1] - 1))
            v = int((1 - coord[1]) * (uv_img.shape[0] - 1))  # 纹理坐标的V轴需要反转
            colors.append(uv_img[v, u])
        colors = torch.stack(colors)
        color_list.append(colors)
    color_list = torch.stack(color_list)
    return color_list


def uniformSample(sample_num, pos: torch.Tensor, face: torch.Tensor, tex: torch.Tensor):
    pos_max = pos.abs().max()
    pos = pos / pos_max

    area = (pos[face[1]] - pos[face[0]]).cross(
        pos[face[2]] - pos[face[0]],
        dim=1,
    )
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    sample = torch.multinomial(prob, sample_num, replacement=True)
    face = face[:, sample]

    frac = torch.rand(sample_num, 2, device=pos.device)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]
    t_vec1 = tex[face[1]] - tex[face[0]]
    t_vec2 = tex[face[2]] - tex[face[0]]

    pos_sampled = pos[face[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2
    
    tex_sampled = tex[face[0]]
    tex_sampled += frac[:, :1] * t_vec1
    tex_sampled += frac[:, 1:] * t_vec2

    pos_sampled = pos_sampled * pos_max

    return pos_sampled, tex_sampled