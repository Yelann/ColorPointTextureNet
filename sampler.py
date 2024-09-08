import os
import numpy as np
from action import Action
from models.samplenet.samplenet import SampleNet
from utuils.checkpoint import load_samplenet, load_samplenet2
import trimesh
import torch

pixel_num = 128
in_points = 2048
sample_num = 256
name = 'e5f797c0266733a49b8de25d88149da8'
# name = 'airplane'
path = f"/data2/ShapeNetCoreColor/uv_model_512/04379243/{name}"
init_path = os.path.join(path, f'uniform_{in_points}.npz')


load_path = '/data2/ShapeNetCoreColor/output_model/main/128/model_main_snet_8e-4_pre_1024_256_last.pth'
# load_path = '/data2/ModelNet40_Sampled/log/SampleNet/M_I1024_O256_E400_SGD_0.0008_sampleronly_sampler_last.pth'
save_path = '/data2/ShapeNetCoreColor/output/sample/'

def get_data():
    # mesh_path = "/data2/ModelNet40_Sampled/uniform/2048/airplane/test/airplane_0726.ply"
    # mesh = trimesh.load(mesh_path)

    # vertices = np.array(mesh.vertices)
    # point_set = np.vstack(vertices)

    point_set = np.load(init_path)['points']
    choice = np.random.choice(len(point_set), 1024, replace=True)
    point_set = np.expand_dims(point_set[choice, :], 0)
    point_set = torch.from_numpy(point_set.astype(np.float32))

    # mean = np.expand_dims(np.mean(points, axis=0), 0)
    # points = points - mean  # center
    # dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    # points = points / dist  # scale
    return point_set


def save_ply(out_dir, name, points):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, name)
    trimesh.Trimesh(vertices=points, process=False).export(out_path)



sampler = SampleNet(
    num_out_points=sample_num,
    bottleneck_size=128,
    group_size=8,
    initial_temperature=1.0,
    input_shape="bnc",
    output_shape="bnc",
    skip_projection=False,
)
load_samplenet(sampler, load_path)
# load_samplenet2(sampler, load_path)
sampler = sampler.cuda()
sampler.eval()


points = get_data()
print(points.shape)

_, sampled_points = sampler(points.cuda())

sampled_points = sampled_points.reshape(-1, 3).cpu().numpy()
print(sampled_points.shape)

save_ply(f"{save_path}/{name}", f"{sample_num}_after.ply", sampled_points)
# save_ply(f"{save_path}/{name}", "ori_after.ply", points)
# np.savez(, points=sample_points[9].cpu().detach().numpy(), colors=pred_colors[9].cpu().detach().numpy())