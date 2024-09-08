<<<<<<< Updated upstream
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import tqdm
from tqdm import tqdm
from src import FPSSampler, RandomSampler
from models.cptnet import CPTNet
from typing import Any, List
from utuils.point_sample_gather import pc_to_uv
import matplotlib.pyplot as plt
from utuils.sample_utuils import save_sample_tex_img


class Action:
    def __init__(self, args):
        # sampler
        self.NUM_IN_POINTS = args.num_in_points
        self.NUM_OUT_POINTS = args.num_out_points
        self.SAMPLER = args.sampler
        self.args = args

    def create_model(self):
        model = CPTNet(self.args)
        print("0")
        model.cuda()
        print("0")
        # model.load_state_dict(torch.load(self.POINTNET))
        # if self.TRAIN_POINTNET:
        #     model.requires_grad_(True)
        #     model.train()
        # else:
        #     model.requires_grad_(False)
        #     model.eval()

        # Attach sampler to pcrnet_model
        return model

    def train_1(self, model, trainloader, optimizer, epoch):
        print("1.1")
        vloss = 0.0
        count = 0
        batch_num = len(trainloader)
        mse_loss = nn.MSELoss()

        print(f"=====Start Train Epoch {epoch}=====")
        progress_bar = tqdm(trainloader, desc="Training", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            print("1.2")
            uv_imgs, coord_imgs, masks, sample_points, sample_colors, sample_uvs = data
            uv_imgs = uv_imgs.cuda()
            coord_imgs = coord_imgs.cuda()
            masks = masks.cuda()
            sample_points = sample_points.cuda()
            sample_colors = sample_colors.cuda()    # [-1, 1]
            sample_uvs = sample_uvs.cuda()
            data = (uv_imgs, coord_imgs, masks, sample_points, sample_colors)

            # model
            pred_colors = model(data)
            pred_colors = torch.clamp(pred_colors, min=0, max=1)    # [0, 1]

            color_input = pred_colors.float().permute(0, 1, 2)
            point_input = sample_points.transpose(1, 2)[:, :3, ...].float().permute(0, 2, 1)
            uv_input = uv_imgs.permute(0, 3, 1, 2).float()[:, :3, ...].permute(0, 2, 3, 1)

            pred_imgs = pc_to_uv(color_input, point_input, uv_input)
            pred_imgs = pred_imgs * masks   # [0, 1]
            gt_imgs = (uv_imgs + 1) / 2 * masks     # [0, 1]
            sample_colors_01 = (sample_colors + 1) / 2

            loss1 = mse_loss(pred_imgs.float().cuda(), gt_imgs.float().cuda())
            loss2 = mse_loss(pred_colors.float().cuda(), sample_colors_01.float().cuda())
            # print(loss1.item(), loss2.item())
            loss = loss1 + loss2

            progress_bar.set_postfix(l1=loss1.item(), l2=loss2.item(), loss=loss.item())

            # # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            vloss += loss.item()
            count += 1

        avg_vloss = float(vloss) / count
        print("TRAIN===loss", avg_vloss)
        return avg_vloss


    def eval_1(self, model, testloader, epoch):
        vloss = 0.0
        count = 0
        batch_num = len(testloader)
        mse_loss = nn.MSELoss()

        # print("=====Start Eval=====")
        progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            uv_imgs, coord_imgs, masks, sample_points, sample_colors, sample_uvs = data
            uv_imgs = uv_imgs.cuda()
            coord_imgs = coord_imgs.cuda()
            masks = masks.cuda()
            sample_points = sample_points.cuda()
            sample_colors = sample_colors.cuda()
            sample_uvs = sample_uvs.cuda()
            data = (uv_imgs, coord_imgs, masks, sample_points, sample_colors)

            # model
            pred_colors = model(data)
            pred_colors = torch.clamp(pred_colors, min=0, max=1)

            color_input = pred_colors.float().permute(0, 1, 2)
            point_input = sample_points.transpose(1, 2)[:, :3, ...].float().permute(0, 2, 1)
            uv_input = uv_imgs.permute(0, 3, 1, 2).float()[:, :3, ...].permute(0, 2, 3, 1)

            pred_imgs = pc_to_uv(color_input, point_input, uv_input)
            pred_imgs = pred_imgs * masks
            gt_imgs = (uv_imgs + 1) / 2 * masks
            sample_colors_01 = (sample_colors + 1) / 2

            loss1 = mse_loss(pred_imgs.float().cuda(), gt_imgs.float().cuda())
            loss2 = mse_loss(pred_colors.float().cuda(), sample_colors_01.float().cuda())
            # print(loss1, loss2)
            loss = loss1 + loss2

            if i % 5 == 0:
                if epoch == 0:
                    # img = gt_imgs[0].cpu().detach().numpy()
                    # img[:, :, [0, 2]] = img[:, :, [2, 0]]
                    # plt.imsave(f'/mnt/d/Project/ColorPointTextureNet/output/ori/{epoch}_{i}_{0}.png', img)
                    save_sample_tex_img(f'sample_{i}_{0}', sample_uvs[0], sample_colors_01[0], masks[0])
            
                img = pred_imgs[0].cpu().detach().numpy()
                img[:, :, [0, 2]] = img[:, :, [2, 0]]
                plt.imsave(f'/mnt/d/Project/ColorPointTextureNet/output/{epoch}_{i}_{0}.png', img)
                save_sample_tex_img(f'pred_{epoch}_{i}_{0}', sample_uvs[0], pred_colors[0], masks[0])

            vloss += loss.item()
            count += 1

        avg_vloss = float(vloss) / count
        print("EVAL===loss", avg_vloss)
        return avg_vloss


    def test_1(self, model, testloader, epoch, writer):
        vloss = 0.0
        count = 0

        print("=====Start Test=====")
        for i, data in enumerate(testloader):
            uv_imgs, coord, mask, sample_points = data

            # model
            sample_colors = model(data)
            sample_colors = (1 + sample_colors) / 2

            color_input = sample_colors.float().permute(0, 1, 2)
            point_input = sample_points.transpose(1, 2)[:, :3, ...].float().permute(0, 2, 1)
            uv_input = uv_imgs.permute(0, 3, 1, 2).float()[:, :3, ...].permute(0, 2, 3, 1)

            pred_image = pc_to_uv(color_input, point_input, uv_input)

            mse_loss = nn.MSELoss()
            loss = mse_loss(pred_image.float().cuda(), uv_imgs.float().cuda())
            # update
            vloss += loss.item()
            count += 1

        avg_vloss = float(vloss) / count
        print("TEST===loss", avg_vloss)
        return avg_vloss
=======
import torch
import torch.nn as nn
from torch.autograd import Variable

import os

# import tqdm
from tqdm import tqdm
from models.cptnet import CPTNet
from typing import Any, List
import matplotlib.pyplot as plt
from utuils.sample_utuils import create_sample_tex_img, get_all_colors

import numpy as np
# from models.cptnet import ColorEncoder
from models.color_vae import ColorVAE
from models.samplenet.samplenet import SampleNet
from utuils.checkpoint import save_checkpoint, load_checkpoint, load_vae
from utuils.point_sample_gather import pc_to_uv, find_uv_from_pos

import trimesh

SCALE = 100
mse_loss = nn.MSELoss()

class Action:
    def __init__(self, args):
        # sampler
        self.NUM_IN_POINTS = args.num_in_points
        self.NUM_OUT_POINTS = args.num_out_points
        self.SAMPLER = args.sampler
        self.args = args
        self.OUT_PATH = args.outfile
        self.PIXEL_NUM = args.pixel_num
        self.WRITER_PATH = args.writer_path
        self.TRAIN_MAIN = args.train_main
        self.TRAIN_VAE = args.train_vae
        self.TEST_VAE = args.test_vae
        self.TEST_MAIN = args.test_main

        self.COLOR_EMB_DIM = args.num_emb_dim

    def create_model(self):
        model = CPTNet(self.args)
        vae = ColorVAE(input_dim=3, embedding_dim=self.COLOR_EMB_DIM, pixel_num=self.PIXEL_NUM)
        
        if self.args.vae_path != '':
            load_path = self.args.outfile + f"/output_model/vae/{self.PIXEL_NUM}/" + self.args.vae_path
            load_vae(vae, load_path)
        
        if self.TRAIN_MAIN:
            model.requires_grad_(True)
            model.train()
        else:
            model.requires_grad_(False)
            model.eval()

        if self.TRAIN_VAE:
            vae.requires_grad_(True)
            vae.train()
        else:
            vae.requires_grad_(False)
            vae.eval()

        if self.SAMPLER == 'samplenet':
            sampler = SampleNet(
                num_out_points=self.NUM_OUT_POINTS,
                bottleneck_size=128,
                group_size=8,
                initial_temperature=1.0,
                input_shape="bnc",
                output_shape="bnc",
                skip_projection=False,
            )
            if self.TEST_MAIN:
                sampler.requires_grad_(False)
                sampler.eval()
            else:
                sampler.requires_grad_(True)
                sampler.train()
            sampler.cuda()
        else:
            sampler = 0

        model.cuda()
        vae.cuda()
        return model, vae, sampler


    def train_vae(self, model, trainloader, optimizer, epoch):
        total_loss = 0.0
        save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
        if epoch == 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(save_path + "vae_train")
            os.makedirs(save_path + "vae_eval")

        progress_bar = tqdm(trainloader, desc="Training", ncols=120, ascii=True)
        for i, data in enumerate(progress_bar):
            uv_imgs, _, _, masks = data
            pred_uv_imgs, _, loss = model.vae(uv_imgs.cuda(), masks.cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix(loss = loss.item())
            total_loss += loss.item()

            if i == 0 and epoch % 10 == 0:
                save_img(uv_imgs, save_path + f'vae_train/{epoch}_ori')
                save_img(pred_uv_imgs, save_path + f'vae_train/{epoch}')
            

        avg_loss = float(total_loss) / len(trainloader)
        print(f"TRAIN {epoch} === Loss {avg_loss}.")
        return avg_loss

    
    def eval_vae(self, vae, testloader, epoch):
        total_loss = 0.0
        save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"

        progress_bar = tqdm(testloader, desc="Testing", ncols=120, ascii=True)
        for i, data in enumerate(progress_bar):
            uv_imgs, _, _, masks = data
            pred_uv_imgs, _, loss = vae(uv_imgs.cuda(), masks.cuda())
            progress_bar.set_postfix(loss = loss.item())
            total_loss += loss.item()

            if i == 0 and epoch % 1 == 0:
                if epoch == 0:
                    save_img(uv_imgs, save_path + f'vae_eval/ori')
                save_img(pred_uv_imgs, save_path + f'vae_eval/{epoch}')
                
            
        avg_loss = float(total_loss) / len(testloader)
        print(f"EVAL === Loss {avg_loss}.")
        return avg_loss


    def test_vae(self, vae, testloader):
        save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.args.gen_vae:
            rand_imgs, _ = vae.get_random_result(32)
            save_img(rand_imgs, save_path + f'rand')
        elif self.args.test_vae:
            progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
            for i, data in enumerate(progress_bar):
                uv_imgs, _, _, masks = data
                pred_uv_imgs, _, _ = vae(uv_imgs.cuda(), masks.cuda(), False)
                pred_uv_imgs_mask, _, _ = vae(uv_imgs.cuda(), masks.cuda())
                if i == 0:
                    save_img(uv_imgs, save_path + f'ori')
                    save_img(pred_uv_imgs, save_path + f'pred')
                    save_img(pred_uv_imgs_mask, save_path + f'pred_mask')
                
        print(f"TEST === Finish.")
       

    def train_main(self, model, vae, trainloader, optimizer, epoch):
        total_loss_color = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        total_loss_emb = 0.0
        batch_num = len(trainloader)

        progress_bar = tqdm(trainloader, desc="Training", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            torch.cuda.empty_cache()
            uv_imgs, coord_imgs, masks, shape_conditions, sample_points, _, _ = data

            uv_imgs = uv_imgs.cuda()
            masks = masks.cuda()
            data = (uv_imgs, coord_imgs.cuda(), masks, sample_points.cuda())
            # model
            color_embeddings = vae.get_emb(uv_imgs)
            _, pred_imgs, loss1, loss2, loss3 = model(color_embeddings, shape_conditions.cuda(), data)
            pred_embeddings = vae.get_emb(pred_imgs)
            
            loss_emb = mse_loss(color_embeddings, pred_embeddings).cpu()
            loss_color, _ = self.compute_total_loss(loss1, loss2, loss3, loss_emb, 0)

            progress_bar.set_postfix(l1=loss1.item() * SCALE, l_color=loss_color.item() * SCALE)

            # update
            optimizer.zero_grad()
            loss_color.backward()
            optimizer.step()

            total_loss_color += loss_color.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss_emb += loss_emb.item()


        avg_loss_color = float(total_loss_color) / batch_num * SCALE
        avg_loss1 = float(total_loss1) / batch_num * SCALE
        avg_loss2 = float(total_loss2) / batch_num * SCALE
        avg_loss3 = float(total_loss3) / batch_num * SCALE
        avg_loss_emb = float(total_loss_emb) / batch_num * SCALE
        avg_loss_sample = 0 
        avg_loss_sampler = 0
        avg_loss_proj = 0
        avg_loss_sim = 0

        print(f"TRAIN {epoch} === Color {avg_loss_color}, Sample {avg_loss_sample}, l1 {avg_loss1}, l_emb {avg_loss_emb}.")
        return avg_loss_color, avg_loss_sample, avg_loss1, avg_loss2, avg_loss3, avg_loss_emb, avg_loss_sampler, avg_loss_proj, avg_loss_sim


    def eval_main(self, model, vae, testloader, epoch, sampler):
        total_loss_color = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        total_loss_emb = 0.0

        batch_num = len(testloader)
        save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
        if epoch == 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(save_path + "ori")
            os.makedirs(save_path + "train")
            os.makedirs(save_path + "eval")

        model.eval()

        progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            torch.cuda.empty_cache()
            uv_imgs, coord_imgs, masks, shape_conditions, sample_points, sample_colors, sample_coords = data

            uv_imgs = uv_imgs.cuda()
            masks = masks.cuda()
            data = (uv_imgs, coord_imgs.cuda(), masks, sample_points.cuda())
            # model
            color_embeddings = vae.get_emb(uv_imgs)
            pred_colors, pred_imgs, loss1, loss2, loss3 = model(color_embeddings, shape_conditions.cuda(), data)
            pred_embeddings = vae.get_emb(pred_imgs)
            loss_emb = mse_loss(color_embeddings, pred_embeddings).cpu()

            loss_color, _ = self.compute_total_loss(loss1, loss2, loss3, loss_emb, 0)
            
            if i == 0 and (epoch % 10 == 0 or epoch < 5):
                if epoch == 0:
                    save_imgs(uv_imgs, self.args.batch_size, sample_coords, sample_colors, masks, self.PIXEL_NUM, save_path + '/ori/eval', True)
                save_imgs(pred_imgs, self.args.batch_size, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'/eval/eval_{epoch}', False)

                np.savez(f"/homes/yz723/Project/ColorPointTextureNet/output/pred_9_{epoch}.npz", points=sample_points[9].cpu().detach().numpy(), colors=pred_colors[9].cpu().detach().numpy())
                out_path = f"/homes/yz723/Project/ColorPointTextureNet/output/pred_9_{epoch}.ply"
                trimesh.Trimesh(vertices=sample_points[9].cpu().detach().numpy(), process=False).export(out_path)
                    
                
            total_loss_color += loss_color.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss_emb += loss_emb.item()

        avg_loss_color = float(total_loss_color) / batch_num * SCALE
        avg_loss1 = float(total_loss1) / batch_num * SCALE
        avg_loss2 = float(total_loss2) / batch_num * SCALE
        avg_loss3 = float(total_loss3) / batch_num * SCALE
        avg_loss_emb = float(total_loss_emb) / batch_num * SCALE
        avg_loss_sample = 0
        avg_loss_sampler = 0

        model.train()

        print(f"EVAL {epoch} === Color {avg_loss_color}, Sample {avg_loss_sample}, l1 {avg_loss1}, l_emb {avg_loss_emb}.")
        return avg_loss_color, avg_loss_sample, avg_loss1, avg_loss2, avg_loss3, avg_loss_emb, avg_loss_sampler


    def train_sample(self, model, vae, trainloader, optimizer, epoch, sampler, optimizer_sam):
        total_loss_color = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        total_loss_emb = 0.0
        total_loss_sample = 0.0
        total_loss_sampler = 0.0
        total_loss_proj = 0.0
        total_loss_sim = 0.0
        batch_num = len(trainloader)

        progress_bar = tqdm(trainloader, desc="Training", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            torch.cuda.empty_cache()
            uv_imgs, coord_imgs, masks, shape_conditions, points = data
            (
                sample_points,
                loss_sampler,
                loss_proj,
                loss_sim
            ) = self.compute_samplenet_loss(sampler, points.cuda())

            # sampler.eval()
            # (sample_points, _, _, _) = self.compute_samplenet_loss(sampler, points.cuda())
            # sampler.train()

            uv_imgs = uv_imgs.cuda()
            masks = masks.cuda()
            data = (uv_imgs, coord_imgs.cuda(), masks, sample_points.cuda())
            # model
            color_embeddings = vae.get_emb(uv_imgs)
            _, pred_imgs, loss1, loss2, loss3 = model(color_embeddings, shape_conditions.cuda(), data)
            
            pred_embeddings = vae.get_emb(pred_imgs)
            loss_emb = mse_loss(color_embeddings, pred_embeddings).cpu()

            loss_color, loss_sample = self.compute_total_loss(loss1, loss2, loss3, loss_emb, loss_sampler)

            progress_bar.set_postfix(l1=loss1.item() * SCALE, l_sampler=loss_sampler.item() * SCALE, l_color=loss_color.item() * SCALE, l_sam=loss_sample.item() * SCALE)

            # update
            if self.args.alternative > 0:
                if epoch > 10 and epoch % self.args.alternative * 2 >= self.args.alternative:
                    loss_color.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    loss_sample.backward()
                    optimizer_sam.step()
                    optimizer_sam.zero_grad()
            else:
                optimizer.zero_grad()
                loss_color.backward(retain_graph=True)
                optimizer.step()
                optimizer_sam.zero_grad()
                loss_sample.backward()
                optimizer_sam.step()
                # for name, param in sampler.named_parameters():
                #     if param.requires_grad:
                #         print(f"Layer: {name}")
                #         print(f"Weights: {param.data}")
                #         print(f"Gradients: {param.grad}")

            total_loss_color += loss_color.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss_emb += loss_emb.item()
            total_loss_sample += loss_sample.item() * 100
            total_loss_sampler += loss_sampler.item() * 100
            total_loss_proj += loss_proj.item()
            total_loss_sim += loss_sim.item()

        avg_loss_color  = float(total_loss_color) / batch_num * SCALE
        avg_loss1       = float(total_loss1) / batch_num * SCALE
        avg_loss2       = float(total_loss2) / batch_num * SCALE
        avg_loss3       = float(total_loss3) / batch_num * SCALE
        avg_loss_emb    = float(total_loss_emb) / batch_num * SCALE
        avg_loss_sample = float(total_loss_sample) / batch_num * SCALE
        avg_loss_sampler = float(total_loss_sampler) / batch_num * SCALE
        avg_loss_proj   = float(total_loss_proj) / batch_num
        avg_loss_sim    = float(total_loss_sim) / batch_num

        print(f"TRAIN {epoch} === Color {avg_loss_color}, Sample {avg_loss_sample}, l1 {avg_loss1}, l_emb {avg_loss_emb}, l_sam {avg_loss_sampler}, l_proj {avg_loss_proj}, l_sim {avg_loss_sim}.")
        return avg_loss_color, avg_loss_sample, avg_loss1, avg_loss2, avg_loss3, avg_loss_emb, avg_loss_sampler, avg_loss_proj, avg_loss_sim


    def eval_sample(self, model, vae, testloader, epoch, sampler):
        total_loss_color = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        total_loss_emb = 0.0
        total_loss_sample = 0.0
        total_loss_sampler = 0.0
        batch_num = len(testloader)
        save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
        if epoch == 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(save_path + "ori")
            os.makedirs(save_path + "train")
            os.makedirs(save_path + "eval")


        model.eval()
        sampler.eval()

        progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            torch.cuda.empty_cache()
            uv_imgs, coord_imgs, masks, shape_conditions, points = data
            (
                sample_points,
                loss_sampler,
                _, _
            ) = self.compute_samplenet_loss(sampler, points.cuda())
            loss_sampler = loss_sampler.cpu()

            uv_imgs = uv_imgs.cuda()
            masks = masks.cuda()
            data = (uv_imgs, coord_imgs.cuda(), masks, sample_points.cuda())
            # model
            color_embeddings = vae.get_emb(uv_imgs)
            pred_colors, pred_imgs, loss1, loss2, loss3 = model(color_embeddings, shape_conditions.cuda(), data)
            pred_embeddings = vae.get_emb(pred_imgs)
            loss_emb = mse_loss(color_embeddings, pred_embeddings).cpu()

            loss_color, loss_sample = self.compute_total_loss(loss1, loss2, loss3, loss_emb, loss_sampler)
            
            if i == 0 and (epoch % 10 == 0 or epoch < 5):
                sample_coords = find_uv_from_pos(sample_points.detach().cpu().numpy(), coord_imgs, self.PIXEL_NUM)
                save_imgs(pred_imgs, self.args.batch_size, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'/eval/eval_{epoch}', False)
                np.savez(f"/homes/yz723/Project/ColorPointTextureNet/output/pred_32_{epoch}.npz", points=sample_points[0].cpu().detach().numpy(), colors=pred_colors[0].cpu().detach().numpy())
                out_path = f"/homes/yz723/Project/ColorPointTextureNet/output/pred_32_{epoch}.ply"
                trimesh.Trimesh(vertices=sample_points[9].cpu().detach().numpy(), process=False).export(out_path)
                    
            total_loss_color += loss_color.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss_emb += loss_emb.item()
            total_loss_sample += loss_sample.item() * 100
            total_loss_sampler += loss_sampler.item() * 100

        avg_loss_color = float(total_loss_color) / batch_num * SCALE
        avg_loss1 = float(total_loss1) / batch_num * SCALE
        avg_loss2 = float(total_loss2) / batch_num * SCALE
        avg_loss3 = float(total_loss3) / batch_num * SCALE
        avg_loss_emb = float(total_loss_emb) / batch_num * SCALE
        avg_loss_sample = float(total_loss_sample) / batch_num * SCALE
        avg_loss_sampler = float(total_loss_sampler) / batch_num * SCALE
        
        sampler.train()
        model.train()

        print(f"EVAL {epoch} === Color {avg_loss_color}, Sample {avg_loss_sample}, l1 {avg_loss1}, l_emb {avg_loss_emb}, l_sam {avg_loss_sampler}.")
        return avg_loss_color, avg_loss_sample, avg_loss1, avg_loss2, avg_loss3, avg_loss_emb, avg_loss_sampler


    def test_main(self, model, vae, testloader):
        batch_num = len(testloader)
        save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.eval()
        progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
        save_index = 9
        save_list = [2, 7, 8, 10, 11, 12, 18, 22, 24, 29]
        for i, data in enumerate(progress_bar):
            if i > 30: break
            uv_imgs, coord_imgs, masks, shape_conditions, sample_points, sample_colors, sample_coords = data
            BATCH_SIZE = uv_imgs.shape[0]

            uv_imgs = uv_imgs.cuda()
            masks = masks.cuda()
            data = (coord_imgs.cuda(), masks, sample_points.cuda())
            # model
            if self.args.gen_main:
                rand_imgs, color_embeddings = vae.get_random_result(BATCH_SIZE)
            elif self.args.test_main:
                color_embeddings = vae.get_emb(uv_imgs)
            pred_colors, pred_imgs, pred_imgs_mask = model.predict(color_embeddings, shape_conditions.cuda(), data)
            
            # solve segmentation fault
            pred_colors = pred_colors.cpu().detach().numpy()
            pred_colors = torch.from_numpy(pred_colors).cuda()
            if i in save_list:
                # save_imgs(uv_imgs, BATCH_SIZE, sample_coords, sample_colors, masks, self.PIXEL_NUM, save_path + f'/ori', True)
                # save_imgs(pred_imgs, BATCH_SIZE, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'/pred', False)
                # save_imgs(pred_imgs_mask, BATCH_SIZE, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'/pred', False)
                # save_imgs(pred_imgs_mask, BATCH_SIZE, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'/pred2', False, True)
                save_img_results(pred_imgs_mask[0], sample_coords[0], sample_colors[0], pred_colors[0], masks[0], self.PIXEL_NUM, save_path, i)
                
                # save_img(rand_imgs, save_path + f'/rand')
                # out_path = f"/homes/yz723/Project/ColorPointTextureNet/output/"
                np.savez(f"{save_path}/{i}_pred.npz", points=sample_points[0].cpu().detach().numpy(), colors=pred_colors[0].cpu().detach().numpy())
                # np.savez(f"{save_path}/ori.npz", points=sample_points[0].cpu().detach().numpy(), colors=sample_colors[0].cpu().detach().numpy())
                # trimesh.Trimesh(vertices=sample_points[0].cpu().detach().numpy(), process=False).export(f'{save_path}/pred.ply')
        print("TEST FINISH")


    def test_sample(self, model, vae, sampler, testloader):
        batch_num = len(testloader)
        save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.eval()
        sampler.eval()
        # sampler.train()
        progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
        save_index = 9
        save_list = [2, 7, 8, 10, 11, 12, 18, 22, 24, 29]
        for i, data in enumerate(progress_bar):
            if i > 30: break
            uv_imgs, coord_imgs, masks, shape_conditions, points = data
            (
                sample_points,
                loss_sampler,
                _, _
            ) = self.compute_samplenet_loss(sampler, points.cuda())
            sample_coords = find_uv_from_pos(sample_points.detach().cpu().numpy(), coord_imgs, self.PIXEL_NUM)
            BATCH_SIZE = uv_imgs.shape[0]

            uv_imgs = uv_imgs.cuda()
            masks = masks.cuda()
            data = (coord_imgs.cuda(), masks, sample_points.cuda())
            # model
            if self.args.gen_main:
                rand_imgs, color_embeddings = vae.get_random_result(BATCH_SIZE)
            elif self.args.test_main:
                color_embeddings = vae.get_emb(uv_imgs)
            pred_colors, pred_imgs, pred_imgs_mask = model.predict(color_embeddings, shape_conditions.cuda(), data)
            
            sample_colors = get_all_colors(sample_coords, uv_imgs)
            # 解决segmentation fault问题
            pred_colors = pred_colors.cpu().detach().numpy()
            pred_colors = torch.from_numpy(pred_colors).cuda()
            if i in save_list:
                save_img_results(pred_imgs_mask[0], sample_coords[0], sample_colors[0], pred_colors[0], masks[0], self.PIXEL_NUM, save_path, i)
                np.savez(f"{save_path}/{i}_pred.npz", points=sample_points[0].cpu().detach().numpy(), colors=pred_colors[0].cpu().detach().numpy())
                # np.savez(f"{save_path}/ori.npz", points=sample_points[0].cpu().detach().numpy(), colors=sample_colors[0].cpu().detach().numpy())
                # trimesh.Trimesh(vertices=sample_points[0].cpu().detach().numpy(), process=False).export(f'{save_path}/pred.ply')
        print("TEST FINISH")


    def compute_total_loss(self, loss1, loss2, loss3, loss_emb, loss_sampler):
        loss_color = self.args.alpha * loss1 + self.args.beta * loss2 + 2 * self.args.beta * loss3 + 0.02 * loss_emb
        if self.SAMPLER == 'samplenet':
            # loss_sample = 0.1 * loss_color + 5 * loss_sampler
            loss_sample = loss_color
        else:
            loss_sample = 0
        return loss_color, loss_sample


    def compute_samplenet_loss(self, sampler, points, GAMMA=1, DELTA=0, ALPHA=0.01, LMBDA=0.01):
        """Sample point clouds using SampleNet and compute sampling associated losses."""
        points = points.float()
        p_simplified, sample_points = sampler(points)

        # Sampling loss
        simplification_loss = sampler.get_simplification_loss(
            points, p_simplified, self.NUM_OUT_POINTS, GAMMA, DELTA
        )

        # Projection loss
        projection_loss = sampler.get_projection_loss()
        samplenet_loss = ALPHA * simplification_loss + LMBDA * projection_loss

        sample_points = sample_points.reshape(sample_points.shape[0], -1, 3)

        return sample_points, samplenet_loss, projection_loss, simplification_loss


def save_img(imgs, save_path):
    img_list = []
    row_img_list = []
    for i in range(imgs.shape[0]):
        img = imgs[i].cpu().detach().numpy()
        img[:, :, [0, 2]] = img[:, :, [2, 0]]
        img_list.append(img)
        if (i % 5 == 4):
            row_img = np.concatenate(img_list, axis=1)
            row_img_list.append(row_img)
            img_list = []
            
    img = np.concatenate(row_img_list, axis=0)
    plt.imsave(f'{save_path}_img.png', img)


def save_imgs(imgs, batch_size, sample_coords, pred_colors, masks, pixel_num, save_path, save_mask, use_mask=False):
    img_list = []
    tex_list = []
    row_img_list = []
    row_tex_list = []
    mask_list = []
    row_mask_list = []
    for i in range(batch_size):
        img = imgs[i].cpu().detach().numpy()
        img[:, :, [0, 2]] = img[:, :, [2, 0]]
        img_list.append(img)
        tex = create_sample_tex_img(sample_coords[i], pred_colors[i], masks[i], pixel_num, 3, use_mask).cpu().detach().numpy()
        tex[:, :, [0, 2]] = tex[:, :, [2, 0]]
        tex_list.append(tex)
        if save_mask:
            mask_list.append(masks[i].cpu().detach().numpy())
        if (i % 5 == 4):
            row_img = np.concatenate(img_list, axis=1)
            row_img_list.append(row_img)
            img_list = []
            row_tex = np.concatenate(tex_list, axis=1)
            row_tex_list.append(row_tex)
            tex_list = []
            if save_mask:
                row_mask = np.concatenate(mask_list, axis=1)
                row_mask_list.append(row_mask)
                mask_list = []
            
    if batch_size > 1:
        img = np.concatenate(row_img_list, axis=0)
        tex = np.concatenate(row_tex_list, axis=0)
        if save_mask:
            mask = np.concatenate(row_mask_list, axis=0)
    else:
        img = img_list[0]
        tex = tex_list[0]
        if save_mask:
            mask = mask_list[0]
    plt.imsave(f'{save_path}_img.png', img)
    plt.imsave(f'{save_path}_tex.png', tex)
    if save_mask:
        plt.imsave(f'{save_path}_mask.png', mask)


def save_img_results(img, sample_coord, ori_colors, pred_color, mask, pixel_num, save_path, index=0):
    r = 3

    img = img.cpu().detach().numpy()
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    plt.imsave(f'{save_path}/{index}_img.png', img)

    # tex_ori = create_sample_tex_img(sample_coord, ori_colors, mask, pixel_num, r, True).cpu().detach().numpy()
    # tex_ori[:, :, [0, 2]] = tex_ori[:, :, [2, 0]]
    # plt.imsave(f'{save_path}/tex_ori.png', tex_ori)

    tex = create_sample_tex_img(sample_coord, pred_color, mask, pixel_num, r, False).cpu().detach().numpy()
    tex[:, :, [0, 2]] = tex[:, :, [2, 0]]
    plt.imsave(f'{save_path}/{index}_tex.png', tex)

    # tex2 = create_sample_tex_img(sample_coord, pred_color, mask, pixel_num, r, True).cpu().detach().numpy()
    # tex2[:, :, [0, 2]] = tex2[:, :, [2, 0]]
    # plt.imsave(f'{save_path}/tex2.png', tex2)

    r = 2
    img2 = img
    color = torch.tensor([1,1,1])
    for coord in sample_coord:
        u = int(coord[0] * (pixel_num - 1))
        v = int((1 - coord[1]) * (pixel_num - 1))
        for ui in range(r):
            for vi in range(r):
                img2[max(0,v-vi), max(0, u-ui)] = color
                img2[max(0,v-vi), min(pixel_num-1, u+ui)] = color
                img2[min(pixel_num-1,v+vi), min(pixel_num-1, u+ui)] = color
                img2[min(pixel_num-1,v+vi), max(0, u-ui)] = color
    plt.imsave(f'{save_path}/{index}_img2.png', img2)
>>>>>>> Stashed changes
