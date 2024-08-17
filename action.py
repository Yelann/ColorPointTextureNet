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
