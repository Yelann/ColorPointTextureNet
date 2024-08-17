import torch
import torch.nn as nn
from torch.autograd import Variable

import os

# import tqdm
from tqdm import tqdm
from models.cptnet import CPTNet
from typing import Any, List
from utuils.point_sample_gather import pc_to_uv
import matplotlib.pyplot as plt
from utuils.sample_utuils import create_sample_tex_img

import numpy as np
# from models.cptnet import ColorEncoder
from models.color_vae import ColorVAE
from checkpoint import save_checkpoint, load_checkpoint, load_vae

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
        self.TRAIN_BOTH = args.train_both
        self.TEST = args.test
        self.COLOR_EMB_DIM = args.num_emb_dim

    def create_model(self):
        model = CPTNet(self.args)
        vae = ColorVAE(input_dim=3, embedding_dim=self.COLOR_EMB_DIM, pixel_num=self.PIXEL_NUM)
        

        if self.TRAIN_MAIN:
            model.requires_grad_(True)
            model.train()

            model.vae = vae
            if self.args.start_epoch == 0:
                load_dir = self.args.outfile + "/output_model/vae/"
                load_vae(model, load_dir + self.args.model_path)
            vae.requires_grad_(False)
            vae.eval()

        elif self.TRAIN_VAE:
            vae.requires_grad_(True)
            vae.train()
            model.requires_grad_(False)
            model.eval()
            model.vae = vae

        elif self.TRAIN_BOTH:
            model.requires_grad_(True)
            model.train()

            model.vae = vae
            if self.args.start_epoch == 0:
                load_dir = self.args.outfile + "/output_model/vae/"
                load_vae(model, load_dir + self.args.model_path)
            vae.requires_grad_(True)
            vae.train()

        elif self.TEST:
            vae.requires_grad_(False)
            vae.eval()
            model.requires_grad_(False)
            model.eval()
            model.vae = vae

        model.cuda()
        return model

    def train_1(self, model, trainloader, optimizer, epoch):
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        batch_num = len(trainloader)

        progress_bar = tqdm(trainloader, desc="Training", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            
            if self.TRAIN_VAE:
                uv_imgs, coord_imgs, normal_imgs, masks = data
                pred_uv_imgs, color_embeddings, loss = model.vae(uv_imgs.cuda(), masks.cuda())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.set_postfix(loss = loss.item())

                save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
                if i == 0 and epoch % 10 == 0:
                    if epoch == 0 and not os.path.exists(save_path):
                        os.makedirs(save_path)
                        os.makedirs(save_path + "vae_train")
                        os.makedirs(save_path + "vae_eval")
                    save_img(uv_imgs, save_path + f'vae_train/{epoch}_ori')
                    save_img(pred_uv_imgs, save_path + f'vae_train/{epoch}')
                
                total_loss += loss.item()

            if self.TRAIN_MAIN:
                uv_imgs, coord_imgs, normal_imgs, masks, shape_conditions, sample_points, sample_colors, sample_coords = data
                data = (uv_imgs.cuda(), coord_imgs.cuda(), normal_imgs.cuda(), masks.cuda(), sample_points.cuda())
                # model
                _, color_embeddings, _ = model.vae(uv_imgs.cuda(), masks.cuda())
                pred_colors, pred_imgs, loss1, loss2, loss3 = model(color_embeddings, shape_conditions.cuda(), data)
                _, pred_embeddings, _ = model.vae(pred_imgs, masks.cuda())
                loss_emb = 0.05 * mse_loss(color_embeddings, pred_embeddings)

                loss = self.args.alpha * loss1 + self.args.beta * loss2 + 2 * self.args.beta * loss3 + loss_emb
                l1 = self.args.alpha * loss1.item() * SCALE
                l2 = self.args.beta * loss2.item() * SCALE
                l3 = self.args.beta * loss3.item() * SCALE
                progress_bar.set_postfix(l1=l1, l2=l2, l3=l3, l4=loss_emb.item() * SCALE, loss=loss.item() * SCALE)

                # # update
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item()

                save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
                if i == 0 and epoch % 10 == 0:
                    if epoch == 0 and not os.path.exists(save_path):
                        os.makedirs(save_path)
                        os.makedirs(save_path + "ori")
                        os.makedirs(save_path + "train")
                        os.makedirs(save_path + "eval")
                    save_imgs(uv_imgs, self.args.batch_size, sample_coords, sample_colors, masks, self.PIXEL_NUM, save_path + f'/train/train_{epoch}_ori', True)
                    save_imgs(pred_imgs, self.args.batch_size, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'/train/train_{epoch}', False)


        if self.TRAIN_VAE:
            avg_loss = float(total_loss) / batch_num
            print(f"TRAIN {epoch} === Loss {avg_loss}.")
            return avg_loss, 0, 0, 0
        if self.TRAIN_MAIN:
            avg_loss = float(total_loss) / batch_num * SCALE
            avg_loss1 = float(total_loss1) / batch_num * SCALE
            avg_loss2 = float(total_loss2) / batch_num * SCALE
            avg_loss3 = float(total_loss3) / batch_num * SCALE
            print(f"TRAIN {epoch} === Loss {avg_loss}, l1 {avg_loss1}, l2 {avg_loss2}, l3 {avg_loss3}.")
            return avg_loss * SCALE, avg_loss1, avg_loss2, avg_loss3


    def eval_1(self, model, testloader, epoch):
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        batch_num = len(testloader)

        # print("=====Start Eval=====")
        progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):

            if self.TRAIN_VAE:
                uv_imgs, coord_imgs, normal_imgs, masks = data
                pred_uv_imgs, color_embeddings, loss = model.vae(uv_imgs.cuda(), masks.cuda())
                progress_bar.set_postfix(loss = loss.item())

                save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
                if i == 0 and epoch % 1 == 0:
                    save_img(uv_imgs, save_path + f'vae_eval/{epoch}_ori')
                    save_img(pred_uv_imgs, save_path + f'vae_eval/{epoch}')
                
                total_loss += loss.item()
            
            if self.TRAIN_MAIN:
                uv_imgs, coord_imgs, normal_imgs, masks, shape_conditions, sample_points, sample_colors, sample_coords = data
                data = (uv_imgs.cuda(), coord_imgs.cuda(), normal_imgs.cuda(), masks.cuda(), sample_points.cuda())
                # model
                _, color_embeddings, _ = model.vae(uv_imgs.cuda(), masks.cuda())
                pred_colors, pred_imgs, loss1, loss2, loss3 = model(color_embeddings, shape_conditions.cuda(), data)
                _, pred_embeddings, _ = model.vae(pred_imgs, masks.cuda())
                loss_emb = 0.05 * mse_loss(color_embeddings, pred_embeddings)

                loss = self.args.alpha * loss1 + self.args.beta * loss2 + 2 * self.args.beta * loss3 + loss_emb

                # ='/mnt/d/Project/ColorPointTextureNet'
                save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/"
                if i == 0 and (epoch % 10 == 0 or epoch < 5):
                    if epoch == 0:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                            os.makedirs(save_path + "ori")
                            os.makedirs(save_path + "train")
                            os.makedirs(save_path + "eval")
                        save_imgs(uv_imgs, self.args.batch_size, sample_coords, sample_colors, masks, self.PIXEL_NUM, save_path + '/ori/eval', True)
                    save_imgs(pred_imgs, self.args.batch_size, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'/eval/eval_{epoch}', False)
                    np.savez(f"/homes/yz723/Project/ColorPointTextureNet/output/pred_{epoch}.npz", points=sample_points[9].cpu().detach().numpy(), colors=pred_colors[9].cpu().detach().numpy())
                    
                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item()

        if self.TRAIN_VAE:
            avg_loss = float(total_loss) / batch_num
            print(f"EVAL === Loss {avg_loss}.")
            return avg_loss, 0, 0, 0
        if self.TRAIN_MAIN:
            avg_loss = float(total_loss) / batch_num * SCALE
            avg_loss1 = float(total_loss1) / batch_num * SCALE
            avg_loss2 = float(total_loss2) / batch_num * SCALE
            avg_loss3 = float(total_loss3) / batch_num * SCALE
            print(f"EVAL === Loss {avg_loss}, l1 {avg_loss1}, l2 {avg_loss2}, l3 {avg_loss3}.")
            return avg_loss, avg_loss1, avg_loss2, avg_loss3


    def test_1(self, model, testloader):
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        batch_num = len(testloader)

        progress_bar = tqdm(testloader, desc="Testing", ncols=100, ascii=True)
        for i, data in enumerate(progress_bar):
            uv_imgs, coord_imgs, normal_imgs, masks, shape_conditions, sample_points, sample_colors, sample_coords = data
            data = (uv_imgs.cuda(), coord_imgs.cuda(), normal_imgs.cuda(), masks.cuda(), sample_points.cuda())
            # model
            _, color_embeddings, _ = model.vae(uv_imgs.cuda(), masks.cuda())
            pred_colors, pred_imgs, loss1, loss2, loss3 = model(color_embeddings, shape_conditions.cuda(), data)
            _, pred_embeddings, _ = model.vae(pred_imgs, masks.cuda())
            loss_emb = 0.05 * mse_loss(color_embeddings, pred_embeddings)

            loss = self.args.alpha * loss1 + self.args.beta * loss2 + 2 * self.args.beta * loss3 + loss_emb

            save_path = f"{self.OUT_PATH}/output/{self.PIXEL_NUM}/{self.WRITER_PATH}_{self.NUM_OUT_POINTS}/test/"
            if i == 0:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_imgs(uv_imgs, self.args.batch_size, sample_coords, sample_colors, masks, self.PIXEL_NUM, save_path + 'ori', True)
                save_imgs(pred_imgs, self.args.batch_size, sample_coords, pred_colors, masks, self.PIXEL_NUM, save_path + f'pred', False)
                    
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()

        avg_loss = float(total_loss) / batch_num * SCALE
        avg_loss1 = float(total_loss1) / batch_num * SCALE
        avg_loss2 = float(total_loss2) / batch_num * SCALE
        avg_loss3 = float(total_loss3) / batch_num * SCALE
        return avg_loss, avg_loss1, avg_loss2, avg_loss3


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


def save_imgs(imgs, batch_size, sample_coords, pred_colors, masks, pixel_num, save_path, save_mask):
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
        tex = create_sample_tex_img(sample_coords[i], pred_colors[i], masks[i], pixel_num).cpu().detach().numpy()
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
            
    img = np.concatenate(row_img_list, axis=0)
    tex = np.concatenate(row_tex_list, axis=0)
    if save_mask:
        mask = np.concatenate(row_mask_list, axis=0)
    plt.imsave(f'{save_path}_img.png', img)
    plt.imsave(f'{save_path}_tex.png', tex)
    if save_mask:
        plt.imsave(f'{save_path}_mask.png', mask)