import argparse
import logging
import os
import os.path
import sys

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

import torch
from dataset.dataset_loader import ShapeNetDataset, ShapeNetVAEDataset

from action import Action
from torch.utils.tensorboard import SummaryWriter
from utuils.checkpoint import save_checkpoint, load_checkpoint, load_vae, load_model, save_vae, load_samplenet


torch.manual_seed(0)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

def options(argv=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # setting for data and load
    parser.add_argument('-o', '--outfile', default='/data2/ShapeNetCoreColor', type=str, metavar='BASENAME', help='output filename (prefix)')
    parser.add_argument('--datafolder', default='/data2/ShapeNetCoreColor', type=str, help='dataset folder')

    parser.add_argument('-in', '--num_in_points', default=50000, type=int, metavar='N')
    parser.add_argument('-out', '--num_out_points', default=256, type=int, metavar='N')
    parser.add_argument('-emb', '--num_emb_dim', default=48, type=int, metavar='N')
    parser.add_argument('-pos', '--num_pos_dim', default=16, type=int, metavar='N')
    parser.add_argument('-shape', '--num_shape_dim', default=16, type=int, metavar='N')
    parser.add_argument('--sampler', default='fps', choices=['fps', 'uniform', 'random', 'geodesic', 'samplenet'])

    # settings for on training
    parser.add_argument('-b', '--batch_size',   default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-w', '--writer_path',  default='0726_5')
    parser.add_argument('-p', '--pixel_num',    default=128, type=int, metavar='N')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSProp'], metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--lr',     type=float, default=5e-4) 
    parser.add_argument('--lr_sam', type=float, default=5e-4) 
    parser.add_argument('--alpha',  type=float, default=2) 
    parser.add_argument('--beta',   type=float, default=1) 

    parser.add_argument('--train_main', action='store_true', help='Allow Main Training.')
    parser.add_argument('--train_vae', action='store_true', help='Allow VAE Training.')
    parser.add_argument('--test_vae', action='store_true', help='Allow Testing.')
    parser.add_argument('--test_main', action='store_true', help='Allow Testing.')
    parser.add_argument('--gen_vae', action='store_true')
    parser.add_argument('--gen_main', action='store_true')
    parser.add_argument('--generation', action='store_true', help='Allow Random Generation.')
    parser.add_argument('--vae_path', default='vae_e48_h64_2e-4_best.pth', type=str)
    parser.add_argument('--model_path', default='', type=str)

    parser.add_argument('--alternative', default=0, type=int)
    
    args = parser.parse_args(argv)
    return args


def get_datasets(args):
    if args.train_main:
        trainset = ShapeNetDataset(
            root=args.datafolder,
            pixel_num=args.pixel_num,
            in_points=args.num_in_points,
            sample_points=args.num_out_points,
            sampler=args.sampler,
            shape_dim=args.num_shape_dim,
            split='train')

        testset = ShapeNetDataset(
            root=args.datafolder,
            pixel_num=args.pixel_num,
            in_points=args.num_in_points,
            sample_points=args.num_out_points,
            sampler=args.sampler,
            shape_dim=args.num_shape_dim,
            split='test')
        
    elif args.train_vae:
        trainset = ShapeNetVAEDataset(
            root=args.datafolder,
            pixel_num=args.pixel_num,
            sampler=args.sampler,
            split='train')

        testset = ShapeNetVAEDataset(
            root=args.datafolder,
            pixel_num=args.pixel_num,
            sampler=args.sampler,
            split='test')
        

    return trainset, testset



def train(args, trainset, testset, action):
    # dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # create model
    model, vae, sampler = action.create_model()
    print(model.parameters())

    # Optimizer
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())


    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params, lr=args.lr)
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(learnable_params, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr, momentum=0.9)
    
    if args.sampler == 'samplenet':
        learnable_params_sam = filter(lambda p: p.requires_grad, sampler.parameters())
        # optimizer_sam = torch.optim.Adam(learnable_params_sam, lr=args.lr_sam)
        optimizer_sam = torch.optim.SGD(learnable_params_sam, lr=args.lr_sam, momentum=0.9)
    else:
        optimizer_sam = 0

    if args.model_path != '':
        load_path = args.outfile + "/output_model/main/" + args.model_path
        if args.start_epoch == 0:
            load_checkpoint(model, vae, sampler, optimizer, optimizer_sam, load_path, load_optimizer=False)
        else:
            load_checkpoint(model, vae, sampler, optimizer, optimizer_sam, load_path, args.sampler=='samplenet')

    writer_path = f'runs/{args.writer_path}_{args.pixel_num}'
    writer = SummaryWriter(writer_path)

    LOGGER.debug("===TRAIN START===")
    if args.train_vae:
        learnable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(learnable_params, lr=args.lr)
        train_vae(args, vae, trainloader, testloader, optimizer, action, writer)
    elif args.train_main:
        train_main(args, model, vae, trainloader, testloader, optimizer, action, writer, sampler, optimizer_sam)
    LOGGER.debug("===TRAIN FINISH==")

    writer.close()


def train_main(args, model, vae, trainloader, testloader, optimizer, action, writer, sampler, optimizer_sam):
    min_loss = float("inf")
    save_dir = args.outfile + f"/output_model/main/{args.pixel_num}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(args.start_epoch, args.epochs):
        if args.sampler == 'samplenet':
            t_color, t_sample, t_l1, t_l2, t_l3, t_emb, t_sampler, t_proj, t_sim = action.train_sample(model, vae, trainloader, optimizer, epoch, sampler, optimizer_sam)
            e_color, e_sample, e_l1, e_l2, e_l3, e_emb, e_sampler = action.eval_sample(model, vae, testloader, epoch, sampler)
        else:
            t_color, t_sample, t_l1, t_l2, t_l3, t_emb, t_sampler, t_proj, t_sim = action.train_main(model, vae, trainloader, optimizer, epoch)
            e_color, e_sample, e_l1, e_l2, e_l3, e_emb, e_sampler = action.eval_main(model, vae, testloader, epoch, sampler)

        writer.add_scalar('train_loss_l1',      t_l1,       epoch)
        writer.add_scalar('eval_loss_l1',       e_l1,       epoch)
        writer.add_scalar('train_loss_mean',    t_l2,       epoch)
        writer.add_scalar('eval_loss_mean',     e_l2,       epoch)
        writer.add_scalar('train_loss_var',     t_l3,       epoch)
        writer.add_scalar('eval_loss_var',      e_l3,       epoch)
        writer.add_scalar('train_loss_emb',     t_emb,      epoch)
        writer.add_scalar('eval_loss_emb',      e_emb,      epoch)
        writer.add_scalar('train_loss_sampler', t_sampler,  epoch)
        writer.add_scalar('eval_loss_sampler',  e_sampler,  epoch)
        writer.add_scalar('train_loss_color',   t_color,    epoch)
        writer.add_scalar('eval_loss_color',    e_color,    epoch)
        writer.add_scalar('train_loss_sample',  t_sample,   epoch)
        writer.add_scalar('eval_loss_sample',   e_sample,   epoch)

        writer.add_scalar('train_proj',  t_proj,   epoch)
        writer.add_scalar('train_sim',  t_sim,   epoch)
        
        save_checkpoint(model, vae, sampler, optimizer, optimizer_sam, save_dir + f"model_{args.writer_path}_last.pth", args.sampler=='samplenet')
        if e_l1 < min_loss:
            save_checkpoint(model, vae, sampler, optimizer, optimizer_sam, save_dir + f"model_{args.writer_path}_{epoch}.pth", args.sampler=='samplenet')
        min_loss = min(e_l1, min_loss)


def train_vae(args, vae, trainloader, testloader, optimizer, action, writer):
    min_loss = float("inf")
    save_dir = args.outfile + f"/output_model/vae/{args.pixel_num}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = action.train_vae(vae, trainloader, optimizer, epoch)
        val_loss   = action.eval_vae(vae, testloader, epoch)

        writer.add_scalar('train_loss_vae', train_loss, epoch)
        writer.add_scalar('eval_loss_vae',  val_loss,   epoch)
        
        save_vae(vae, optimizer, save_dir + f"model_{args.writer_path}_last.pth")
        if val_loss < min_loss:
            save_vae(vae, optimizer, save_dir + f"model_{args.writer_path}_{epoch}.pth")
        min_loss = min(val_loss, min_loss)


def test(args, testset, action):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=args.workers
    )
    # create model
    model, vae, sampler = action.create_model()

    if args.test_vae:
        test_vae(args, vae, testloader, action)
    elif args.text_main:
        load_dir = args.outfile + f"/output_model/main/{args.pixel_num}/"
        load_model(model, vae, load_dir + args.model_path)
        if args.sampler == 'samplenet':
            load_samplenet(sampler, load_dir + args.model_path)
            test_sample(args, model, vae, sampler, testloader, action)
        else:
            test_main(args, model, vae, testloader, action)
        

def test_main(args, model, vae, testloader, action):
    # action.eval_main(model, vae, testloader, 0, 0)
    action.test_main(model, vae, testloader)


def test_vae(args, vae, testloader, action):
    action.test_vae(vae, testloader)


def test_sample(args, model, vae, sampler, testloader, action):
    action.test_sample(model, vae, sampler, testloader)


def main(args):
    trainset, testset = get_datasets(args)
    action = Action(args)
    if args.test:
        test(args, testset, action)
    else:
        train(args, trainset, testset, action)


if __name__ == "__main__":
    ARGS = options()
    _ = main(ARGS)
