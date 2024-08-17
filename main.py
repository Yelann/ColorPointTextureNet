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
from checkpoint import save_checkpoint, load_checkpoint, load_vae, load_model


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
    parser.add_argument('--sampler', default='fps', choices=['fps', 'uniform', 'geodesic'])

    # settings for on training
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-p', '--pixel-num', default=128, type=int, metavar='N')
    parser.add_argument('-w', '--writer-path', default='0726_5')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSProp'], metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default = 1e-3) 
    parser.add_argument('--alpha', type=float, default = 2) 
    parser.add_argument('--beta', type=float, default = 1) 

    parser.add_argument('--train-main', action='store_true', help='Allow Main Training.')
    parser.add_argument('--train-vae', action='store_true', help='Allow VAE Training.')
    parser.add_argument('--train-both', action='store_true', help='Allow Both Training.')
    parser.add_argument('--test', action='store_true', help='Allow Testing.')
    parser.add_argument('--model-path', default='vae_0.0002_128_48_64.pth', type=str)
    parser.add_argument('--check-path', default='vae_0.0002_128_48_64.pth', type=str)
    
    args = parser.parse_args(argv)
    return args


def get_datasets(args):
    if args.train_main or args.train_both or args.test:
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
    model = action.create_model()
    print(model.parameters())

    # Optimizer
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params, lr=args.lr)
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(learnable_params, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr, momentum=0.9)

    if args.start_epoch > 0:
        load_dir = args.outfile + "/output_model/main/"
        load_checkpoint(model, optimizer, load_dir + args.check_path)

    writer_path = f'runs/{args.writer_path}_{args.pixel_num}'
    writer = SummaryWriter(writer_path)

    LOGGER.debug("===TRAIN START===")
    if args.train_vae:
        train_vae(args, model, trainloader, testloader, optimizer, action, writer)
    elif args.train_main:
        train_main(args, model, trainloader, testloader, optimizer, action, writer)
    LOGGER.debug("===TRAIN FINISH==")

    writer.close()


def train_main(args, model, trainloader, testloader, optimizer, action, writer):
    min_loss = float("inf")
    save_dir = args.outfile + f"/output_model/main/{args.pixel_num}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_l1, train_l2, train_l3 = action.train_1(model, trainloader, optimizer, epoch)
        val_loss, val_l1, val_l2, val_l3        = action.eval_1(model, testloader, epoch)

        writer.add_scalar('train_loss_l1',      train_l1,   epoch)
        writer.add_scalar('eval_loss_l1',       val_l1,     epoch)
        writer.add_scalar('train_loss_mean',    train_l2,   epoch)
        writer.add_scalar('eval_loss_mean',     val_l2,     epoch)
        writer.add_scalar('train_loss_var',     train_l3,   epoch)
        writer.add_scalar('eval_loss_var',      val_l3,     epoch)
        writer.add_scalar('train_loss',         train_loss, epoch)
        writer.add_scalar('eval_loss',          val_loss,   epoch)
        
        save_checkpoint(model, optimizer, save_dir + f"model_{args.writer_path}_last.pth")
        if val_l1 < min_loss:
            save_checkpoint(model, optimizer, save_dir + f"model_{args.writer_path}_{epoch}.pth")

        min_loss = min(val_l1, min_loss)


def train_vae(args, model, trainloader, testloader, optimizer, action, writer):
    min_loss = float("inf")
    save_dir = args.outfile + f"/output_model/vae/{args.pixel_num}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, _, _, _ = action.train_1(model, trainloader, optimizer, epoch)
        val_loss, _, _, _   = action.eval_1(model, testloader, epoch)

        writer.add_scalar('train_loss_vae', train_loss, epoch)
        writer.add_scalar('eval_loss_vae',  val_loss,   epoch)
        
        save_checkpoint(model, optimizer, save_dir + f"model_{args.writer_path}_last.pth")
        if val_loss < min_loss:
            save_checkpoint(model, optimizer, save_dir + f"model_{args.writer_path}_{epoch}.pth")
        min_loss = min(val_loss, min_loss)


def test(args, testset, action):
    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # create model
    model = action.create_model()

    load_dir = args.outfile + f"/output_model/main/{args.pixel_num}/"
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)
    load_model(model, load_dir + args.model_path)        
    test_main(args, model, testloader, action)
        

def test_main(args, model, testloader, action):
    save_dir = args.outfile + "/output_model/main/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    val_loss, val_l1, val_l2, val_l3 = action.test_1(model, testloader)
    print(f"VAL Loss {val_loss}, l1 {val_l1}, l2 {val_l2}, l3 {val_l3}")




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