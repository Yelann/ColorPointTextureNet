import argparse
import logging
import os
import os.path
import sys
import tqdm
import torch
from dataset.dataset_loader import ShapeNetDataset

from action import Action
from torch.utils.tensorboard import SummaryWriter

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
torch.manual_seed(0)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

def options(argv=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # setting for data and load
    parser.add_argument('-o', '--outfile', default='/mnt/d/Project/ColorPointTextureNet/output_model/', type=str,
                        metavar='BASENAME', help='output filename (prefix)')  # the result: ${BASENAME}_model_best.pth
    parser.add_argument('--datafolder', default='/mnt/d/Project/Dataset/PointUVDiff/', type=str, help='dataset folder')

    # For testing
    parser.add_argument('--test', action='store_true', help='Perform testing routine. Otherwise, the script will train.')


    parser.add_argument('-in', '--num_in_points', default=1024, type=int, metavar='N')
    parser.add_argument('-out', '--num_out_points', default=256, type=int, metavar='N')
    parser.add_argument('-emb', '--num_emb_dim', default=16, type=int, metavar='N')
    parser.add_argument('--sampler', default='fps', choices=['fps', 'random', 'geo'])

    # settings for on training
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSProp'], metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default = 5e-3) 
    
    args = parser.parse_args(argv)
    return args


def get_datasets(args):
    trainset = ShapeNetDataset(
        root=args.datafolder,
        spoints=args.num_out_points,
        split='train')

    testset = ShapeNetDataset(
        root=args.datafolder,
        spoints=args.num_out_points,
        split='test')

    return trainset, testset



def train(args, trainset, testset, action):
    # dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # create model
    model = action.create_model()
    print("00")
    print(model.parameters())
    # Optimizer
    min_loss = float("inf")
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params, lr=args.lr)
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(learnable_params, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr, momentum=0.9)

    # training
    LOGGER.debug("===TRAIN START===")
    writer = SummaryWriter('runs/v1')
    for epoch in range(args.start_epoch, args.epochs):
        print("1")
        train_loss = action.train_1(
            model, trainloader, optimizer, epoch
        )
        print("2")
        val_loss = action.eval_1(
            model, testloader, epoch
        )
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('eval_loss', val_loss, epoch)
        is_best = val_loss < min_loss

        if is_best:
            save_checkpoint(model.state_dict(), args.outfile, f"model_best_{epoch}")

        min_loss = min(val_loss, min_loss)

    LOGGER.debug("===TRAIN FINISH==")


def save_checkpoint(state, filename, suffix):
    torch.save(state, "{}_{}.pth".format(filename, suffix))


def main(args):
    trainset, testset = get_datasets(args)
    action = Action(args)

    train(args, trainset, testset, action)


if __name__ == "__main__":
    ARGS = options()
    _ = main(ARGS)