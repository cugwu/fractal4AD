import os
import sys

# # Define the path to your source directory
# src_path = os.path.abspath("/home/cugwu/Documents/codes/fractal4AD/src")
# # Add the source directory to the beginning of sys.path
# sys.path.insert(0, src_path)

import time
import argparse
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

from models import *
from utils import accuracy, AverageMeter, ProgressMeter, Summary, save_checkpoint, MultiFeatureDataset

# Parse input arguments
parser = argparse.ArgumentParser(description='Fractals training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# loading
parser.add_argument('--dataset_path', default='./fractals_data', type=str,
                    help='path to dataset or metadata.csv')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--root_model', type=str, default='checkpoint')
# training
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training')
parser.add_argument("--val_split", default=0.0, type = float, help="Validation split, set to 0.0 or 1.0 if you want training set equal to validation")
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--patience', type=int, default=10, help='time window where over-fitting is accepted')
parser.add_argument('--base_lr', type=float, default=0.1, help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument("--weight_decay", default=1e-4, type = float, help="weight decay")
parser.add_argument('--warmup_epochs', type=float, default=10, help='number of warmup epochs')
parser.add_argument('--print_freq', type=float, default=10, help='print result every N batch')
parser.add_argument('--target_loss', type=float, default=1e-4, help='target loss to control over-fitting')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 8)')
# model & data
parser.add_argument('--num_class', type=int, default=1000)
parser.add_argument('--per_class', type=int, default=1000)
parser.add_argument('--resnet_type', default='resnet50', type=str)
parser.add_argument('--n_max', type=int, default=5, help='max number of systems')
parser.add_argument('--crop_size', type=int, default=224, help='size of the image')
parser.add_argument('--grayscale', action='store_true', help='use grayscale images for training')
# ddp
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs in each node')
args = parser.parse_args()

start_epoch = 0
best_acc = 0

def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.num_gpus > torch.cuda.device_count():
        args.num_gpus = torch.cuda.device_count()

    args.world_size = args.num_gpus

    args.distributed = args.world_size > 1

    if args.distributed:
        print(f"Distributed training. Number of GPUs use: {args.world_size}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9950'
        torch.multiprocessing.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        print(f"Single GPU training")
        main_worker(0 , args)


def main_worker(rank, args):
    global start_epoch
    global best_acc

    if args.distributed:
        # Set the default GPU for each process
        torch.cuda.set_device(rank)
        # Initialize group process
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)

    # Training settings
    normalize = torchvision.transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])

    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop((args.crop_size,args.crop_size)),
                                                      torchvision.transforms.ToTensor(), normalize])
    train_set = MultiFeatureDataset(args.dataset_path, transforms=train_transform, use_gray=args.grayscale,
                                    val_split=args.val_split, train_mode=True)

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((args.crop_size, args.crop_size)),
                                                     torchvision.transforms.ToTensor(), normalize])
    test_set =  MultiFeatureDataset(args.dataset_path, transforms=test_transform, use_gray=args.grayscale,
                                    val_split=args.val_split, train_mode=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size,
                                                                        rank=rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=args.world_size,
                                                                       rank=rank)
        args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               drop_last=True, sampler=train_sampler, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                              sampler=test_sampler, num_workers=args.workers)

    # Model creation
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    model = globals()[args.resnet_type](pretrained=False, num_classes=args.num_class).to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Wrap model with DistributedDataParallel
        model = DDP(model, device_ids=[rank])


    # Define loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Define the SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1) # meaning no scheduler just to check which is better

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(rank)
                checkpoint = torch.load(args.resume, map_location=loc)
            else:
                checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizers'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    tf_writer = None
    filename = os.path.join(args.root_log, args.store_name)

    if rank == 0:
        os.makedirs(filename, exist_ok=True)
        with open(os.path.join(filename, 'args.txt'), 'w') as f: f.write(str(args))
        tf_writer = SummaryWriter(log_dir=filename)

    if args.evaluate:
        _ = test(model, test_loader, start_epoch, loss_fn, device, rank, tf_writer, args)
        return

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(model, optimizer, train_loader, epoch, loss_fn, device, rank, tf_writer)
        torch.cuda.empty_cache()

        acc1 = test(model, test_loader, epoch, loss_fn, device, rank, tf_writer, args)
        torch.cuda.empty_cache()

        scheduler.step()

        # Remember the best loss and save checkpoint
        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)

        if epoch % 1 == 0 and rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.resnet_type,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizers': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, args)

        if epoch % args.print_freq == 0 and rank == 0:
            # to check over-fitting during the training process
            state = {
                'epoch': epoch + 1,
                'arch': args.resnet_type,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizers': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            filename = os.path.join(args.root_model, args.store_name, f"checkpoint_{epoch}_{acc1:.2f}.pth.tar")
            torch.save(state, filename)

    print(args.resnet_type, 'best accuracy:', best_acc)
    if args.distributed:
        dist.destroy_process_group()


def train(model, optimizer, train_loader, epoch, loss_fn, device, rank, tf_writer):
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Loading', ':6.3f')
    losses = AverageMeter('Training Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        labels = labels.to(device)
        images = images.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        acc1 = accuracy(outputs, labels, topk=(1,))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            progress.display(i + 1)
            tf_writer.add_scalar('loss/train', losses.avg, epoch * len(train_loader) + i)
            tf_writer.add_scalar('acc/train_top1', top1.avg, epoch * len(train_loader) + i)

    return


def test(model, test_loader, epoch, loss_fn, device, rank, tf_writer, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, labels) in enumerate(loader):
                i = base_progress + i

                labels = labels.to(device)
                images = images.to(device)

                outputs = model(images)
                # loss
                loss = loss_fn(outputs, labels)
                acc1 = accuracy(outputs, labels, topk=(1,))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))

                batch_time.update(time.time() - end, images.size(0))
                end = time.time()

                if i % args.print_freq == 0 and rank == 0:
                    progress.display(i + 1)

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)


    batch_time = AverageMeter('Images/sec', ':6.3f', Summary.NONE)
    losses = AverageMeter('Validation Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(test_loader) + (args.distributed and (len(test_loader.sampler) * args.world_size < len(test_loader.dataset))),
        [batch_time, losses, top1],
        prefix='Test: ')

    model.eval()
    run_validate(test_loader)
    if args.distributed:
        losses.all_reduce()

    if args.distributed and (len(test_loader.sampler) * args.world_size < len(test_loader.dataset)):
        aux_val_dataset = torch.utils.data.Subset(test_loader.dataset, range(len(test_loader.sampler) * args.world_size, len(test_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(aux_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(test_loader))

    return top1.avg


if __name__ == '__main__':
    main()
