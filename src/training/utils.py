import os
import numpy as np
import shutil
from PIL import Image
from enum import Enum
import torch
from torch.utils.data import random_split
import torch.distributed as dist
from torchvision.datasets import ImageFolder


class MultiFeatureDataset(ImageFolder):
    def __init__(
        self,
        data_path,
        transforms = None,
        use_gray = False,
        val_split = 0.0,
        train_mode = True
    ):
        super(MultiFeatureDataset, self).__init__(data_path, transforms)
        self.use_gray = use_gray
        if 0 < val_split < 1:
            self.do_split = True
            self.val_split = val_split
            self._split_dataset()  # Call the function to split the dataset
        else:
            self.do_split = False
        self.train_mode = train_mode

    def _split_dataset(self):
        """Splits the dataset into train and validation sets based on val_split."""
        dataset_size = len(self.samples)
        val_size = int(self.val_split * dataset_size)  # Validation size
        train_size = dataset_size - val_size  # Remaining for training

        # Randomly split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(self.samples, [train_size, val_size])

    def __len__(self):
        """Returns the length of the dataset depending on mode."""
        if self.do_split:
            return len(self.train_dataset) if self.train_mode else len(self.val_dataset)
        else:
            return len(self.samples)

    def __getitem__(self, index):
        # Fetch the data from the appropriate subset based on train_mode
        if self.do_split:
            if self.train_mode:
                img_path, target = self.train_dataset[index]  # Access via train_dataset
            else:
                img_path, target = self.val_dataset[index]    # Access via val_dataset
        else:
            img_path, target = self.samples[index]

        if self.use_gray:
            gray_image = Image.open(img_path).convert('L')
            image = Image.merge("RGB", (gray_image, gray_image, gray_image))
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, target


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """Warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * float(step) / (args.warmup_epochs * len_epoch)

    elif args.coslr:
        nmax = len_epoch * args.epochs
        lr = args.lr * 0.5 * (np.cos(step / nmax * np.pi) + 1)
    else:
        decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
        lr = args.lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, args):
    if not os.path.exists(os.path.join(args.root_model, args.store_name)):
        os.makedirs(os.path.join(args.root_model, args.store_name))
    filename = os.path.join(args.root_model, args.store_name, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))



