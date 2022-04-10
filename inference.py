import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

from train_argument import parser, print_args

from time import time
from utils import *
from models import *
from trainer import *


def main(args):
    save_folder = args.affix
    data_dir = args.data_root

    # log_folder = os.path.join(args.log_root, save_folder)
    log_folder = os.path.join(args.model_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder,"best_acc_model.pth")

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.model == "VGG16":
        net = vgg(dataset=args.dataset, depth=16)
        if args.mask:
            net = masked_vgg(dataset=args.dataset, depth=16)
    elif args.model == "WideResNet":
        net = WideResNet(depth=28, num_classes=args.dataset == 'cifar10' and 10 or 100, widen_factor=8)
        if args.mask:
            net = MaskedWideResNet(depth=28, num_classes=args.dataset == 'cifar10' and 10 or 100, widen_factor=8)

    net.load_state_dict(
            torch.load(args.model_folder, map_location=lambda storage, loc: storage))

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)
    net.to(device)

    trainer = Trainer(args, logger)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=100, shuffle=True, **kwargs)
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=100, shuffle=True, **kwargs)
    trainer.test(net, device, test_loader)


if __name__ == '__main__':
    args = parser()
    # print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
