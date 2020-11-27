import time

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder
from utils import AverageMeter, ProgressMeter, accuracy, save_ckpt, load_ckpt
import utils
import numpy as np
from rexnet import Model

import argparse
import os


parser = argparse.ArgumentParser(description='ReXNet')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset: ')
parser.add_argument('--datapath', default='../data', type=str,
                    help='where you want to load/save your dataset? (default: ../data)')
parser.add_argument('--savepath', default='./checkpoint/', type=str,
                    help='where you want to load/save checkpoint?')
parser.add_argument('--num_workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=0.1, type=float,
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--nesterov', default=True, type=bool,
                    help='use nesterov momentum?')
parser.add_argument('--scheduler', default='multistep', type=str, help='scheduler: ')
parser.add_argument('--step_size', default=30,
                    type=int, metavar='STEP',
                    help='period of learning rate decay / '
                        'maximum number of iterations for '
                        'cosine annealing scheduler (default: 30)')
parser.add_argument('--milestones', default=[100,150], type=int, nargs='+',
                    help='list of epoch indices for multi step scheduler '
                        '(must be increasing) (default: 100 150)')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='multiplicative factor of learning rate decay (default: 0.1)')
parser.add_argument('--print_freq', default=100, type=int)
parser.add_argument('--resume', action='store_true', help='resume?')
parser.add_argument('--beta', default=1.0, type=float, help='cutmix beta')
parser.add_argument('--cutmix_prob', default=0.0, type=float,help='cutmix probability')

args = parser.parse_args()


def main(args):
    model = Model()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    if args.scheduler=='multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones,
                                                         gamma=args.gamma)
    elif args.scheduler=='cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.step_ssize)
    criterion = torch.nn.CrossEntropyLoss()


    model = model.cuda()
    criterion = criterion.cuda()

    start_epoch = 0

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    if not os.path.exists('{}'.format(args.savepath)):
        os.makedirs('{}'.format(args.savepath))

    # resume
    if args.resume:
      model, optimizer, start_epoch = load_ckpt(model, optimizer, args)

    
    # Dataloader
    if args.dataset=='cifar10':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        trainset = CIFAR10(
            root=args.datapath, train=True, download=True,
            transform=transform_train)
        valset = CIFAR10(
            root=args.datapath, train=False, download=True,
            transform=transform_val)
    elif args.dataset=='cifar100':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        trainset = CIFAR100(
            root=args.datapath, train=True, download=True,
            transform=transform_train)
        valset = CIFAR100(
            root=args.datapath, train=False, download=True,
            transform=transform_val)
    elif args.dataset=='ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_val = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = ImageNet(
            root=args.datapath, split='train', download=False,
            transform=transform_train)
        valset = ImageNet(
            root=args.datapath, split='val', download=False,
            transform=transform_val)
    elif args.dataeset=='tiny-imagenet-200':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_val = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = ImageFolder(
            root=args.datapath, split='train', download=False,
            transform=transform_train)
        valset = ImageFolder(
            root=args.datapath, split='val', download=False,
            transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)


    # start training
    last_top1_acc = 0
    acc1_valid = 0
    best_acc1 = 0
    is_best = False
    for epoch in range(start_epoch, args.epochs):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        last_top1_acc = train(train_loader, epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))

        # validate for one epoch
        start_time = time.time()
        acc1_valid = validate(val_loader, model, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to validate this epoch\n'.format(
            elapsed_time))


        # learning rate scheduling
        scheduler.step()

        is_best = acc1_valid > best_acc1
        best_acc1 = max(acc1_valid, best_acc1)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        save_ckpt(checkpoint, is_best, args)

        #if is_best:
        #  torch.save(model.state_dict(), args.savepath+'model_weight_best.pth')

        # Save model each epoch
        #torch.save(model.state_dict(), args.savepath+'model_weight_epoch{}.pth'.format(epoch))

    print(f"Last Top-1 Accuracy: {last_top1_acc}")
    print(f"Number of parameters: {pytorch_total_params}")



def train(train_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
          # compute output
          output = model(input)
          loss = criterion(output, target)

        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    print('=> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), batch_time, losses, top1, top5, prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            total_loss += loss.item()

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()

        print(
            "====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )
        total_loss = total_loss / len(val_loader)

    return top1.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


if __name__ == "__main__":
    main(args)
