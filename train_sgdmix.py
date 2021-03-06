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

import csv
import shutil
import pathlib
from copy import deepcopy
from os import remove
from os.path import isfile
from collections import OrderedDict


parser = argparse.ArgumentParser(description='ReXNet')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset: ')
parser.add_argument('--datapath', default='../data', type=str,
                    help='where you want to load/save your dataset? (default: ../data)')
parser.add_argument('--name', default='rexnetv1', type=str,
                    help='experience name')
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
parser.add_argument('--scheduler', default='step', type=str, help='scheduler: ')
parser.add_argument('--step_size', default=30,
                    type=int, metavar='STEP',
                    help='period of learning rate decay / '
                        'maximum number of iterations for '
                        'cosine annealing scheduler (default: 30)')
parser.add_argument('--milestones', default=[30,60,90], type=int, nargs='+',
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
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if args.scheduler=='multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones,
                                                         gamma=args.gamma)
    elif args.scheduler=='cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.step_size)

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
        if epoch==28:
            print("\n-----change optimizer-----")
            #optimizer = SGDP(model.parameters(), lr=args.learning_rate,
            #                    momentum=args.momentum, weight_decay=args.weight_decay,
            #                    nesterov=args.nesterov)
            optimizer = SGDP(model.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

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

        summary = [epoch, last_top1_acc, acc1_valid.item()]


        is_best = acc1_valid > best_acc1
        best_acc1 = max(acc1_valid, best_acc1)

        save_summary('rexnetv1', args.dataset, args.name, summary)


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
    print(f"Best valid Top-1 Accuracy: {best_acc1}")
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



def save_summary(arch_name, dataset, name, summary):
    r"""Save summary i.e. top-1/5 validation accuracy in each epoch
    under `summary` directory
    """
    dir_summary = pathlib.Path('summary')
    dir_path = dir_summary / 'csv'
    dir_path.mkdir(parents=True, exist_ok=True)

    file_name = '{}_{}_{}.csv'.format(arch_name, dataset, name)
    file_summ = dir_path / file_name

    if summary[0] == 0:
        with open(file_summ, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['Epoch', 'Acc@1_train', 'Acc@5_train', 'Acc@1_valid', 'Acc@5_valid']
            writer.writerow(header_list)
            writer.writerow(summary)
    else:
        file_temp = dir_path / 'temp.csv'
        shutil.copyfile(file_summ, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(file_summ, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow(summary)
        remove(file_temp)


"""
AdamP
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import math

class SGDP(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps=1e-8, delta=0.1, wd_ratio=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, delta=delta, wd_ratio=wd_ratio)
        super(SGDP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                # SGD
                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    d_p, wd_ratio = self._projection(p, grad, d_p, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio / (1-momentum))

                # Step
                p.data.add_(d_p, alpha=-group['lr'])

        return loss





if __name__ == "__main__":
    main(args)
