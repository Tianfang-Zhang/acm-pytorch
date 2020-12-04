import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import math
import numpy as np
import time

from utils.data import SirstDataset
from utils.lr_scheduler import adjust_learning_rate
from model.segmentation import ASKCResNetFPN, ASKCResUNet
from model.loss import SoftLoULoss
from model.metrics import SigmoidMetric, SamplewiseSigmoidMetric


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of ACM model')

    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')

    #
    # Training parameters
    #
    parser.add_argument('--batch-size', type=int, default=8, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')

    #
    # Net parameters
    #
    parser.add_argument('--backbone-mode', type=str, default='FPN', help='backbone mode: FPN, UNet')
    parser.add_argument('--fuse-mode', type=str, default='AsymBi', help='fuse mode: BiLocal, AsymBi, BiGlobal')
    parser.add_argument('--blocks-per-layer', type=int, default=4, help='blocks per layer')

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args

        ## dataset
        trainset = SirstDataset(args, mode='train')
        valset = SirstDataset(args, mode='val')
        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size)

        ## model
        layer_blocks = [args.blocks_per_layer] * 3
        channels = [8, 16, 32, 64]
        assert args.backbone_mode in ['FPN', 'UNet']
        if args.backbone_mode == 'FPN':
            self.net = ASKCResNetFPN(layer_blocks, channels, args.fuse_mode)
        elif args.backbone_mode == 'UNet':
            self.net = ASKCResUNet(layer_blocks, channels, args.fuse_mode)
        else:
            NameError

        self.net.apply(self.weight_init)
        self.net = self.net.cuda()

        ## criterion
        self.criterion = SoftLoULoss()

        ## optimizer
        self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        ## evaluation metrics
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0

        ## folders
        folder_name = '%s_%s_%s' % (time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                                         args.backbone_mode, args.fuse_mode)
        self.save_folder = ops.join('result/', folder_name)
        self.save_pkl = ops.join(self.save_folder, 'checkpoint')
        if not ops.exists('result'):
            os.mkdir('result')
        if not ops.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not ops.exists(self.save_pkl):
            os.mkdir(self.save_pkl)

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=self.save_folder)
        self.writer.add_text(folder_name, 'Args:%s, ' % args)

        ## Print info
        print('folder: %s' % self.save_folder)
        print('Args: %s' % args)
        print('backbone: %s' % args.backbone_mode)
        print('fuse mode: %s' % args.fuse_mode)
        print('layer block number:', layer_blocks)
        print('channels', channels)

    def training(self, epoch):
        # training step
        losses = []
        self.net.train()
        tbar = tqdm(self.train_data_loader)
        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()

            output = self.net(data)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f'
                                 % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses)))

        adjust_learning_rate(self.optimizer, epoch, args.epochs, args.learning_rate,
                             args.warm_up_epochs, 1e-6)

        self.writer.add_scalar('Losses/train loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch):
        self.iou_metric.reset()
        self.nIoU_metric.reset()

        eval_losses = []
        self.net.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.net(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, labels)
            eval_losses.append(loss.item())
            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()

            tbar.set_description('  Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f'
                                 %(epoch, np.mean(eval_losses), IoU, nIoU))

        pkl_name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f.pkl' % (epoch, IoU, nIoU)
        if IoU > self.best_iou:
            torch.save(self.net, ops.join(self.save_pkl, pkl_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(self.net, ops.join(self.save_pkl, pkl_name))
            self.best_nIoU = nIoU

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)


if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    for epoch in range(1, args.epochs+1):
        trainer.training(epoch)
        trainer.validation(epoch)

    print('Best IoU: %.5f, best nIoU: %.5f' % (trainer.best_iou, trainer.best_nIoU))





