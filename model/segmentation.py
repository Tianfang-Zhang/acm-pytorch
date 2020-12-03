import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AsymBiChaFuseReduce

class CIFARBasicBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(CIFARBasicBlockV1, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class ASKCResNetFPN(nn.Module):
    def __init__(self):
        super(ASKCResNetFPN, self).__init__()

        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layer_num=4,
                                       in_channels=16, out_channels=16, stride=1)
        self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layer_num=4,
                                       in_channels=16, out_channels=32, stride=2)
        self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layer_num=4,
                                       in_channels=32, out_channels=64, stride=2)

        self.fuse23 = self._fuse_layer(64, 32, 32)
        self.fuse12 = self._fuse_layer(32, 16, 16)

        self.head = _FCNHead(16, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        out = self.layer3(c2)

        out = F.interpolate(out, size=[hei//8, wid//8], mode='bilinear')
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei//4, wid//4], mode='bilinear')
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')

        return out

    def _make_layer(self, block, layer_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layer = []
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(layer_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        return layer
