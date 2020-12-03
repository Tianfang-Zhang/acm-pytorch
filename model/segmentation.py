import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AsymBiChaFuseReduce, BiLocalChaFuseReduce, BiGlobalChaFuseReduce


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
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
    def __init__(self, layer_blocks, channels, fuse_mode='AsymBi'):
        super(ASKCResNetFPN, self).__init__()

        stem_width = channels[0]
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width*2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.fuse23 = self._fuse_layer(channels[3], channels[2], channels[2], fuse_mode)
        self.fuse12 = self._fuse_layer(channels[2], channels[1], channels[1], fuse_mode)

        self.head = _FCNHead(channels[1], 1)

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

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layer = []
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer


class ASKCResUNet(nn.Module):
    def __init__(self, layer_blocks, channels, fuse_mode='AsymBi'):
        super(ASKCResUNet, self).__init__()

        stem_width = int(channels[0])
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, 2*stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*stem_width),
            nn.ReLU(True),

            nn.MaxPool2d(3, 2, 1),
        )

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)
        self.uplayer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                         in_channels=channels[2], out_channels=channels[2], stride=1)
        self.fuse2 = self._fuse_layer(channels[3], channels[2], channels[2], fuse_mode)
        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)
        self.uplayer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                         in_channels=channels[1], out_channels=channels[1], stride=1)
        self.fuse1 = self._fuse_layer(channels[2], channels[1], channels[1], fuse_mode)

        self.head = _FCNHead(channels[1], 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        deconc2 = self.deconv2(c3)
        fusec2 = self.fuse2(deconc2, c2)
        upc2 = self.uplayer2(fusec2)

        deconc1 = self.deconv1(upc2)
        fusec1 = self.fuse1(deconc1, c1)
        upc1 = self.uplayer1(fusec1)

        pred = self.head(upc1)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')
        return out



    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer
