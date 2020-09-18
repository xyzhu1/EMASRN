import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()
        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out


class FastSigmoid(nn.Module):
    def __init__(self):
        super(FastSigmoid, self).__init__()

    def forward(self, x):
        abs = torch.abs(x) + 1
        return torch.div(x, abs)


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.shrink = nn.AdaptiveAvgPool2d(1)
        modules_body = [
            self.shrink,
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y=None):
        attention = self.body(x)
        return x * attention


# Channel Attention (CA) Layer
class FastCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FastCALayer, self).__init__()
        self.shrink = nn.AdaptiveAvgPool2d(1)
        self.fast_sigmoid = FastSigmoid()
        modules_body = [
            self.shrink,
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            self.fast_sigmoid
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y=None):
        attention = self.body(x)
        return x * attention

# channel attention residual block
class CAResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(CAResidualBlock, self).__init__()
        modules_body = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            CALayer(out_channels, reduction)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y=None, mode='origin'):
        out = self.body(x)
        if mode == 'origin':
            out = F.relu(out + x)
        elif mode == 'separate':
            out = F.relu(out + y)
        else:
            assert False, 'mode is wrong !'
        return out


# channel attention residual block
class CAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, group=1):
        super(CAEResidualBlock, self).__init__()
        modules_body = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            CALayer(out_channels, reduction)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


# channel attention residual block
class FastCAResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(FastCAResidualBlock, self).__init__()
        modules_body = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            FastCALayer(out_channels, reduction)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y=None, mode='origin'):
        out = self.body(x)
        if mode == 'origin':
            out = F.relu(out + x)
        elif mode == 'separate':
            out = F.relu(out + y)
        else:
            assert False, 'mode is wrong !'
        return out


class BilinearUpsampleBlock(nn.Module):
    def __init__(self, scale, multi_scale):
        super(BilinearUpsampleBlock, self).__init__()
        if multi_scale:
            self.bilinear2 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.bilinear3 = nn.Upsample(scale_factor=3, mode='bilinear')
            self.bilinear4 = nn.Upsample(scale_factor=4, mode='bilinear')
        else:
            self.bilinear =  nn.Upsample(scale_factor=scale, mode='bilinear')
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.bilinear2(x)
            elif scale == 3:
                return self.bilinear3(x)
            elif scale == 4:
                return self.bilinear4(x)
        else:
            return self.bilinear(x)


class my_UpsampleBlock(nn.Module):
    def __init__(self,
                 filters, last_filters, scale, multi_scale,
                 group=1):
        super(my_UpsampleBlock, self).__init__()
        if multi_scale:
            self.up2 = _my_UpsampleBlock(filters, last_filters, scale=2, group=group)
            self.up3 = _my_UpsampleBlock(filters, last_filters, scale=3, group=group)
            self.up4 = _my_UpsampleBlock(filters, last_filters, scale=4, group=group)
        else:
            self.up =  _my_UpsampleBlock(filters, last_filters, scale=scale, group=group)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _my_UpsampleBlock(nn.Module):
    def __init__(self,
                 filters, last_filters, scale,
                 group=1):
        super(_my_UpsampleBlock, self).__init__()
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                if _ == 0:
                    modules += [nn.Conv2d(filters, last_filters, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Conv2d(last_filters//4, last_filters, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(filters, last_filters//4*9, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out