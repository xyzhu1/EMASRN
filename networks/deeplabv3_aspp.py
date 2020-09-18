import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.deeplabv3_batchnorm import SynchronizedBatchNorm2d
from .blocks import ConvBlock, DeconvBlock, MeanShift


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()


        self._init_weight()

    def forward(self, x):
        y = self.atrous_conv(x)
        x = self.bn(y)

        return self.relu(x), y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class _ASPPModule2(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule2, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        # self.conv=nn.Conv2d(64, 32, 1)
        self.conv = ConvBlock(2*32, 32, kernel_size=1, act_type='prelu', norm_type= None)

        self._init_weight()

    def forward(self, x, y):
        x = self.atrous_conv(x)
        y=torch.cat([x, y], 1)
        y=self.conv(y)
        x = self.bn(y)

        return self.relu(x), y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride=16, BatchNorm=SynchronizedBatchNorm2d):
        super(ASPP, self).__init__()

        inplanes = 32
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 32, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule2(inplanes, 32, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule2(inplanes, 32, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule2(inplanes, 32, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 32, 1, stride=1, bias=False),
                                             BatchNorm(32),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(160, 32, 1, bias=False)
        # self.bn1 = BatchNorm(32)
        # self.relu = nn.ReLU()
        # self.conv_5 = ConvBlock(2 * 32, 32, kernel_size=1, act_type='prelu', norm_type=None)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):

        # print("x.shapex.shapex.shapex.shapex.shapex.shape****************************",x.shape)
        x1, y1 = self.aspp1(x)
        x2, y2 = self.aspp2(x, y1)
        x3, y3 = self.aspp3(x, y2)
        x4, y4 = self.aspp4(x, y3)
        # print("x444444444444444444444444444444444****************************", x4.shape)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # x5 = torch.cat([x5, y4], 1)
        # x5=self.conv_5(x5)


        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)

        # x = self.bn1(x)
        # x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


