from functools import partial
import math


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from networks.deeplabv3_batchnorm import SynchronizedBatchNorm2d

norm_layer = partial(SynchronizedBatchNorm2d, momentum= 3e-4)


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.block_size, stride=self.block_size)
        return unfolded_x.view(n, c * self.block_size ** 2, h // self.block_size, w // self.block_size)

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c=32, k=64,upscale_factor=4, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, upscale_factor*upscale_factor*c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(upscale_factor*upscale_factor*c, c, 1, bias=False),
            norm_layer(c))

        self.down = SpaceToDepth(upscale_factor)#对特征图下采样
        self.up = nn.PixelShuffle(upscale_factor)#对特征图上采样

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv 第一个卷积
        x = self.conv1(x)

        x=self.up(x)#先进行上采样，使得在HR尺寸范围内进行自注意力

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k   E步，公式（12）
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k     M步，公式（13）
                mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = self.down(x)  # 恢复到LR的尺寸大小
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

