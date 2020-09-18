import torch
import torch.nn as nn
import networks.macn_ops as ops

class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseBlock, self).__init__()
        reduction=8
        self.b1 = ops.CAResidualBlock(in_channels, out_channels, reduction=reduction)
        self.b2 = ops.CAResidualBlock(in_channels, out_channels, reduction=reduction)
        self.b3 = ops.CAResidualBlock(in_channels, out_channels, reduction=reduction)

    def forward(self, x):
        assert False, 'Need overwrite.'


class first_block(BaseBlock):
    def __init__(self, in_channels, out_channels, group=1, mode='None'):
        super(first_block, self).__init__(in_channels, out_channels)
        self.c1 = ops.BasicBlock(in_channels*2, out_channels, 1, 1, 0)
        self.c2 = ops.BasicBlock(in_channels*3, out_channels, 1, 1, 0)
        self.c3 = ops.BasicBlock(in_channels*4, out_channels, 1, 1, 0)

    def forward(self, x):
        result = list()
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        result.append(o1)
        b2 = self.b2(o1)
        c2 = torch.cat([c0, b1, b2], dim=1)
        o2 = self.c2(c2)
        result.append(o2)
        b3 = self.b3(o2)
        c3 = torch.cat([c0, b1, b2, b3], dim=1)
        o3 = self.c3(c3)
        result.append(o3)
        return result


class normal_block(BaseBlock):
    def __init__(self, in_channels, out_channels):
        super(normal_block, self).__init__(in_channels, out_channels)
        self.c0 = ops.BasicBlock(in_channels * 1 + in_channels, out_channels, 1, 1, 0)
        self.c0_fix = ops.BasicBlock(in_channels * 2 + in_channels, out_channels, 1, 1, 0)
        self.c1 = ops.BasicBlock(in_channels * 2 + in_channels, out_channels, 1, 1, 0)
        self.c2 = ops.BasicBlock(in_channels * 3 + in_channels, out_channels, 1, 1, 0)
        self.c3 = ops.BasicBlock(in_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x_list, parallel_input=None):
        result = list()
        if parallel_input is None:
            c0 = torch.cat([x_list[-1], x_list[0]], dim=1)
            o0 = self.c0(c0)
        else:
            c0 = torch.cat([x_list[-1], x_list[0], parallel_input], dim=1)
            o0 = self.c0_fix(c0)
        b1 = self.b1(o0)
        c1 = torch.cat([o0, b1, x_list[1]], dim=1)
        o1 = self.c1(c1)
        result.append(o1)
        b2 = self.b2(o1)
        c2 = torch.cat([o0, b1, b2, x_list[2]], dim=1)
        o2 = self.c2(c2)
        result.append(o2)
        b3 = self.b3(o2)
        c3 = torch.cat([o0, b1, b2, b3], dim=1)
        o3 = self.c3(c3)
        result.append(o3)
        return result


class first_carn_cell(nn.Module):
    def __init__(self, filter):
        super(first_carn_cell, self).__init__()
        self.cell1 = first_block(filter, filter)
        self.cell2 = normal_block(filter, filter)
        self.cell3 = normal_block(filter, filter)

    def forward(self, x):
        result = list()
        c0 = o0 = x
        x_list1 = self.cell1(o0)
        x_list2 = self.cell2(x_list1)
        x_list3 = self.cell3(x_list2)
        result.append(x_list1[-1])
        result.append(x_list2[-1])
        result.append(x_list3[-1])
        return result


class normal_carn_cell(nn.Module):
    def __init__(self, filter):
        super(normal_carn_cell, self).__init__()
        self.cell1 = first_block(filter, filter)
        self.cell2 = normal_block(filter, filter)
        self.cell3 = normal_block(filter, filter)
        self.c0 = ops.BasicBlock(filter * 1 + filter, filter, 1, 1, 0)

    def forward(self, x_list):
        result = list()
        c0 = torch.cat([x_list[-1], x_list[0]], dim=1)
        o0 = self.c0(c0)
        x_list1 = self.cell1(o0)
        x_list2 = self.cell2(x_list1, x_list[1])
        x_list3 = self.cell3(x_list2, x_list[2])
        result.append(x_list1[-1])
        result.append(x_list2[-1])
        result.append(x_list3[-1])
        return result


class MACN(nn.Module):
    def __init__(self):
        super(MACN, self).__init__()
        first_filter = 64
        filter = 24
        last_filter = 128
        multi_scale = False
        #scale = kwargs.get("scale") if not multi_scale else 0
        scale=4
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry = nn.Conv2d(3, first_filter, 3, 1, 1)
        self.entry_shrink = nn.Conv2d(first_filter, filter, 3, 1, 1)
        self.cell1 = first_carn_cell(filter)
        self.cell2 = normal_carn_cell(filter)
        self.cell3 = normal_carn_cell(filter)
        self.fusion = nn.Conv2d(filter*3, filter*3, 3, 1, 1, bias=False)
        self.shrink = nn.Conv2d(filter*3, filter,   3, 1, 1, bias=False)
        self.upsample = ops.my_UpsampleBlock(filter, last_filter, scale=scale, multi_scale=multi_scale, group=1)
        self.exit = nn.Conv2d(last_filter//4, 3, 3, 1, 1, bias=False)
        self.bilinear = ops.BilinearUpsampleBlock(multi_scale=multi_scale, scale=scale)

    def forward(self, x):
        scale=4
        n = self.sub_mean(x)
        hr_bilinear = self.bilinear(n, scale)
        n = self.entry(n)
        tmp_long = n = self.entry_shrink(n)
        x_list1 = self.cell1(n)
        x_list2 = self.cell2(x_list1)
        x_list3 = self.cell3(x_list2)
        n = torch.cat([x_list1[-1], x_list2[-1], x_list3[-1]], dim=1)
        n = self.fusion(n)
        n = self.shrink(n)
        n = n + tmp_long
        n = self.upsample(n, scale=scale)
        n = self.exit(n)
        n = n + hr_bilinear
        n = self.add_mean(n)
        return n