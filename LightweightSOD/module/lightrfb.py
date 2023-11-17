import sys
import time

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import numpy as np


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # ([1, 512, 33, 33])
        y = self.avg_pool(x).view(b, c)  # ([1, 512])
        y = self.fc(y).view(b, c, 1, 1)  # ([1, 512, 1, 1])
        out = x * y  # ([1, 512, 33, 33])
        return out


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None
        # self.relu = h_sigmoid() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LightRFB(nn.Module):
    def __init__(self, channels_in, channels_mid, channels_out):
        super(LightRFB, self).__init__()
        self.global_se = SELayer(channels_in)
        self.reduce = nn.Sequential(nn.Conv2d(channels_in, channels_mid, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(channels_mid),
                                    nn.PReLU(channels_mid))
        self.br0 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=1, bias=False,
                      bn=True, relu=True),
            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=1, padding=1, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.br1 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=1, padding=1, groups=channels_mid, bias=False,
                      bn=True, relu=False),
            BasicConv(channels_mid, channels_mid, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=3, padding=3, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.br2 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=5, dilation=1, padding=2, groups=channels_mid, bias=False,
                      bn=True, relu=False),
            BasicConv(channels_mid, channels_mid, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=5, padding=5, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.br3 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=7, dilation=1, padding=3, groups=channels_mid, bias=False,
                      bn=True, relu=False),
            BasicConv(channels_mid, channels_mid, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=7, padding=7, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.point_global = BasicConv(channels_mid * 4 + channels_in, channels_out, kernel_size=1, bias=False, bn=True,
                                      relu=True)

    def forward(self, x):
        # x_reduce = self.reduce(self.global_se(x))
        x_reduce = self.reduce(x)
        x0 = self.br0(x_reduce)
        x1 = self.br1(x_reduce)
        x2 = self.br2(x_reduce)
        x3 = self.br3(x_reduce)
        out = self.point_global(torch.cat([x, x0, x1, x2, x3], dim=1))
        # print('x2', x2.size())
        return out


if __name__ == "__main__":
    m = LightRFB(64, 64, 256)
    t = torch.zeros(1, 64, 28, 28)
    input_shapes = t
    start_time = time.time()
    m(t)
    end_time = time.time()
    time_eslapsed = (end_time - start_time) * 1000
    print("time =", time_eslapsed, "mili seconds")
    total_paramters = sum([np.prod(p.size()) for p in m.parameters()])
    print('Total network parameters: ' + str(total_paramters))
    macs, params = get_model_complexity_info(m, (64, 28, 28))
    print('Total flops: ', macs)
    print('Number of parameters: ', params)

    # print(sys.getsizeof(LightRFB))
