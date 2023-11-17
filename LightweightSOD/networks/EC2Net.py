import time
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from module.lightrfb import LightRFB
from module.attention import DCCF, Self_ST
from ptflops import get_model_complexity_info

affine_par = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: ' + n)
        if isinstance(m, nn.Conv2D):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, value=0.0)

        elif isinstance(m, (nn.BatchNorm2D, nn.InstanceNorm2D)):
            nn.init.constant_(m.weight, value=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, value=0.0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, value=0.0)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2D):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2D):
            pass
        elif isinstance(m, nn.Upsample):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.MaxPool2D):
            pass
        else:
            m.init_weight()


# 1x1 Conv
class MappingModule(nn.Module):
    def __init__(self):
        super(MappingModule, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(1024, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(2048, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, out1, out2, out3, out4, out5):
        out2 = self.cv1(out2)
        out3 = self.cv2(out3)
        out4 = self.cv3(out4)
        out5 = self.cv4(out5)
        return out1, out2, out3, out4, out5

    def init_weight(self):
        weight_init(self)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class ChannelAttentionSE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttentionSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# for low level feature Cross Channel Spatial Attention
class CSCA(nn.Module):
    def __init__(self):
        super(CSCA, self).__init__()
        self.ca = ChannelAttentionSE(64)
        self.sa = SpatialAttention()
        self.combine = conbine_feature()
        self.cv = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x2, x3):
        xc2 = self.ca(x2)  # ([1, 64, 56, 56])
        xs3 = self.sa(x3)  # ([1, 1, 56, 56])

        # Cross
        xs3 = F.interpolate(xs3, size=x2.shape[2:], mode='bilinear')  # ([1, 1, 56, 56])
        out_2 = xs3 * x2
        xc2 = F.interpolate(xc2, size=x3.shape[2:], mode='bilinear')  # ([1, 1, 56, 56])
        out_3 = xc2 * x3
        out_3 = F.interpolate(out_3, size=x2.shape[2:], mode='bilinear')  # ([1, 1, 56, 56])
        out_2 = torch.cat([out_2, out_3], dim=1)
        out = self.cv3(self.cv(out_2))

        return out

    def init_weight(self):
        weight_init(self)


# for low level feature f2 and f3
class LFs(nn.Module):
    def __init__(self):
        super(LFs, self).__init__()
        self.lf = CSCA()

    def forward(self, x2, x3):
        xs = self.lf(x2, x3)  # ([1, 64, 56, 56])
        return xs

    def init_weight(self):
        weight_init(self)


# for high level feature f4 and f5
class HFs(nn.Module):
    def __init__(self):
        super(HFs, self).__init__()
        # LightRFB
        self.fa4 = LightRFB(channels_in=64, channels_mid=64, channels_out=256)
        self.fa5 = LightRFB(channels_in=64, channels_mid=64, channels_out=256)
        # Self-Attention
        self.sa4 = Self_ST(64, 64)
        self.sa5 = Self_ST(64, 64)

        self.cv = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 1),  # 128
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(64 * 4, 64, 1),  # 128
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.combine = conbine_feature()

    def forward(self, x4, x5):   # x5 ([1, 64, 21, 21])
        x4_fa, x5_fa = self.fa4(x4), self.fa5(x5)   # ([1, 128, 21, 21])
        # x4_sa, x5_sa = self.sa4(x4_fa), self.sa5(x5_fa)  # ([1, 128, 21, 21])
        # Cross
        x5_sa = self.cv2(x5_fa)
        x4_sa = self.cv2(x4_fa)
        x4_out = x4 * x5_sa
        x5_out = x5 * x4_sa
        hf = torch.concat([x4_out, x5_out], 1)
        hf = self.cv(hf)

        # multiplication
        # hf = self.cv2(x4_sa * x5_sa)
        # print('hf', hf.size())
        return hf

    def init_weight(self):
        weight_init(self)


class conbine_feature(nn.Module):
    def __init__(self):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(64, 64)
        self.up2_low = nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(64)
        self.up2_act = nn.PReLU(64)
        self.refine = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU())

    def forward(self, low_fea, high_fea):
        high_fea = self.up2_high(high_fea)
        low_fea = self.up2_bn2(self.up2_low(low_fea))
        refine_feature = self.refine(self.up2_act(high_fea + low_fea))
        return refine_feature


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, l, h):
        h = F.interpolate(h, size=l.shape[2:], mode='bilinear')
        f = l * h
        f = self.cv1(f)
        l = self.cv2(l)
        f = F.relu(f + l)
        f = self.cv3(f)
        return f

    def init_weight(self):
        weight_init(self)


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        # self.act = nn.PReLU(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)
        # output = self.act(output)

        return output


# RESNET-50 BACKBONE
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        return out1, out2, out3, out4, out5


class ResNet_locate(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_locate, self).__init__()
        self.resnet = ResNet(block, layers)
        self.mp = MappingModule()
        self.ld = LFs()
        self.hd = HFs()
        self.cross = DCCF(64)
        self.linear = nn.Conv2d(64, 1, 3, 1, 1)
        self.combine = conbine_feature()
        self.fusion = Fusion()

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model, strict=False)

    def forward(self, x):
        # in_channel(64, 256, 512, 1024, 2048)
        layer1, layer2, layer3, layer4, layer5 = self.resnet(x)
        layer1, layer2, layer3, layer4, layer5 = self.mp(layer1, layer2, layer3, layer4, layer5)
        xlow = self.ld(layer2, layer3)
        xhigh = self.hd(layer4, layer5)
        xhigh = F.interpolate(xhigh, size=(xlow.shape[-2], xlow.shape[-1]), mode="bilinear", align_corners=False)
        x_cross = self.cross(xlow, xhigh)
        out = F.interpolate(self.linear(x_cross), mode='bilinear', size=x.shape[2:])
        return out


def resmot50_locate():
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    return model


if __name__ == "__main__":
    net = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    #   print(net)
    input = torch.randn(1, 3, 336, 336)
    input.to(device)
    start_time = time.time()
    result = 0
    output = net(input)
    end_time = time.time()
    time_eslapsed = (end_time - start_time) * 1000
    print("time =", time_eslapsed, "mili seconds")

    total_paramters = sum([np.prod(p.size()) for p in net.parameters()])
    print('Total network parameters: ' + str(total_paramters))
    flops, params = get_model_complexity_info(net, (3, 336, 336), as_strings=True, print_per_layer_stat=False,
                                              verbose=True)
    print('Total flops: ' + str(flops))

