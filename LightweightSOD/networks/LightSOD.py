import time
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from module.DWTLayer import DWT_2D

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


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


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


class DWCon(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DWCon, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, groups=in_planes)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
        self.groupN = nn.GroupNorm(4, out_planes)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.groupN(x)
        x = self.relu(x)
        return x


class DWFS(nn.Module):
    def __init__(self, wavename='haar'):
        super(DWFS, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)  # return LL,LH,HL,HH

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return torch.cat([LL, LH + HL + HH], dim=1)


class DWTC(nn.Module):
    def __init__(self, wavename='haar'):
        super(DWTC, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)  # return LL,LH,HL,HH

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        LL = LL + LH + HL + HH
        result = torch.sum(LL, dim=[2, 3])  # x:torch.Size([64, 256, 56, 56])
        return result  ###torch.Size([64, 256])


# for low level feature
class SFBR(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SFBR, self).__init__()
        self.dwt2D = DWFS()  # Spatial attention
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_planes * 2, in_planes, 1),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
        self.dwconv3 = DWCon(in_planes, out_planes)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes * 2, in_planes // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 2, in_planes * 2, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        xori = x
        x = F.interpolate(x, size=(x.shape[-2] * 2, x.shape[-1] * 2), mode="bilinear",
                          align_corners=False)  # change resolutions # torch.Size([1, 256, 112, 112])
        dwt2 = self.dwt2D(x)  # torch.Size([1, 128, 56, 56])
        dwt = self.cv1(self.fc(dwt2))  # torch.Size([1, 64, 56, 56])
        att_dwt = dwt * xori
        res_dwt = self.dwconv3(att_dwt)
        out_sfbr = xori + res_dwt
        # print('out_sfbr', out_sfbr.size())
        return out_sfbr


class CAM1(nn.Module):
    def __init__(self, all_channel):
        super(CAM1, self).__init__()
        self.linear_e = nn.Linear(all_channel // 2, all_channel // 2, bias=False)
        self.dwconv1 = DWCon(all_channel // 2, all_channel // 2)
        self.dwconv3 = DWCon(all_channel * 2, all_channel * 2)

    def forward(self, x1, x2):
        b, c, H, W = x1.size()  # torch.Size([1, 64, 56, 56])
        channels_per_group = c // 2
        # split channel # torch.Size([1, 32, 56, 56])
        x11, x12 = torch.split(x1, channels_per_group, dim=1)
        x21, x22 = torch.split(x2, channels_per_group, dim=1)

        all_dim = H * W
        x12_flat = x12.view(-1, channels_per_group, all_dim)  # torch.Size([1, 32, 56, 56])
        x21_flat = x21.view(-1, channels_per_group, all_dim)  # torch.Size([1, 32, 56, 56])
        x12_t = torch.transpose(x12_flat, 1, 2).contiguous()  # [1, HxW, 32]
        x12_corr = self.linear_e(x12_t)
        A = torch.bmm(x21_flat, x12_corr)  # N,C2,H*W x N,H*W,C1 = N,C2,C1

        A1 = F.softmax(A.clone(), dim=2)  # N,C2,C1. dim=2 is row-wise norm. Sr
        B = F.softmax(torch.transpose(A, 1, 2), dim=2)  # N,C1,C2 column-wise norm. Sc

        x11_flat = x11.view(-1, channels_per_group, all_dim)
        x22_flat = x22.view(-1, channels_per_group, all_dim)
        br1_att = torch.bmm(A1, x11_flat).contiguous()  # N,C2,C1 X N,C1,H*W = N,C2,H*W
        br2_att = torch.bmm(B, x22_flat).contiguous()  # N,C1,C2 X N,C2,H*W = N,C1,H*W

        br1_att = br1_att.view(-1, channels_per_group, H, W)  # N,C1,H*W -> N,C1,H,W
        br1_out = self.dwconv1(br1_att + x11)

        br2_att = br2_att.view(-1, channels_per_group, H, W)  # N,C1,H*W -> N,C1,H,W
        br2_out = self.dwconv1(br2_att + x22)
        out1 = torch.add(br1_out, br2_out)  # N,C1,H,W
        out2 = torch.subtract(br1_out, br2_out)  # N,C1,H,W

        out_cross1 = torch.cat([out1, out2], dim=1)  # N,2C1,H,W
        out_cross2 = torch.cat([br1_out, br2_out], dim=1)  # N,2C1,H,W

        out_cross = torch.cat([out_cross1, out_cross2], dim=1)
        # Channel Shuffle
        out_shuffle = channel_shuffle(out_cross, 4)
        out_CaM1 = self.dwconv3(out_shuffle)  # torch.Size([1, 128, H,W])
        # print('out_CaM1', out_CaM1.size())
        return out_CaM1

    def init_weight(self):
        weight_init(self)


class LFs(nn.Module):
    def __init__(self):
        super(LFs, self).__init__()
        self.lf = SFBR(64, 64)
        self.fusion = CAM1(64)
        self.cv1 = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x2 = self.cv1(x2)
        x1_sfrb = self.lf(x1)  # torch.Size([1, 64, 56, 56])
        x2_sfrb = self.lf(x2)  # torch.Size([1, 64, 56, 56])
        x12_cam = self.fusion(x1_sfrb, x2_sfrb)  # torch.Size([1, 128, 56, 56])
        # print('x12_cam', x12_cam.size())
        return x12_cam

    def init_weight(self):
        weight_init(self)


class HFs(nn.Module):
    def __init__(self):
        super(HFs, self).__init__()
        self.mse1 = MSE2(512, 512)
        self.mse2 = MSE2(512, 512)
        self.fusion = CAM2(256)
        self.dwconv1 = DWCon(1024, 256)
        self.dwconv2 = DWCon(512, 256)
        self.cv0 = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.cv1 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x3, x4):
        # [1, 512, 56, 56], [1, 1024, 28, 28]
        x_mse1 = self.mse1(x3)  # torch.Size([1, 512, 28, 28])
        x_mse2 = self.mse2(self.cv0(x4))  # torch.Size([1, 1024, 14, 14])
        x_mse2 = F.interpolate(x_mse2, size=(x_mse1.shape[-2], x_mse1.shape[-1]), mode="bilinear", align_corners=False)
        x_mse2 = self.cv2(x_mse2)
        x_mse1 = self.cv2(x_mse1)
        x_high = self.fusion(x_mse1, x_mse2)  # torch.Size([1, 512, 28, 28])
        return x_high

    def init_weight(self):
        weight_init(self)


class MSE2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(MSE2, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_planes // 4, out_planes // 4, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_planes // 4),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(in_planes // 4, out_planes // 4, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(out_planes // 4),
            nn.ReLU()
        )
        self.cv5 = nn.Sequential(
            nn.Conv2d(in_planes // 4, out_planes // 4, kernel_size=3, dilation=5, padding=5),
            nn.BatchNorm2d(out_planes // 4),
            nn.ReLU()
        )
        self.cv7 = nn.Sequential(
            nn.Conv2d(in_planes // 4, out_planes // 4, kernel_size=3, dilation=7, padding=7),
            nn.BatchNorm2d(out_planes // 4),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, H, W = x.size()
        channels_per_group = c // 4

        # split channel
        x1, x2, x3, x4 = torch.split(x, channels_per_group, dim=1)
        out_x1 = self.cv1(x1)
        out_x1_s = self.sigmoid(out_x1)
        out_x2 = self.cv3(out_x1)
        out_x2_s = self.sigmoid(out_x2)

        out_x3 = self.cv5(out_x2)
        out_x3_s = self.sigmoid(out_x3)
        out_x4 = self.cv7(out_x3)
        out_x4_s = self.sigmoid(out_x4)

        br1 = out_x1_s * out_x2
        br2 = out_x2_s * out_x1
        br3 = out_x3_s * out_x4
        br4 = out_x4_s * out_x3

        out = torch.cat([br1, br2, br3, br4], dim=1)
        out_shuffle = channel_shuffle(out, 4)  # torch.Size([1, 512, 28, 28])
        out_mse = out_shuffle + x
        # print('out_mse', out_mse.size())
        return out_mse


class CAM2(nn.Module):
    def __init__(self, all_channel):
        super(CAM2, self).__init__()
        self.linear_e = nn.Linear(all_channel // 2, all_channel // 2, bias=False)
        self.dwconv1 = DWCon(all_channel // 2, all_channel // 2)
        self.dwconv3 = DWCon(all_channel * 2, all_channel * 2)

    def forward(self, x1, x2):
        b, c, H, W = x1.size()  # torch.Size([1, 24, 112, 112])
        channels_per_group = c // 2
        # split channel # torch.Size([1, 12, 112, 112])
        x11, x12 = torch.split(x1, channels_per_group, dim=1)
        x21, x22 = torch.split(x2, channels_per_group, dim=1)

        all_dim = H * W
        x12_flat = x12.view(-1, channels_per_group, all_dim)  # torch.Size([1, 12, 112, 112])
        x21_flat = x21.view(-1, channels_per_group, all_dim)  # torch.Size([1, 12, 112, 112])
        x12_t = torch.transpose(x12_flat, 1, 2).contiguous()  # [1, HxW, 12]
        x12_corr = self.linear_e(x12_t)
        A = torch.bmm(x21_flat, x12_corr)  # N,C2,H*W x N,H*W,C1 = N,C2,C1

        A1 = F.softmax(A.clone(), dim=2)  # N,C2,C1. dim=2 is row-wise norm. Sr
        B = F.softmax(torch.transpose(A, 1, 2), dim=2)  # N,C1,C2 column-wise norm. Sc

        x11_flat = x11.view(-1, channels_per_group, all_dim)
        x22_flat = x22.view(-1, channels_per_group, all_dim)
        br1_att = torch.bmm(A1, x11_flat).contiguous()  # N,C2,C1 X N,C1,H*W = N,C2,H*W
        br2_att = torch.bmm(B, x22_flat).contiguous()  # N,C1,C2 X N,C2,H*W = N,C1,H*W

        br1_att = br1_att.view(-1, channels_per_group, H, W)  # N,C1,H*W -> N,C1,H,W
        br1_out = self.dwconv1(br1_att + x11)

        br2_att = br2_att.view(-1, channels_per_group, H, W)  # N,C1,H*W -> N,C1,H,W
        br2_out = self.dwconv1(br2_att + x22)

        out1 = torch.add(br1_out, br2_out)  # N,C1,H,W
        out2 = torch.subtract(br1_out, br2_out)  # N,C1,H,W

        out_cross1 = torch.cat([out1, out2], dim=1)  # N,2C1,H,W
        out_cross2 = torch.cat([br1_out, br2_out], dim=1)  # N,2C1,H,W

        out_cross = torch.cat([out_cross1, out_cross2], dim=1)

        # Channel Shuffle
        out = channel_shuffle(out_cross, 4)

        # refine
        out_CaM1 = self.dwconv3(out)  # torch.Size([1, 112, H,W])

        return out_CaM1

    def init_weight(self):
        weight_init(self)


class GSRM(nn.Module):
    def __init__(self):
        super(GSRM, self).__init__()
        self.dwconv1 = DWCon(512, 512)
        self.sigmoid = nn.Sigmoid()
        self.dwconv2 = DWCon(1024, 1024)
        self.groupN = nn.GroupNorm(4, 512)
        self.cv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

    def forward(self, x5):
        x5 = self.cv1(x5)
        b, c, H, W = x5.size()  # torch.Size([1, 1024, 14, 14])
        channels_per_group = c // 2

        # split channel  torch.Size([1, 512, 14, 14])
        x51, x52 = torch.split(x5, channels_per_group, dim=1)
        x51_ = self.groupN(x51)
        x51_ = self.dwconv1(x51_)
        x51_ = self.sigmoid(x51_)
        x51_ = x51 * x51_

        x52_ = self.groupN(x52)
        x52_ = self.dwconv1(x52_)
        x52_ = self.sigmoid(x52_)
        x52_ = x52 * x52_
        x5_ = torch.cat([x51_, x52_], dim=1)

        # Channel Shufle
        out_shuffle = channel_shuffle(x5_, 4)
        out_gsrm = self.dwconv2(out_shuffle)
        return out_gsrm

    def init_weight(self):
        weight_init(self)


class Decoder3(nn.Module):
    def __init__(self, channel12, channel34, channel5):
        super(Decoder3, self).__init__()
        # Decoder 1
        self.dwconv1 = DWCon(channel12, 128)
        self.dwconv2 = DWCon(128, 64)

        # Decoder 2
        self.dwconv3 = DWCon(channel34, 256)
        self.dwconv4 = DWCon(256, 128)
        self.dwconv5 = DWCon(128, 64)

        # Decoder 3
        self.dwconv6 = DWCon(channel5, 512)
        self.dwconv7 = DWCon(512, 256)
        self.dwconv8 = DWCon(256, 128)

        self.dwconv9 = DWCon(64 * 3, 64)
        self.dwconv10 = DWCon(64, 64)

    def forward(self, x12, x34, x5):
        # torch.Size([1, 128, 56, 56]) x12       # torch.Size([1, 256, 28, 28]) x34
        # torch.Size([1, 1024, 14, 14]) x5
        # Decoder 1
        x12 = self.dwconv1(self.dwconv1(x12))  # torch.Size([1, 128, 56, 56])
        x12 = F.interpolate(x12, size=(x12.shape[-2] * 2, x12.shape[-1] * 2), mode="bilinear", align_corners=False)
        x12 = self.dwconv2(x12)  # 64x112x112
        # Decoder 2
        x34 = self.dwconv4(self.dwconv3(x34))
        x34 = F.interpolate(x34, size=(x34.shape[-2] * 4, x34.shape[-1] * 4), mode="bilinear", align_corners=False)
        x34 = self.dwconv5(x34)  # 64x112x112
        # Decoder 3
        x5 = self.dwconv7(self.dwconv6(x5))
        x5 = self.dwconv8(x5)  # 128x56x56
        x5 = F.interpolate(x5, size=(x34.shape[-2], x34.shape[-1]), mode="bilinear",
                           align_corners=False)  # 128x112x112
        br1 = x34 * x12  # 64x112x112
        x5 = self.dwconv2(x5)
        br2 = x34 * x5  # # 64x112x112

        br3 = br1 * br2
        br4 = br3 * x5
        br = torch.cat([br1, br3, br4], dim=1)  # 192x112x112
        br = self.dwconv9(br)
        out_decoder = self.dwconv10(self.dwconv10(br + x12))
        return out_decoder

    def init_weight(self):
        weight_init(self)


# RESNET-50 BACKBONE
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__=2)  # change

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
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)  # torch.Size([1, 64, 56, 56])
        out2 = self.layer1(out1)  # torch.Size([1, 256, 56, 56])
        out3 = self.layer2(out2)  # torch.Size([1, 512, 28, 28])
        out4 = self.layer3(out3)  # torch.Size([1, 1024, 14, 14])
        out5 = self.layer4(out4)  # torch.Size([1, 2048, 14, 14])
        # print('out1', out1.size())
        return out1, out2, out3, out4, out5


class ResNet_locate(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_locate, self).__init__()
        self.resnet = ResNet(block, layers)
        self.ld = LFs()
        self.hd = HFs()
        self.top = GSRM()
        self.linear = nn.Conv2d(64, 1, 3, 1, 1)
        self.decoder = Decoder3(128, 512, 1024)
        self.sigmoid = nn.Sigmoid()

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model, strict=False)

    def forward(self, x):
        # torch.Size([1, 64, 56, 56]), [1, 256, 56, 56], [1, 512, 28, 28], [1, 1024, 14, 14], [1, 2048, 14, 14]
        layer1, layer2, layer3, layer4, layer5 = self.resnet(x)
        # print('layer2', layer2.size())
        xlow = self.ld(layer1, layer2)  # torch.Size([1, 128, 56, 56])
        xhigh = self.hd(layer3, layer4)  # torch.Size([1, 512, 28, 28])
        xtop = self.top(layer5)  # torch.Size([1, 1024, 14, 14])
        # print('xhigh', xhigh.size())
        xfuse = self.decoder(xlow, xhigh, xtop)
        out = F.interpolate(self.linear(xfuse), mode='bilinear', size=x.shape[2:])
        # out_sigmoid = self.sigmoid(out)
        # print('out', out.size())
        return out


def resmot50_locate():
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    return model


if __name__ == "__main__":
    net = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    #   print(net)
    input = torch.randn(1, 3, 224, 224)
    input.to(device)
    start_time = time.time()
    result = 0
    output = net(input)
    end_time = time.time()
    time_eslapsed = (end_time - start_time) * 1000
    print("time =", time_eslapsed, "mili seconds")

    total_paramters = sum([np.prod(p.size()) for p in net.parameters()])
    print('Total network parameters: ' + str(total_paramters))
    flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False,
                                              verbose=True)
    print('Total flops: ' + str(flops))

    # print(sys.getsizeof(net))

    # #   measure inference time
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 1
    # timings = np.zeros((repetitions,1))
    # #GPU-WaRM-UP
    # for _ in range(10):
    #     _ = net(input)
    #
    # #MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = net(input)
    #         ender.record()
    #     # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    #
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(mean_syn)
