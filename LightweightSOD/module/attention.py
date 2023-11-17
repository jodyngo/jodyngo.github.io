import time

import ptflops
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Self_ST(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim):
        super(Self_ST, self).__init__()
        self.chanel_in = in_dim
        self.chanel_out = out_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #
        self.stride = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=3)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        b, c, h, w = x.shape
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x)
        proj_query = proj_query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # print('proj_query', proj_query.size())

        # Process Key
        proj_key = self.key_conv(x)
        proj_key = self.stride(proj_key)
        width_s, height_s = proj_key.size()[2:]
        proj_key = proj_key.view(m_batchsize, -1, width_s * height_s)
        # print('proj_key', proj_key.size())
        energy = torch.bmm(proj_query, proj_key)  # transpose check  torch.Size([1, 196, 196])
        attention = self.softmax(energy)  # BX (N) X (N)  [1, 7744, 7744]

        # Process Value
        # proj_value = self.value_conv(x)
        proj_value = self.value_conv(x)
        proj_value = self.stride(proj_value)
        proj_value = proj_value.view(m_batchsize, -1, width_s * height_s)
        # print('proj_value', proj_value.size())
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # print('out', out.size())
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out
        return out + x


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class DCCF(nn.Module):
    def __init__(self, in_dim):
        super(DCCF, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=3 // 2)

    def forward(self, x_low, x_high):
        m_batchsize, _, height, width = x_low.size()
        proj_query = self.query_conv(x_low)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x_high)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x_high)

        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        out = self.gamma * (out_H + out_W) + x_high
        # print('outDCCF', out.size())
        return out


if __name__ == '__main__':
    net = Self_ST(64, 64)
    start_time = time.time()
    output = net(torch.randn(1, 64, 84, 84))
    end_time = time.time()
    time_eslapsed = (end_time - start_time) * 1000
    print("time =", time_eslapsed, "mili seconds")
    total_parameters = sum([np.prod(p.size()) for p in net.parameters()])
    print('Total network parameters: ' + str(total_parameters))

    macs, params = ptflops.get_model_complexity_info(net, (64, 84, 84))
    print('Computational complexity: ', macs)
    print('Number of parameters: ', params)
