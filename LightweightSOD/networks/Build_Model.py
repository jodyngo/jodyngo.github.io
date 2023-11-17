import torch
from torch import nn

from networks.LightSOD import resmot50_locate
# from networks.EC2Net import resmot50_locate


class PSAMNet(nn.Module):
    def __init__(self, base):
        super(PSAMNet, self).__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)  # list
        return out


def build_model(base_model_cfg='resmot'):
    if base_model_cfg == 'resmot':
        return PSAMNet(resmot50_locate())


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == "__main__":
    net = build_model(base_model_cfg='resmot')
    # print(net)
    output = net(torch.randn(1, 3, 224, 224))

