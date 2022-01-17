import torch
from torch import nn
import torch.nn.functional as F


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv3d(nn.Module):
    """
    Split-Attention Conv3d, a 3D version of SplAtConv2d proposed by [1].

    [1] Zhang et al., ResNeSt: Split-Attention Networks

    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, groups=1,
                 radix=2, reduction_factor=4):
        super(SplAtConv3d, self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = nn.Conv3d(in_channels, channels*radix, kernel_size, stride,
                           padding=[(i - 1) // 2 for i in kernel_size], groups=groups*radix)
        self.norm0 = nn.InstanceNorm3d(channels*radix, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin0 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.fc1 = nn.Conv3d(channels, inter_channels, 1, groups=self.cardinality)
        self.norm1 = nn.InstanceNorm3d(inter_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.fc2 = nn.Conv3d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        print(x.size())
        x = self.norm0(x)
        x = self.nonlin0(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool3d(gap, 1)
        gap = self.fc1(gap)

        gap = self.norm1(gap)
        gap = self.nonlin1(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()
