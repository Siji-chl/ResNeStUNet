import torch.nn as nn
from splat import SplAtConv3d


class ConvNormLReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=None):

        super(ConvNormLReLU, self).__init__()

        if stride is not None:
            self.conv = nn.Conv3d(input_channels, output_channels, kernel_size,
                                  padding=[(i - 1) // 2 for i in kernel_size],
                                  stride=stride)
        else:
            self.conv = nn.Conv3d(input_channels, output_channels, kernel_size,
                                  padding=[(i - 1) // 2 for i in kernel_size])

        self.norm = nn.InstanceNorm3d(output_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)

        self.nonlin = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=None):

        super(ResidualBlock, self).__init__()

        self.kernel_size = kernel_size

        self.stride = stride
        self.out_planes = out_planes
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                               stride=stride)
        self.norm1 = nn.InstanceNorm3d(out_planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size])
        self.norm2 = nn.InstanceNorm3d(out_planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        if (self.stride is not None and any((i != 1 for i in self.stride))) or (in_planes != out_planes):
            stride_here = stride if stride is not None else 1
            self.downsample_skip = nn.Sequential(nn.Conv3d(in_planes, out_planes, 1, stride_here, bias=False),
                                                 nn.InstanceNorm3d(out_planes, eps=1e-5, momentum=0.1, affine=True,
                                                                   track_running_stats=False))
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):

        residual = x
        out = self.nonlin1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        residual = self.downsample_skip(residual)
        out += residual

        return self.nonlin2(out)


class ResNeStBlock(nn.Module):
    def __init__(self, inplanes, kernel_size, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64, last_gamma=False):
        super(ResNeStBlock, self).__init__()
        group_width = int(inplanes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(group_width, eps=1e-5, momentum=0.1, affine=True,
                                       track_running_stats=False)
        self.nonlin1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.radix = radix

        self.attention = SplAtConv3d(
            group_width, group_width, kernel_size=kernel_size,
            stride=stride, groups=cardinality,
            radix=radix)

        self.conv2 = nn.Conv3d(
            group_width, inplanes, kernel_size=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(inplanes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.norm2.weight)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin1(out)

        out = self.attention(out)

        out = self.conv2(out)
        out = self.norm2(out)

        residual = self.downsample(residual)

        out += residual
        out = self.nonlin2(out)

        return out


class ResNeStLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_blocks, first_stride,
                 block_down, block, radix=2, cardinality=1,
                 bottleneck_width=64):
        super(ResNeStLayer, self).__init__()

        self.convs = nn.Sequential(
            block_down(input_channels, output_channels, kernel_size, first_stride),
            *[block(output_channels, kernel_size, radix=radix, cardinality=cardinality,
                    bottleneck_width=bottleneck_width) for _ in range(num_blocks - 1)])

    def forward(self, x):
        return self.convs(x)


class StackedConvLayers(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_convs, first_stride=None):
        super(StackedConvLayers, self).__init__()

        self.convs = nn.Sequential(
            ConvNormLReLU(input_channels, output_channels, kernel_size, first_stride),
            *[ConvNormLReLU(output_channels, output_channels, kernel_size) for _ in
              range(num_convs - 1)]
        )

    def forward(self, x):
        return self.convs(x)