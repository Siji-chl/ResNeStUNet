import numpy as np
import torch
from torch import nn
from blocks import StackedConvLayers, ResNeStLayer, ResNeStBlock, ResidualBlock



class ResNeStUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage,
                 conv_kernel_sizes, strides, block_down, block, max_num_features=320,
                 radix=2, cardinality=1, bottleneck_width=64):
        super(ResNeStUNetEncoder, self).__init__()

        num_stages = len(conv_kernel_sizes)
        assert len(strides) == len(conv_kernel_sizes) == len(num_blocks_per_stage)

        self.stages = []
        self.stage_output_features = []
        self.stage_conv_kernel_size = []
        self.stage_stride = []
        self.num_blocks_per_stage = num_blocks_per_stage

        self.initial_conv = nn.Conv3d(input_channels, base_num_features, 3, padding=1, stride=1, dilation=1, bias=False)
        self.initial_norm = nn.InstanceNorm3d(base_num_features, eps=1e-5, momentum=0.1, affine=True,
                                              track_running_stats=False)
        self.initial_nonlin = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        current_input_features = base_num_features
        for stage in range(num_stages):
            current_output_features = min(base_num_features * 2 ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_stride = strides[stage]

            current_stage = ResNeStLayer(current_input_features, current_output_features, current_kernel_size,
                                          self.num_blocks_per_stage[stage], current_stride, block_down, block,
                                         radix=radix, cardinality=cardinality, bottleneck_width=bottleneck_width)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_kernel_size.append(current_kernel_size)
            self.stage_stride.append(current_stride)

            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)

    def forward(self, x):

        skip_outputs = []
        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))
        for s in self.stages:
            x = s(x)
            skip_outputs.append(x)
        return skip_outputs


class PlainConvDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None):
        super(PlainConvDecoder, self).__init__()
        self.num_classes = num_classes

        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_stride = previous.stage_stride
        previous_stage_conv_kernel_size = previous.stage_conv_kernel_size

        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        self.stage_output_features = previous_stage_output_features
        self.stage_stride = previous_stage_stride
        self.stage_conv_kernel_size = previous_stage_conv_kernel_size

        num_stages = len(previous_stages) - 1

        self.upsamplings = []
        self.stages = []
        self.deep_supervision_outputs = []

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]

            self.upsamplings.append(nn.ConvTranspose3d(features_below, features_skip, self.stage_stride[s + 1],
                                                       self.stage_stride[s + 1], bias=False))

            self.stages.append(StackedConvLayers(2 * features_skip, features_skip,
                                                 self.stage_conv_kernel_size[s],
                                                 num_blocks_per_stage[i]))

            if s != 0:
                seg_layer = nn.Conv3d(features_skip, num_classes, 1, 1, 0, 1, 1, bias=False)
                self.deep_supervision_outputs.append(seg_layer)

        self.segmentation_output = nn.Conv3d(features_skip, num_classes, 1, 1, 0, 1, 1, bias=False)

        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips):
        skips = skips[::-1]
        seg_outputs = []

        x = skips[0]
        for i in range(len(self.upsamplings)):
            x = self.upsamplings[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            if i != len(self.upsamplings) - 1:
                ds_out = self.deep_supervision_outputs[i](x)
                seg_outputs.append(ds_out)

        seg_outputs.append(self.segmentation_output(x))
        return seg_outputs[::-1]


class ResNeStUNet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_classes, num_blocks_per_stage_encoder,
                 conv_kernel_sizes, strides, num_blocks_per_stage_decoder,
                 block_down, block, max_features=320,
                 radix=2, cardinality=1, bottleneck_width=64):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = ResNeStUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                          conv_kernel_sizes, strides, block_down=block_down, block=block,
                                          max_num_features=max_features, radix=radix, cardinality=cardinality,
                                          bottleneck_width=bottleneck_width)

        self.decoder = PlainConvDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder)

    def forward(self, x):
        out = self.encoder(x)
        return self.decoder(out)


if __name__ == "__main__":
    num_blocks_per_stage_encoder = (1, 2, 3, 4, 4, 4, 4)
    num_blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2)
    strides = ([1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2])
    conv_kernel_sizes = ([1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3])
    with torch.no_grad():
        network = ResNeStUNet(3, 32, 2, num_blocks_per_stage_encoder, conv_kernel_sizes, strides,
                              num_blocks_per_stage_decoder, ResidualBlock, ResNeStBlock).cuda()
        print(network)

        x = torch.rand((2, 3, 14, 320, 320)).cuda()
        y = network(x)
        for i in y:
            print(i.shape)