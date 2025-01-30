from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class DAFTBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_dim=32, ndim_non_img=3):
        super().__init__()
        self.in_channels = in_channels
        self.aux_dims = 2 * self.in_channels
        self.bottleneck_dim = bottleneck_dim
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # create aux net
        layers = [
            (
                "aux_base",
                nn.Linear(ndim_non_img + in_channels, self.bottleneck_dim, bias=False),
            ),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.aux_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def forward(self, feature_map, x_aux):
        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)

        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)

        v_scale, v_shift = torch.split(attention, self.in_channels, dim=1)
        v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
        v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)

        return (v_scale * feature_map) + v_shift


class SFCN_DAFT(nn.Module):
    def __init__(
        self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True
    ):
        super(SFCN_DAFT, self).__init__()
        n_layer = len(channel_number)

        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]

            out_channel = channel_number[i]

            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1
                    ),
                )
            else:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel, out_channel, maxpool=False, kernel_size=1, padding=0
                    ),
                )

        self.DAFT = DAFTBlock(
            in_channels=channel_number[-1], bottleneck_dim=32, ndim_non_img=3
        )

        self.classifier = nn.Sequential()

        avg_shape = [5, 6, 5]
        self.classifier.add_module("average_pool", nn.AvgPool3d(avg_shape))

        if dropout is True:
            self.classifier.add_module("dropout", nn.Dropout(0.5))

        in_channel = channel_number[-1]
        out_channel = output_dim

        self.classifier.add_module("flatten", nn.Flatten())
        self.classifier.add_module("fc", nn.Linear(in_channel, out_channel))

    @staticmethod
    def conv_layer(
        in_channel,
        out_channel,
        maxpool=True,
        kernel_size=3,
        padding=0,
        maxpool_stride=2,
    ):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channel, out_channel, padding=padding, kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channel, out_channel, padding=padding, kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        return layer

    def forward(self, x, tabular_data):
        x_f = self.feature_extractor(x)
        x_d = self.DAFT(x_f, tabular_data)
        out = self.classifier(x_d)
        return out
