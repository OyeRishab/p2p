import torch
import torch.nn as nn


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=4,
        stride=2,
        padding=1,
        negative_slope=0.2,
        use_norm=True,
    ):

        super(DownsamplingBlock, self).__init__()
        block = []
        block += [
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=(
                    not use_norm
                ),  # No need to use a bias if there is a batchnorm layer after conv
            )
        ]
        if use_norm:
            block += [nn.BatchNorm2d(num_features=c_out)]

        block += [nn.LeakyReLU(negative_slope=negative_slope)]

        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)


class UpsamplingBlock(nn.Module):
    """Defines the Unet upsampling block."""

    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=4,
        stride=2,
        padding=1,
        use_dropout=False,
        use_upsampling=False,
        mode="nearest",
    ):

        super(UpsamplingBlock, self).__init__()
        block = []
        if use_upsampling:
            mode = mode if mode in ("nearest", "bilinear", "bicubic") else "nearest"

            block += [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=mode),
                    nn.Conv2d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=3,
                        stride=1,
                        padding=padding,
                        bias=False,
                    ),
                )
            ]
        else:
            block += [
                nn.ConvTranspose2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]

        block += [nn.BatchNorm2d(num_features=c_out)]

        if use_dropout:
            block += [nn.Dropout(0.5)]

        block += [nn.ReLU()]

        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)
