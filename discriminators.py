import torch.nn as nn
from .blocks import DownsamplingBlock, UpsamplingBlock


class PixelDiscriminator(nn.Module):
    def __init__(self, c_in=3, c_hid=64):
        super(PixelDiscriminator, self).__init__()
        self.model = nn.Sequential(
            DownsamplingBlock(
                c_in, c_hid, kernel_size=1, stride=1, padding=0, use_norm=False
            ),
            DownsamplingBlock(c_hid, c_hid * 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=c_hid * 2, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, c_in=3, c_hid=64, n_layers=3):

        super(PatchDiscriminator, self).__init__()
        model = [DownsamplingBlock(c_in, c_hid, use_norm=False)]

        n_p = 1
        n_c = 1
        for n in range(1, n_layers):
            n_p = n_c
            n_c = min(2**n, 8)

            model += [DownsamplingBlock(c_hid * n_p, c_hid * n_c)]

        n_p = n_c
        n_c = min(2**n_layers, 8)
        model += [DownsamplingBlock(c_hid * n_p, c_hid * n_c, stride=1)]

        model += [
            nn.Conv2d(
                in_channels=c_hid * n_c,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=True,
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchGAN(nn.Module):
    def __init__(self, c_in=3, c_hid=64, mode="patch", n_layers=3):
        super(PatchGAN, self).__init__()
        if mode == "pixel":
            self.model = PixelDiscriminator(c_in, c_hid)
        else:
            self.model = PatchDiscriminator(c_in, c_hid, n_layers)

    def forward(self, x):
        return self.model(x)
