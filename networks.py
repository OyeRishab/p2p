import torch
import torch.nn as nn
from blocks import DownsamplingBlock, UpsamplingBlock


class UnetEncoder(nn.Module):
    def __init__(self, c_in=3, c_out=512):

        super(UnetEncoder, self).__init__()
        self.enc1 = DownsamplingBlock(c_in, 64, use_norm=False)  # C64
        self.enc2 = DownsamplingBlock(64, 128)  # C128
        self.enc3 = DownsamplingBlock(128, 256)  # C256
        self.enc4 = DownsamplingBlock(256, 512)  # C512
        self.enc5 = DownsamplingBlock(512, 512)  # C512
        self.enc6 = DownsamplingBlock(512, 512)  # C512
        self.enc7 = DownsamplingBlock(512, 512)  # C512
        self.enc8 = DownsamplingBlock(512, c_out)  # C512

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        out = [x8, x7, x6, x5, x4, x3, x2, x1]  # latest activation is the first element
        return out


class UnetDecoder(nn.Module):
    def __init__(self, c_in=512, c_out=64, use_upsampling=False, mode="nearest"):

        super(UnetDecoder, self).__init__()
        self.dec1 = UpsamplingBlock(
            c_in, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode
        )  # CD512
        self.dec2 = UpsamplingBlock(
            1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode
        )  # CD1024
        self.dec3 = UpsamplingBlock(
            1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode
        )  # CD1024
        self.dec4 = UpsamplingBlock(
            1024, 512, use_upsampling=use_upsampling, mode=mode
        )  # C1024
        self.dec5 = UpsamplingBlock(
            1024, 256, use_upsampling=use_upsampling, mode=mode
        )  # C1024
        self.dec6 = UpsamplingBlock(
            512, 128, use_upsampling=use_upsampling, mode=mode
        )  # C512
        self.dec7 = UpsamplingBlock(
            256, 64, use_upsampling=use_upsampling, mode=mode
        )  # C256
        self.dec8 = UpsamplingBlock(
            128, c_out, use_upsampling=use_upsampling, mode=mode
        )  # C128

    def forward(self, x):
        x9 = torch.cat([x[1], self.dec1(x[0])], 1)  # (N,1024,H,W)
        x10 = torch.cat([x[2], self.dec2(x9)], 1)  # (N,1024,H,W)
        x11 = torch.cat([x[3], self.dec3(x10)], 1)  # (N,1024,H,W)
        x12 = torch.cat([x[4], self.dec4(x11)], 1)  # (N,1024,H,W)
        x13 = torch.cat([x[5], self.dec5(x12)], 1)  # (N,512,H,W)
        x14 = torch.cat([x[6], self.dec6(x13)], 1)  # (N,256,H,W)
        x15 = torch.cat([x[7], self.dec7(x14)], 1)  # (N,128,H,W)
        out = self.dec8(x15)  # (N,64,H,W)
        return out


class UnetGenerator(nn.Module):
    def __init__(self, c_in=3, c_out=3, use_upsampling=False, mode="nearest"):
        super(UnetGenerator, self).__init__()
        self.encoder = UnetEncoder(c_in=c_in)
        self.decoder = UnetDecoder(use_upsampling=use_upsampling, mode=mode)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        outE = self.encoder(x)
        outD = self.decoder(outE)
        out = self.head(outD)
        return out
