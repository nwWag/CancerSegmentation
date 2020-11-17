import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted but strongly changed from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class SuperUNet(nn.Module):
    def __init__(self, encoder, channels_d=[3, 16, 24, 40, 112, 1280], channels_u=[1280, 256, 256, 128, 64, 64, 2], bilinear=True):
        super().__init__()

        self.encoder = encoder
        self.ux = nn.ModuleList([Up(channels_u[l], channels_d[-2-l], channels_u[l+1], bilinear=bilinear)
                                 for l in range(0, len(channels_u) - 2)])
        self.uo = nn.Conv2d(channels_u[-2], channels_u[-1], kernel_size=1)

    def forward(self, x):
        # Down
        residuals = self.encoder(x, endpoints=True)
        x = residuals[-1]

        # Up
        for u_id in range(len(self.ux)):
            x = self.ux[u_id](x, residuals[-2 - u_id])

        logits = self.uo(x)
        return logits


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.PReLU):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            activation(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.PReLU, pool=nn.MaxPool2d(2)):
        super().__init__()
        self.down_conv = nn.Sequential(
            pool,
            DoubleConv(in_channels, out_channels,
                       mid_channels=mid_channels, activation=activation)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, res_channels, out_channels, mid_channels=None, activation=nn.PReLU, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels + res_channels, out_channels, in_channels // 2 if mid_channels is None else mid_channels, activation=activation)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + res_channels, out_channels // 2, in_channels //
                                   2 if mid_channels is None else mid_channels, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
