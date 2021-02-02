import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import transpose, flatten, reshape, unsqueeze, matmul, zeros
from math import sqrt, floor
import random

# Adapted but strongly changed from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class AttUNet(nn.Module):
    def __init__(self, channels_d=[3, 16, 24, 40, 112, 512], channels_u=[512, 256, 256, 128, 64, 2], bilinear=True):
        super().__init__()

        self.d1 = DoubleConv(channels_d[0], channels_d[1])
        self.dx = nn.ModuleList([Down(channels_d[l], channels_d[l+1])
                                 for l in range(1, len(channels_d) - 1)])
        self.ux = nn.ModuleList([Up(channels_u[l], channels_d[-2-l], channels_u[l+1], bilinear=bilinear)
                                 for l in range(0, len(channels_u) - 2)])
        self.uo = nn.Conv2d(channels_u[-2], channels_u[-1], kernel_size=1)
        self.attention_rpp = RandSelfAttention2D(24, key_dim=24, value_dim=24, rand_portion=0.05)
        self.attention_rp = RandSelfAttention2D(40, key_dim=40, value_dim=40, rand_portion=0.1)
        #self.attention_r = RandSelfAttention2D(112, key_dim=112, value_dim=112, rand_portion=0.3)
        self.attention_l = RandSelfAttention2D(512, rand_portion=1.0) 

        self.upsample = nn.Upsample((225,225), mode='nearest')

    def forward(self, x):
        x = self.d1(x)

        # down
        residuals = [x]
        for d_id in range(len(self.dx)):
            x = self.dx[d_id](x)
            residuals.append(x)
        
        x, scores_l = self.attention_l(x)
        #residuals[-2], scores_h = self.attention_r(residuals[-2])
        residuals[-3], scores_hp = self.attention_rp(residuals[-3])
        residuals[-4], scores_hpp = self.attention_rpp(residuals[-4])
        with torch.no_grad():
            scores = (self.upsample(scores_l) + self.upsample(scores_hp) + self.upsample(scores_hpp))

        # Up
        for u_id in range(len(self.ux)):
            x = self.ux[u_id](x, residuals[-2 - u_id])

        logits = self.uo(x)
        return logits, scores / torch.max(scores)


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

class RandSelfAttention2D(nn.Module):
        """
        Module with 1-head self-attention layer for 2D images. 
        Additionally only a random fixed-size portion of the pixels is considered for the attention.
        The input is a (batch_size, channels, length, height)-shaped tensor.
        The output has the same shape as the input.
        :param channels: number of input channels (== output channels)
        :param key_dim: dimension of the keys and queries
        :param value_dim: dimension of the values
        :param bias: determines whether the linear layers use a bias
        :param rand_portion: portion of pixels that is considered for the attention.
        """
        
        def __init__(self, channels, key_dim=512, value_dim=512, bias=False, rand_portion=0.1):
                super().__init__()

                self.key_dim = key_dim
                self.value_dim = value_dim
                self.rand_portion = rand_portion

                self.key_trans = nn.Linear(channels, key_dim, bias=bias)
                self.query_trans = nn.Linear(channels, key_dim, bias=bias)
                self.value_trans = nn.Linear(channels, value_dim, bias=bias)

                self.scores_softmax = nn.Softmax(1)

        def forward_single(self, input):
                """
                Determines the attention of a single input of a batch of inputs.
                """
                
                in_channels, length, height = input.shape
                x = torch.transpose(torch.transpose(input, 0, 1), 1, 2)
                x = torch.flatten(x, 0, 1)

                # randomized choice of pixels
                pixel_num = x.size()[0]
                att_size = floor(self.rand_portion * pixel_num)
                att_choice = random.sample(range(pixel_num), att_size)

                # determining the keys, queries and values from the input
                keys = self.key_trans(x)[att_choice]
                queries = self.query_trans(x)
                values = self.value_trans(x)[att_choice]

                # using scaled dot-product to determine scores between keys and queries
                scaling = sqrt(self.key_dim)
                scores = matmul(queries, transpose(keys, 0, 1)) / scaling
                scores = self.scores_softmax(scores)
                scores_out = torch.zeros((scores.shape[0])).to(scores.device)
                scores_out[att_choice] = torch.sum(scores,dim=0)

                # weighting the values of each pixel with the scores
                weighted_values = values[:, None] * transpose(scores, 0, 1)[:, :, None]
                output = weighted_values.sum(dim=0)

                output = unsqueeze(output, 1)
                output = reshape(output, (length, height, self.value_dim))
                output = transpose(transpose(output, 0, 2), 1, 2)
                return output, scores_out

        def forward(self, input):
                b_size, in_channels, height, width = input.size()
                output = zeros((b_size, self.value_dim, height, width)).cuda()
                scores_out = []
                for i, x in enumerate(input):
                        output[i], score_out = self.forward_single(x)
                        scores_out.append(score_out)
                return output, torch.stack(scores_out).reshape(b_size, 1, height, width)


class SelfAttention2D(nn.Module):
        """
        Module with 1-head self-attention layer for 2D images. 
        The input is a (batch_size, channels, length, height)-shaped tensor.
        The output has the same shape as the input.
        :param channels: number of input channels (== output channels)
        :param key_dim: dimension of the keys and queries
        :param value_dim: dimension of the values
        :param bias: determines whether the linear layers use a bias
        """
        
        def __init__(self, channels, key_dim=512, value_dim=512, bias=False):
                super().__init__()

                self.key_dim = key_dim
                self.value_dim = value_dim

                self.key_trans = nn.Linear(channels, key_dim, bias=bias)
                self.query_trans = nn.Linear(channels, key_dim, bias=bias)
                self.value_trans = nn.Linear(channels, value_dim, bias=bias)

                self.scores_softmax = nn.Softmax(1)

        def forward_single(self, input):
                """
                Determines the attention of a single input of a batch of inputs.
                """
                
                in_channels, length, height = input.shape
                x = transpose(transpose(input, 0, 1), 1, 2)
                x = flatten(x, 0, 1)

                # determining the keys, queries and values from the input
                keys = self.key_trans(x)
                queries = self.query_trans(x)
                values = self.value_trans(x)

                # using scaled dot-product to determine scores between keys and queries
                scaling = sqrt(self.key_dim)
                scores = matmul(queries, transpose(keys, 0, 1)) / scaling
                scores = self.scores_softmax(scores)
                scores_out = torch.sum(scores,dim=0)

                # weighting the values of each pixel with the scores
                weighted_values = values[:, None] * transpose(scores, 0, 1)[:, :, None]
                output = weighted_values.sum(dim=0)

                output = unsqueeze(output, 1)
                output = reshape(output, (length, height, self.value_dim))
                output = transpose(transpose(output, 0, 2), 1, 2)
                return output, scores_out

        def forward(self, input):
                b_size, in_channels, height, width = input.size()
                output = zeros((b_size, self.value_dim, height, width)).cuda()
                scores_out = []
                for i, x in enumerate(input):
                        output[i], score_out = self.forward_single(x)
                        scores_out.append(score_out)
                return output, torch.stack(scores_out).reshape(b_size, 1, height, width)
