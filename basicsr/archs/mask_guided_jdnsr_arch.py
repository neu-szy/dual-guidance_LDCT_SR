import os

import torch
from torch import nn
import math
from basicsr.utils.registry import ARCH_REGISTRY



class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3
    ):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class Fusion(nn.Module):
    def __init__(
            self,
            num_feat
    ):
        super(Fusion, self).__init__()
        self.Q = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.K = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.V = nn.Conv2d(num_feat, num_feat, 3, 1, 1)


    def forward(self, x, avg_ct, mask):
        b, c, w, h = x.shape

        q = self.Q(mask)
        k = self.K(x)
        v = self.V(avg_ct)

        qk = torch.matmul(q.view(b, c, w*h), k.view(b, w*h, c)) / math.sqrt(w*h)
        qk = torch.softmax(qk.view(b, c*c), dim=1).view(b, c, c)
        return torch.matmul(qk, v.view(b, c, w*h)).view(b, c, w, h) + x


class SAM(nn.Module):
    def __init__(
            self,
            num_feat
    ):
        super(SAM, self).__init__()
        self.conv_prelu_deconv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, dilation=2),
            nn.PReLU(),
            nn.ConvTranspose2d(num_feat, num_feat, 3, 1, 1, dilation=2)
        )
        self.deconv_prelu_conv = nn.Sequential(
                nn.ConvTranspose2d(num_feat, num_feat, 3, 1, 1, dilation=2),
                nn.PReLU(),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1, dilation=2)
            )

    def forward(self, x):
        return self.conv_prelu_deconv(x) + self.deconv_prelu_conv(x) + x


class CAM(nn.Module):
    def __init__(
            self,
            num_feat,
            squeeze_factor=16
    ):
        super(CAM, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(num_feat, num_feat // squeeze_factor),
            nn.PReLU(),
            nn.Linear(num_feat // squeeze_factor, num_feat)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w, h = x.shape
        shortcut = x
        x = self.avg_pooling(x).view(b, c)
        x = self.mlp(x)
        x = self.sigmoid(x).view(b, c, 1, 1)
        x = x * shortcut
        return x + shortcut


class CALayer(nn.Module):
    def __init__(self, num_feat, reduction=16):
        super(CALayer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // reduction, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(num_feat // reduction, num_feat, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)


class ConvPReluCAM(nn.Module):
    def __init__(
            self,
            num_feat
    ):
        super(ConvPReluCAM, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.PReLU(),
            CAM(num_feat)
        )

    def forward(self, x):
        return self.f(x)


class Block(nn.Module):
    def __init__(
            self,
            num_feat
    ):
        super(Block, self).__init__()
        self.block1 = ConvPReluCAM(num_feat)
        self.block2 = ConvPReluCAM(num_feat)
        self.block3 = ConvPReluCAM(num_feat)
        self.block4 = ConvPReluCAM(num_feat)
        self.sam1 = SAM(num_feat)
        self.sam2 = SAM(num_feat)

    def forward(self, x):
        x0 = x
        sam1 = self.sam1(x0)
        x1 = self.block1(x0)
        sam2 = self.sam2(x1)
        x2 = self.block2(x1) + sam1
        x3 = self.block3(x2) + sam2
        x4 = self.block4(x3)
        return x4


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SRHead(nn.Module):
    def __init__(
            self,
            scale,
            out_channels,
            num_feat
    ):
        super(SRHead, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.PReLU()
        )
        self.upsample = Upsample(scale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        features = self.upsample(x)
        sr = self.conv_last(features)
        return sr, features


class DenoiseHead(nn.Module):
    def __init__(
            self,
            out_channels,
            num_feat
    ):
        super(DenoiseHead, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        features = self.conv1(x)
        denoise = self.conv2(features)
        return denoise, features


@ARCH_REGISTRY.register()
class MaskGuidedJDNSR(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            scale=2,
            num_feat=64,
            num_block=6,
            mode="jdnsr",
            loop=None
    ):
        super(MaskGuidedJDNSR, self).__init__()
        # assert mode in ["jdnsr", "segmentation"], f"Support jdnsr and segmentatio mode only, do not support {mode}"
        self.mode = mode
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1, bias=True)
        backbone = []
        if mode == "jdnsr":
            # super resolution and denoisng
            for _ in range(num_block):
                backbone.append(
                    nn.ModuleList(
                        [
                            Fusion(num_feat),
                            Block(num_feat)
                        ]
                    )
                )
            self.backbone = nn.ModuleList(backbone)
            self.sr_head = SRHead(scale=scale, out_channels=out_channels, num_feat=num_feat)
            self.denoise_head = DenoiseHead(out_channels=out_channels, num_feat=num_feat)

            self.bicubic_up = nn.Upsample(scale_factor=scale, mode="bicubic")


        else:
            raise NotImplemented(f"do not support the mode {mode}")

    def forward(self, x, avg_ct=None, mask=None):
        if self.mode == "jdnsr":
            x_origin = x
            x_origin_bicubic = self.bicubic_up(x_origin)
            x = self.conv_first(x)
            avg_ct = self.conv_first(avg_ct)
            mask = self.conv_first(mask)
            for (f, b) in self.backbone:
                x = f(x, avg_ct, mask)
                x = b(x)
            hrLD, hrLD_features = self.sr_head(x)
            LRnd, LRnd_features = self.denoise_head(x)
            hrnd1, _ = self.denoise_head(hrLD_features)
            hrnd2, _ = self.sr_head(LRnd_features)

            hrLD = hrLD + x_origin_bicubic
            LRnd = LRnd + x_origin
            hrnd1 = hrnd1 + x_origin_bicubic
            hrnd2 = hrnd2 + x_origin_bicubic

            return hrLD, LRnd, (hrnd1 + hrnd2) / 2

