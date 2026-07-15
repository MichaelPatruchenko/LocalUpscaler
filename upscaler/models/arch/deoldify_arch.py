"""DeOldify U-Net architecture for inference only.

Reimplemented from https://github.com/jantic/DeOldify (MIT License).
Matches the exact state_dict key structure produced by fastai's DynamicUnetWide/Deep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, resnet34


def _spectral_norm(module):
    """Apply old-style spectral norm for compatibility with legacy state dicts."""
    return torch.nn.utils.spectral_norm(module)


def _weight_norm(module):
    """Apply old-style weight norm for compatibility with legacy state dicts."""
    return torch.nn.utils.weight_norm(module)


class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.query = _spectral_norm(nn.Conv1d(n_channels, n_channels // 8, 1, bias=False))
        self.key = _spectral_norm(nn.Conv1d(n_channels, n_channels // 8, 1, bias=False))
        self.value = _spectral_norm(nn.Conv1d(n_channels, n_channels, 1, bias=False))
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        b, c, h, w = x.shape
        f = x.view(b, c, -1)
        q = self.query(f)
        k = self.key(f)
        v = self.value(f)
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        o = self.gamma * torch.bmm(v, beta.transpose(1, 2)) + f
        return o.view(b, c, h, w)


class PixelShuffleICNR(nn.Module):
    """PixelShuffle upsampling matching fastai's key structure.

    State dict keys: conv.0.weight_orig/weight_u/weight_v, conv.1.weight/bias/...
    """

    def __init__(self, ni, nf, scale=2):
        super().__init__()
        self.conv = nn.Sequential(
            _spectral_norm(nn.Conv2d(ni, nf * (scale ** 2), kernel_size=1, bias=False)),
            nn.BatchNorm2d(nf * (scale ** 2)),
        )
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x))


class PixelShuffleWN(nn.Module):
    """Final PixelShuffle using weight_norm (not spectral_norm). No BN.

    State dict keys: conv.0.weight_g/weight_v/bias
    """

    def __init__(self, ni, nf, scale=2):
        super().__init__()
        self.conv = nn.Sequential(
            _weight_norm(nn.Conv2d(ni, nf * (scale ** 2), kernel_size=1)),
        )
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x))


class UnetBlockWide(nn.Module):
    """Matches fastai's UnetBlockWide state_dict.

    Keys: shuf.conv.{0,1}.*, bn.*, conv.{0,2[,3].*}
    """

    def __init__(self, up_in_c, skip_c, shuf_out_c, conv_out_c, self_attention=False):
        super().__init__()
        self.shuf = PixelShuffleICNR(up_in_c, shuf_out_c)
        self.bn = nn.BatchNorm2d(skip_c)

        conv_in = shuf_out_c + skip_c
        layers = [
            _spectral_norm(nn.Conv2d(conv_in, conv_out_c, 3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv_out_c),
        ]
        if self_attention:
            layers.append(SelfAttention(conv_out_c))
        self.conv = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in, skip):
        up_out = self.shuf(up_in)
        cat_x = self.relu(self.bn(skip))
        if up_out.shape[-2:] != cat_x.shape[-2:]:
            up_out = F.interpolate(up_out, size=cat_x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([up_out, cat_x], dim=1)
        return self.conv(x)


class UnetBlockDeep(nn.Module):
    """Deep variant decoder block with conv1 + conv2.

    State dict keys: shuf.conv.{0,1}.*, bn.*, conv1.{0,2}.*, conv2.{0,2[,3]}.*
    """

    def __init__(self, up_in_c, skip_c, shuf_out_c, conv_out_c, self_attention=False):
        super().__init__()
        self.shuf = PixelShuffleICNR(up_in_c, shuf_out_c)
        self.bn = nn.BatchNorm2d(skip_c)

        conv_in = shuf_out_c + skip_c
        self.conv1 = nn.Sequential(
            _spectral_norm(nn.Conv2d(conv_in, conv_out_c, 3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv_out_c),
        )
        conv2_layers = [
            _spectral_norm(nn.Conv2d(conv_out_c, conv_out_c, 3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv_out_c),
        ]
        if self_attention:
            conv2_layers.append(SelfAttention(conv_out_c))
        self.conv2 = nn.Sequential(*conv2_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in, skip):
        up_out = self.shuf(up_in)
        cat_x = self.relu(self.bn(skip))
        if up_out.shape[-2:] != cat_x.shape[-2:]:
            up_out = F.interpolate(up_out, size=cat_x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([up_out, cat_x], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class FinalBlock(nn.Module):
    """Final convolution block with .layers attribute matching fastai naming.

    Keys: layers.0.0.*, layers.1.0.*
    """

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(_spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))),
            nn.Sequential(_spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))),
        )

    def forward(self, x):
        return self.layers(x)


class DeOldifyWide(nn.Module):
    """DeOldify Wide generator (Stable/Video) - ResNet101 backbone.

    Structure matches fastai DynamicUnetWide state_dict exactly:
    layers[0]=body, [1]=BN, [2]=ReLU, [3]=middle, [4-7]=decoders,
    [8]=final_shuf, [9]=merge, [10]=final_conv, [11]=output
    """

    def __init__(self, nf_factor=2.0, y_range=(-3.0, 3.0)):
        super().__init__()
        self.y_range = y_range

        # Build flat ResNet101 body
        backbone = resnet101(weights=None)
        body = nn.Sequential(
            backbone.conv1,    # 0: Conv2d(3, 64, 7, stride=2)
            backbone.bn1,      # 1: BN(64)
            backbone.relu,     # 2: ReLU
            backbone.maxpool,  # 3: MaxPool2d → 64ch
            backbone.layer1,   # 4: → 256ch
            backbone.layer2,   # 5: → 512ch
            backbone.layer3,   # 6: → 1024ch
            backbone.layer4,   # 7: → 2048ch
        )

        enc_out = 2048

        # Middle conv block
        middle = nn.Sequential(
            nn.Sequential(
                _spectral_norm(nn.Conv2d(enc_out, enc_out * 2, 3, padding=1, bias=False)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(enc_out * 2),
            ),
            nn.Sequential(
                _spectral_norm(nn.Conv2d(enc_out * 2, enc_out, 3, padding=1, bias=False)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(enc_out),
            ),
        )

        # Decoder blocks
        # dec4: input=2048, skip=1024, shuf→512, conv→512
        dec4 = UnetBlockWide(up_in_c=2048, skip_c=1024, shuf_out_c=512, conv_out_c=512)
        # dec3: input=512, skip=512, shuf→512, conv→512, +self_attention
        dec3 = UnetBlockWide(up_in_c=512, skip_c=512, shuf_out_c=512, conv_out_c=512,
                             self_attention=True)
        # dec2: input=512, skip=256, shuf→512, conv→512
        dec2 = UnetBlockWide(up_in_c=512, skip_c=256, shuf_out_c=512, conv_out_c=512)
        # dec1: input=512, skip=64, shuf→256, conv→256
        dec1 = UnetBlockWide(up_in_c=512, skip_c=64, shuf_out_c=256, conv_out_c=256)

        # Final pixel shuffle (weight_norm, no BN)
        final_shuf = PixelShuffleWN(256, 256)

        # Merge placeholder (no params - concat with input gives 256+3=259 channels)
        merge = nn.Identity()

        # Final conv block (259→259)
        final_conv = FinalBlock(259)

        # Output conv (259→3)
        output_conv = nn.Sequential(
            _spectral_norm(nn.Conv2d(259, 3, 1)),
        )

        self.layers = nn.ModuleList([
            body,            # 0
            nn.BatchNorm2d(enc_out),  # 1
            nn.ReLU(),       # 2
            middle,          # 3
            dec4,            # 4
            dec3,            # 5
            dec2,            # 6
            dec1,            # 7
            final_shuf,      # 8
            merge,           # 9
            final_conv,      # 10
            output_conv,     # 11
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        orig = x
        encoder = self.layers[0]

        # Run encoder, capture intermediate features
        skips = {}
        h = x
        for i, layer in enumerate(encoder):
            h = layer(h)
            # Save outputs after maxpool(3), layer1(4), layer2(5), layer3(6)
            if i in (3, 4, 5, 6):
                skips[i] = h

        # Middle
        h = self.layers[1](h)  # BN
        h = self.layers[2](h)  # ReLU
        h = self.layers[3](h)  # middle conv

        # Decoder
        h = self.layers[4](h, skips[6])  # skip from layer3 (1024ch)
        h = self.layers[5](h, skips[5])  # skip from layer2 (512ch)
        h = self.layers[6](h, skips[4])  # skip from layer1 (256ch)
        h = self.layers[7](h, skips[3])  # skip from maxpool (64ch)

        # Final upsample
        h = self.layers[8](h)

        # Merge with original input
        if h.shape[-2:] != orig.shape[-2:]:
            h = F.interpolate(h, size=orig.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, orig], dim=1)  # 256+3=259

        # Final layers
        h = self.layers[10](h)
        h = self.layers[11](h)

        if self.y_range is not None:
            h = self.sigmoid(h) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        return h


class DeOldifyDeep(nn.Module):
    """DeOldify Deep generator (Artistic) - ResNet34 backbone.

    Dimensions derived from actual state dict of ColorizeArtistic_gen.pth:
    dec4: shuf(512→256), conv1(256+256=512→768), conv2(768→768)
    dec3: shuf(768→384), conv1(384+128=512→768), conv2(768→768)+SA
    dec2: shuf(768→384), conv1(384+64=448→672), conv2(672→672)
    dec1: shuf(672→336), conv1(336+64=400→300), conv2(300→300)
    final: shuf_wn(300→300), merge→303, conv(303→303)×2, out(303→3)
    """

    def __init__(self, nf_factor=1.5, y_range=(-3.0, 3.0)):
        super().__init__()
        self.y_range = y_range

        backbone = resnet34(weights=None)
        body = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        enc_out = 512

        middle = nn.Sequential(
            nn.Sequential(
                _spectral_norm(nn.Conv2d(512, 1024, 3, padding=1, bias=False)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(1024),
            ),
            nn.Sequential(
                _spectral_norm(nn.Conv2d(1024, 512, 3, padding=1, bias=False)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512),
            ),
        )

        # Dimensions from state dict analysis
        dec4 = UnetBlockDeep(up_in_c=512, skip_c=256, shuf_out_c=256, conv_out_c=768)
        dec3 = UnetBlockDeep(up_in_c=768, skip_c=128, shuf_out_c=384, conv_out_c=768,
                             self_attention=True)
        dec2 = UnetBlockDeep(up_in_c=768, skip_c=64, shuf_out_c=384, conv_out_c=672)
        dec1 = UnetBlockDeep(up_in_c=672, skip_c=64, shuf_out_c=336, conv_out_c=300)

        final_shuf = PixelShuffleWN(300, 300)
        final_conv = FinalBlock(303)
        output_conv = nn.Sequential(_spectral_norm(nn.Conv2d(303, 3, 1)))

        self.layers = nn.ModuleList([
            body,
            nn.BatchNorm2d(enc_out),
            nn.ReLU(),
            middle,
            dec4, dec3, dec2, dec1,
            final_shuf,
            nn.Identity(),
            final_conv,
            output_conv,
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        orig = x
        encoder = self.layers[0]
        skips = {}
        h = x
        for i, layer in enumerate(encoder):
            h = layer(h)
            if i in (3, 4, 5, 6):
                skips[i] = h

        h = self.layers[1](h)
        h = self.layers[2](h)
        h = self.layers[3](h)

        h = self.layers[4](h, skips[6])
        h = self.layers[5](h, skips[5])
        h = self.layers[6](h, skips[4])
        h = self.layers[7](h, skips[3])

        h = self.layers[8](h)
        if h.shape[-2:] != orig.shape[-2:]:
            h = F.interpolate(h, size=orig.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, orig], dim=1)

        h = self.layers[10](h)
        h = self.layers[11](h)

        if self.y_range is not None:
            h = self.sigmoid(h) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return h


def build_deoldify_model(variant="stable"):
    """Build a DeOldify generator for the given variant."""
    if variant in ("stable", "video"):
        return DeOldifyWide(nf_factor=2.0)
    elif variant == "artistic":
        return DeOldifyDeep(nf_factor=1.5)
    raise ValueError(f"Unknown DeOldify variant: {variant}")
