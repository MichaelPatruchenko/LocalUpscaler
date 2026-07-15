"""DDColor architecture for inference only.

Bundled from https://github.com/piddnad/DDColor (MIT License).
Combines ConvNeXt encoder, UNet decoder, and MultiScaleColorDecoder.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    if hasattr(torch.nn.init, "trunc_normal_"):
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(a, b)
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class NormType:
    Batch = "Batch"
    Weight = "Weight"
    Spectral = "Spectral"


# ---------------------------------------------------------------------------
# LayerNorm (channels_first / channels_last)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ---------------------------------------------------------------------------
# ConvNeXt Encoder
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        for i in range(4):
            self.add_module(f"norm{i}", LayerNorm(dims[i], eps=1e-6, data_format="channels_first"))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f"norm{i}")
            norm_layer(x)
        return self.norm(x.mean([-2, -1]))


# ---------------------------------------------------------------------------
# Hook (captures intermediate encoder features)
# ---------------------------------------------------------------------------

class Hook:
    def __init__(self, m):
        self.feature = None
        self.hook = m.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

def _spectral_norm(module):
    """Old-style spectral norm for compatibility with legacy state dicts."""
    return torch.nn.utils.spectral_norm(module)


def _weight_norm(module):
    """Old-style weight norm for compatibility with legacy state dicts."""
    return torch.nn.utils.weight_norm(module)


def custom_conv_layer(ni, nf, ks=3, stride=1, padding=None, use_activ=True,
                      norm_type=NormType.Spectral, extra_bn=False,
                      self_attention=False):
    """Conv layer matching DDColor's key structure.

    Returns nn.Sequential with indices:
    - [0] = spectral/weight-normed Conv2d
    - [1] = ReLU (if use_activ)
    - [next] = BatchNorm2d (if extra_bn or Batch norm_type)
    - [next] = SelfAttention (if self_attention)
    """
    if padding is None:
        padding = ks // 2
    bn = norm_type in (NormType.Batch,) or extra_bn
    bias = not bn
    conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)
    if norm_type == NormType.Spectral:
        conv = _spectral_norm(conv)
    elif norm_type == NormType.Weight:
        conv = _weight_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(nn.ReLU(inplace=True))
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


class CustomPixelShuffle_ICNR(nn.Module):
    """PixelShuffle upsampling matching DDColor's key structure.

    Keys: conv.0.weight_orig/weight_u/weight_v[, conv.1.weight/bias/...]
    """

    def __init__(self, ni, nf, blur=True, norm_type=NormType.Spectral, scale=2,
                 extra_bn=False):
        super().__init__()
        self.conv = custom_conv_layer(
            ni, nf * (scale ** 2), ks=1, use_activ=False,
            norm_type=norm_type, extra_bn=extra_bn)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0)) if blur else None
        self.blur = nn.AvgPool2d(2, stride=1) if blur else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        if self.blur is not None:
            x = self.blur(self.pad(x))
        return x


class UnetBlockWide(nn.Module):
    """DDColor UNet block matching state dict keys.

    Keys: shuf.conv.{0,1}.*, bn.*, conv.{0,2}.*
    """

    def __init__(self, up_in_c, x_in_c, n_out, hook, blur=True, self_attention=False,
                 norm_type=NormType.Spectral):
        super().__init__()
        self.hook = hook
        up_out = n_out
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, norm_type=norm_type, extra_bn=True)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(
            ni, n_out, norm_type=norm_type, self_attention=self_attention,
            extra_bn=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in):
        up_out = self.shuf(up_in)
        cat_x = self.relu(self.bn(self.hook.feature))
        if up_out.shape[-2:] != cat_x.shape[-2:]:
            up_out = F.interpolate(up_out, size=cat_x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([up_out, cat_x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# Transformer Decoder Components
# ---------------------------------------------------------------------------

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            return tgt + self.dropout(tgt2)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask)[0]
            return tgt + self.dropout(tgt2)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu",
                 normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            return tgt + self.dropout(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ---------------------------------------------------------------------------
# MultiScaleColorDecoder
# ---------------------------------------------------------------------------

class MultiScaleColorDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_queries=100, nheads=8,
                 dim_feedforward=2048, dec_layers=9, pre_norm=False,
                 color_embed_dim=256, enforce_input_project=True, num_scales=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = dec_layers
        self.num_feature_levels = num_scales

        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_scales, hidden_dim)

        self.input_proj = nn.ModuleList()
        for in_ch in in_channels:
            if in_ch != hidden_dim or enforce_input_project:
                proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)
                nn.init.kaiming_uniform_(proj.weight, a=1)
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Sequential())

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0,
                                   normalize_before=pre_norm))
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0,
                                    normalize_before=pre_norm))
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0,
                         normalize_before=pre_norm))

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        src, pos = [], []
        for i, feature in enumerate(x):
            pos.append(self.pe_layer(feature).flatten(2).permute(2, 0, 1))
            src.append(
                (self.input_proj[i](feature).flatten(2) + self.level_embed.weight[i][None, :, None])
                .permute(2, 0, 1)
            )

        bs = src[0].shape[1]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index], pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](
                output, query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output).transpose(0, 1)
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)
        return out


# ---------------------------------------------------------------------------
# DuelDecoder (UNet + Color Decoder)
# ---------------------------------------------------------------------------

class DuelDecoder(nn.Module):
    def __init__(self, hooks, nf=512, blur=True, last_norm=NormType.Spectral,
                 num_queries=256, num_scales=3, dec_layers=9):
        super().__init__()
        self.hooks = hooks
        self.nf = nf

        # Build decoder layers (reverse order of hooks, skip the last one)
        self.layers = nn.ModuleList()
        in_c = hooks[-1].feature.shape[1]
        out_c = nf
        setup_hooks = hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            self.layers.append(
                UnetBlockWide(in_c, feature_c, out_c, hook, blur=blur,
                              norm_type=NormType.Spectral))
            in_c = out_c

        embed_dim = nf // 2
        self.last_shuf = CustomPixelShuffle_ICNR(
            embed_dim, embed_dim, blur=blur, norm_type=last_norm, scale=4)

        # Determine in_channels for color decoder from the decoder layer output dims
        color_in_channels = [nf, nf, nf // 2]
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=color_in_channels, num_queries=num_queries,
            num_scales=num_scales, dec_layers=dec_layers)

    def forward(self):
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        out3 = self.last_shuf(out2)
        return self.color_decoder([out0, out1, out2], out3)


# ---------------------------------------------------------------------------
# ImageEncoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    def __init__(self, encoder_name, hook_names):
        super().__init__()
        if encoder_name == "convnext-t":
            self.arch = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        elif encoder_name == "convnext-l":
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError(f"Unknown encoder: {encoder_name}")
        self.hooks = [Hook(self.arch._modules[name]) for name in hook_names]

    def forward(self, x):
        return self.arch(x)


# ---------------------------------------------------------------------------
# DDColor main model
# ---------------------------------------------------------------------------

class DDColor(nn.Module):
    def __init__(self, encoder_name="convnext-l", decoder_name="MultiScaleColorDecoder",
                 num_input_channels=3, input_size=(256, 256), nf=512,
                 num_output_channels=3, last_norm="Weight", do_normalize=False,
                 num_queries=256, num_scales=3, dec_layers=9, **kwargs):
        super().__init__()

        self.encoder = ImageEncoder(encoder_name, ["norm0", "norm1", "norm2", "norm3"])
        self.encoder.eval()

        # Run a dummy forward to populate hook features (needed for decoder init)
        with torch.no_grad():
            self.encoder(torch.randn(1, num_input_channels, *input_size))

        last_norm_type = getattr(NormType, last_norm, NormType.Spectral)
        self.decoder = DuelDecoder(
            self.encoder.hooks, nf=nf, last_norm=last_norm_type,
            num_queries=num_queries, num_scales=num_scales, dec_layers=dec_layers)

        self.refine_net = nn.Sequential(
            custom_conv_layer(num_queries + 3, num_output_channels, ks=1,
                              use_activ=False, norm_type=NormType.Spectral)
        )

        self.do_normalize = do_normalize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.normalize(x)
        self.encoder(x)
        out_feat = self.decoder()
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)
        if self.do_normalize:
            out = self.denormalize(out)
        return out
