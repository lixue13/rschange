#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
------------------------------------
# @FileName    :MmbNet.py
# @Time        :2024/11/14 14:30
# @Author      :xieyuanzuo
# @description :
------------------------------------
"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
------------------------------------
# @FileName    :CMNet.py
# @Time        :2024/7/30 10:53
# @Author      :xieyuanzuo
# @description :
------------------------------------
"""
import sys
import math

sys.path.append('rscd')

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from rscd.models.mambnet.nd_mamba2 import NdMamba2_2d
Device = torch.device
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

class MFA(nn.Module):
    def __init__(self, plane):
        super(MFA, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, xA, xB):
        # b, c, h, w = x.size()
        node_k = self.node_k(xA)  # torch.Size([1, 64, 256, 256])
        node_v = self.node_v(xA)  # torch.Size([1, 64, 256, 256])
        node_q = self.node_q(xB)  # torch.Size([1, 64, 256, 256])

        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)

        # torch.Size([1, 64, 1024]) q (b,d,N)
        # torch.Size([1, 64, 1024]) v (b,d,n)
        # torch.Size([1, 1024, 64]) k （b,N,d）

        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_v, node_q)  # (1,1024,1024)

        AV = self.softmax(AV)  # (1,1024,1024)

        AV = torch.bmm(node_k, AV)  # torch.Size([1, 64, 1024])

        # AV = AV.transpose(1, 2).contiguous()     # torch.Size([1, 64, 1024])

        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)

        out = F.relu_(self.out(AVW) + xB)
        # print(out.shape)
        return out

class MFB(nn.Module):
    def __init__(self, plane):
        super(MFB, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, xA):
        # b, c, h, w = x.size()
        node_k = self.node_k(xA)  # torch.Size([1, 64, 256, 256])
        node_v = self.node_v(xA)  # torch.Size([1, 64, 256, 256])
        node_q = self.node_q(xA)  # torch.Size([1, 64, 256, 256])

        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # torch.Size([1, 65536, 64])
        # torch.Size([1, 64, 65536])
        # torch.Size([1, 65536, 64])

        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_v, node_q)  # (1,1024,1024)

        AV = self.softmax(AV)  # (1,1024,1024)

        AV = torch.bmm(node_k, AV)  # torch.Size([1, 64, 1024])

        # AV = AV.transpose(1, 2).contiguous()     # torch.Size([1, 64, 1024])

        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)

        out = F.relu_(self.out(AVW) + xA)
        # print(out.shape)
        return out

class TFAM(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel, out_channel=128):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)
        self.recover_conv = dsconv_3x3(in_channel=in_channel, out_channel=out_channel)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w
        fuse = self.recover_conv(fuse)

        return fuse


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图


class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B, C, h, w)
    key_feats: (B, C, h, w)
    value_feats: (B, C, h, w)

    output: (B, C, h, w)
    """

    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (B, h*w, C1)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (B, C1, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (B, h*w, C1)

        sim_map = torch.matmul(query, key)

        #  缩放点积相似度

        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # (B, h*w, K)将相似度矩阵转换为注意力权重矩阵

        context = torch.matmul(sim_map, value)  # (B, h*w, C1)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (B, C1, h, w)

        context = self.out_project(context)  # (B, C, h, w)
        return context

    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]


class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x


class CSFF(nn.Module):
    def __init__(self, in_place=128):
        super(CSFF, self).__init__()

        self.cbam = CBAM(in_place)

    def forward(self, x_small, x_big):
        img_shape = x_small.size(2), x_small.size(3)
        big_weight = self.cbam(x_big)
        big_weight = F.interpolate(big_weight, img_shape, mode="bilinear", align_corners=False)
        x_small = big_weight * x_small
        return x_small




# fca
def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    # MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
    # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
    # planes * 4 -> channel, c2wh[planes] -> dct_h, c2wh[planes] -> dct_w
    # (64*4,56,56)
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape  # (4,256,64,64)
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:  # dct_h=dct_w=56
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))  # (4,256,56,56)
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)  # y:(4,256)

        y = self.fc(y).view(n, c, 1, 1)  # y:(4,256,1,1)
        return x * y.expand_as(x)  # pytorch中的expand_as:扩张张量的尺寸至括号里张量的尺寸 (4,256,64,64)  注意这里是逐元素相乘，不同于qkv的torch.matmul


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    # MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):  # (4,256,56,56)
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight  # weight:(256,56,56)  x:(4,256,56,56)

        result = torch.sum(x, dim=[2, 3])  # result:(4,256)
        return result

    def build_filter(self, pos, freq, POS):  # 对应公式中i/j, h/w, H/W   一般是pos即i/j在变
        # self.build_filter(t_x, u_x, tile_size_x)  self.build_filter(t_y, v_y, tile_size_y)
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)  # 为什么是乘以根号2？

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        # dct_h(height), dct_w(weight), mapper_x, mapper_y, channel(256,512,1024,2048)
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)  # (256,56,56)

        c_part = channel // len(mapper_x)  # c_part = 256/16 = 16

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class LightDecoder(nn.Module):
    def __init__(self, in_channel, num_class):
        super(LightDecoder, self).__init__()
        self.catconv = conv_3x3(in_channel * 4, in_channel)
        self.decoder = nn.Conv2d(in_channel, num_class, 1)

    def forward(self, x1, x2, x3, x4):
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        x3 = F.interpolate(x3, scale_factor=4, mode="bilinear")
        x4 = F.interpolate(x4, scale_factor=8, mode="bilinear")
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        out = self.decoder(self.catconv(torch.cat([x1, x2, x3, x4], dim=1)))
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # print(identity.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class MambNet(nn.Module):
    def __init__(self, num_class, channel_list=[64, 128, 256, 512], transform_feat=128, layer_num=4):
        super(MambNet, self).__init__()

        self.layer_num = layer_num
        self.fe = dict([(64, 56), (128, 28), (256, 14), (512, 7)])

        self.Mamba1 = NdMamba2_2d(128, 128, 64)
        self.Mamba2 = NdMamba2_2d(128, 128, 64)
        self.Mamba3 = NdMamba2_2d(128, 128, 64)
        self.Mamba4 = NdMamba2_2d(128, 128, 64)

        self.TFAM1 = TFAM(channel_list[0], out_channel=128)
        self.TFAM2 = TFAM(channel_list[1], out_channel=128)
        self.TFAM3 = TFAM(channel_list[2], out_channel=128)
        self.TFAM4 = TFAM(channel_list[3], out_channel=128)

        # self.smooth_layer_0 = ResBlock(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1)
        # self.MFA1 = MFA(plane=128)
        # self.MFA2 = MFA(plane=128)
        # self.MFA3 = MFA(plane=128)
        # self.MFB1 = MFB(plane=128)
        # self.MFB2 = MFB(plane=128)
        # self.MFB3 = MFB(plane=128)

        self.catconv = conv_3x3(transform_feat * 4, transform_feat)
        self.lightdecoder = LightDecoder(transform_feat, num_class)

    def forward(self, x):
        # A1, A2, A3, A4, B1, B2, B3, B4 = x
        featuresA, featuresB = x
        A1, A2, A3, A4 = featuresA
        B1, B2, B3, B4 = featuresB
        # 时域融合
        x1 = self.TFAM1(A1, B1)
        x2 = self.TFAM2(A2, B2)
        x3 = self.TFAM3(A3, B3)
        x4 = self.TFAM4(A4, B4)
        # # print(x1.shape)
        x1 =  self.Mamba1(x1)
        x2 =  self.Mamba2(x2)
        x3 =  self.Mamba3(x3)
        x4 =  self.Mamba4(x4)
        #
        # x1 =  self.smooth_layer_3(x1)
        # x2 = self.smooth_layer_2(x2)
        # x3 = self.smooth_layer_1(x3)
        # x4 = self.smooth_layer_0(x4)


        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        #
        # x41 = F.interpolate(x4, scale_factor=2, mode="bilinear")
        #
        # x3_sel = self.MFB1(x3)
        # x3_new = self.MFA3(x41, x3)
        # x3_new = x3_new+x3_sel
        #
        # x31 = F.interpolate(x3, scale_factor=2, mode="bilinear")
        # x2_new = self.MFA2(x31, x2)
        # x2_sel = self.MFB2(x2)
        # x2_new = x2_new + x2_sel
        #
        # x21 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        # x1_new = self.MFA1(x21, x1)
        # x1_sel = self.MFB3(x1)
        # x1_new = x1_new + x1_sel
        #
        #
        # print(x1_new.shape)
        # print(x2_new.shape)
        # print(x3_new.shape)
        # x56 = torch.cat([x4,x1_new,x2_new,x3_new],dim=1)
        # print(x56.shape)
        # x4_new = self.catconv(torch.cat([x4, x1_new, x2_new, x3_new], dim=1))
        # print(x4_new.shape)

        out = self.lightdecoder(x1, x2, x3, x4)

        out = F.interpolate(out, scale_factor=4, mode="bilinear")

        return out

class MambNet1(nn.Module):
    def __init__(self, num_class, channel_list=[64, 128, 256, 512], transform_feat=128, layer_num=4):
        super(MambNet1, self).__init__()

        self.layer_num = layer_num
        self.fe = dict([(64, 56), (128, 28), (256, 14), (512, 7)])

        self.Mamba1 = NdMamba2_2d(128, 128, 64)
        self.Mamba2 = NdMamba2_2d(128, 128, 64)
        self.Mamba3 = NdMamba2_2d(128, 128, 64)
        self.Mamba4 = NdMamba2_2d(128, 128, 64)



        # self.TFAM1 = TFAM(channel_list[0], out_channel=128)
        # self.TFAM2 = TFAM(channel_list[1], out_channel=128)
        # self.TFAM3 = TFAM(channel_list[2], out_channel=128)
        # self.TFAM4 = TFAM(channel_list[3], out_channel=128)
        #
        # self.MFA1 = MFA(plane=128)
        # self.MFA2 = MFA(plane=128)
        # self.MFA3 = MFA(plane=128)
        # self.MFB1 = MFB(plane=128)
        # self.MFB2 = MFB(plane=128)
        # self.MFB3 = MFB(plane=128)

        self.catconv = conv_3x3(transform_feat * 4, transform_feat)
        self.lightdecoder = LightDecoder(transform_feat, num_class)

    def forward(self, x):
        # A1, A2, A3, A4, B1, B2, B3, B4 = x
        featuresA, featuresB = x
        A1, A2, A3, A4 = featuresA
        B1, B2, B3, B4 = featuresB


        # 时域融合
        x1 = self.TFAM1(A1, B1)
        x2 = self.TFAM2(A2, B2)
        x3 = self.TFAM3(A3, B3)
        x4 = self.TFAM4(A4, B4)
        # # print(x1.shape)
        x1 =  self.Mamba1(x1)
        x2 =  self.Mamba2(x2)
        x3 =  self.Mamba3(x3)
        x4 =  self.Mamba4(x4)


        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        #
        # x41 = F.interpolate(x4, scale_factor=2, mode="bilinear")
        #
        # x3_sel = self.MFB1(x3)
        # x3_new = self.MFA3(x41, x3)
        # x3_new = x3_new+x3_sel
        #
        # x31 = F.interpolate(x3, scale_factor=2, mode="bilinear")
        # x2_new = self.MFA2(x31, x2)
        # x2_sel = self.MFB2(x2)
        # x2_new = x2_new + x2_sel
        #
        # x21 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        # x1_new = self.MFA1(x21, x1)
        # x1_sel = self.MFB3(x1)
        # x1_new = x1_new + x1_sel
        #
        #
        # print(x1_new.shape)
        # print(x2_new.shape)
        # print(x3_new.shape)
        # x56 = torch.cat([x4,x1_new,x2_new,x3_new],dim=1)
        # print(x56.shape)
        # x4_new = self.catconv(torch.cat([x4, x1_new, x2_new, x3_new], dim=1))
        # print(x4_new.shape)

        out = self.lightdecoder(x1, x2, x3, x4)

        out = F.interpolate(out, scale_factor=4, mode="bilinear")

        return out

import thop
def test_cmnet():
    # Parameters
    num_class = 2
    channel_list = [64, 128, 256, 512]
    transform_feat = 128
    layer_num = 4

    # Create the STNet model
    model = MambNet(num_class, channel_list, transform_feat, layer_num)

    # Print model summary (optional)

    # print(model)
    # Generate dummy data
    batch_size = 2
    height, width = 512, 512

    xA1 = torch.randn(batch_size, 64, 128, 128)
    xA2 = torch.randn(batch_size, 128, 64, 64)
    xA3 = torch.randn(batch_size, 256, 32, 32)
    xA4 = torch.randn(batch_size, 512, 16, 16)
    xB1 = torch.randn(batch_size, 64, 128, 128)
    xB2 = torch.randn(batch_size, 128, 64, 64)
    xB3 = torch.randn(batch_size, 256, 32, 32)
    xB4 = torch.randn(batch_size, 512, 16, 16)

    # inputs = (xA1, xB1,xA2, xB2,xA3,xB3 ,xA4, xB4)
    inputs = (xA1, xA2, xA3, xA4, xB1, xB2, xB3, xB4)
    # Forward pass
    output = model(inputs)

    # Check output shape
    print(f"Output shape: {output.shape}")
    flops, params = thop.profile(model, inputs=(inputs,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    # 假设输入通道为 64，输出通道为 128，stride 为 2，使用 downsample
    in_channels = 64
    out_channels = 64
    stride = 1

    # 定义 downsample 模块
    downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
    )

    # 创建 ResBlock 实例
    block = ResBlock(in_channels, out_channels, stride=stride, downsample=None)

    # 打印模块结构
    print(block)

    # 创建随机输入
    x = torch.randn(1, in_channels, 32, 32)  # (batch_size, channels, height, width)

    # 前向传播
    output = block(x)

    # 输出检查
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 验证是否可以正常反向传播
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(block.parameters(), lr=0.01)

    # 假设目标输出
    target = torch.randn_like(output)
    loss = criterion(output, target)
    loss.backward()  # 反向传播

    optimizer.step()  # 更新参数

    print("Backward pass and parameter update successful!")

