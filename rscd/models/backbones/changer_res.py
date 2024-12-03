
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
------------------------------------
# @FileName    :Changer-CD.py
# @Time        :2024/9/6 19:57
# @Author      :xieyuanzuo
# @description :
------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import random
from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class TwoIdentity(nn.Module):
    """简单的交互层，仅做占位符，不做任何改变"""

    def __init__(self):
        super(TwoIdentity, self).__init__()

    def forward(self, x1, x2):
        return x1, x2

class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
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

class SA(nn.Module):
    def __init__(self,in_channels=64, kernel_size=7):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(
            1,  # 输入通道数
            in_channels,  # 输出通道数与输入通道一致
            kernel_size=1,  # 1x1卷积
            bias=False
        )
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.conv1x1(x)

        return x

class SpatialExchange(nn.Module):
    """空间交换模块"""

    def __init__(self, p=1 / 2):
        super(SpatialExchange,self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class ChannelExchange(nn.Module):
    """通道交换模块"""

    def __init__(self, p=1 / 2):
        super(ChannelExchange,self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class IA_ResNet(nn.Module):
    """交互式ResNet模型"""

    def __init__(self,
                 interaction_cfg=(None, None, None, None),
                 spa_config=(None,None,None,None),
                 depth=18,
                 num_classes=1000,
                 out_indices=(0, 1, 2, 3),
                 pretrained=False):
        super(IA_ResNet, self).__init__()

        # 加载基础的ResNet
        if depth == 18:
            if pretrained:
                weights = ResNet18_Weights.DEFAULT
            else:
                weights = None
            self.resnet = resnet18(weights=weights)
        elif depth == 50:
            if pretrained:
                weights = ResNet50_Weights.DEFAULT
            else:
                weights = None
            self.resnet = resnet50(weights=weights)
        else:
            raise ValueError("Unsupported depth: choose from [18, 50]")
        self.out_indices = out_indices

        # 确保交互配置与网络阶段数一致
        assert len(interaction_cfg) == 4, '交互配置的长度必须等于4'

        # 构建交互层
        self.ccs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg is None:
                ia_cfg = TwoIdentity()  # 默认使用两个相同的特征直接传递
            else:
                ia_type = ia_cfg['type']
                if ia_type == 'SpatialExchange':
                    ia_cfg = SpatialExchange(p=ia_cfg.get('p', 0.5))
                elif ia_type == 'ChannelExchange':
                    ia_cfg = ChannelExchange(p=ia_cfg.get('p', 0.5))
            self.ccs.append(ia_cfg)
        self.ccs = nn.ModuleList(self.ccs)
        # 构建空间注意力层
        self.spp = []
        for ia_cfg in spa_config:
            if ia_cfg is None:
                ia_cfg = TwoIdentity()  # 默认使用两个相同的特征直接传递
            else:
                ia_type = ia_cfg['type']
                if ia_type == 'SA':
                    ia_cfg = SA(in_channels=ia_cfg.get('c', 64))
                elif ia_type == 'CA':
                    ia_cfg = CA()
            self.spp.append(ia_cfg)
        self.spp = nn.ModuleList(self.spp)
    def _forward_resnet_stage(self, x, stage):
        # 获取resnet的层名，并通过这些层名获取对应的子模块
        layers = {
            0: [self.resnet.layer1],
            1: [self.resnet.layer2],
            2: [self.resnet.layer3],
            3: [self.resnet.layer4],
            4: [self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool]
        }
        if stage in layers:
            for layer in layers[stage]:
                x = layer(x)
        return x

    def forward(self, x1, x2):
        outs = []
        x1 = self._forward_resnet_stage(x1, 4)
        x2 = self._forward_resnet_stage(x2, 4)
        # 逐阶段处理输入并交互
        for i in range(4):
            x1 = self._forward_resnet_stage(x1, i)
            x2 = self._forward_resnet_stage(x2, i)
            # 使用交互层对两个输入进行处理
            x1, x2 = self.ccs[i](x1, x2)
            if i == 0:
                x1 = self.spp[i](x1)
                x2 = self.spp[i](x2)
            else:
                x1, x2 = self.spp[i](x1, x2)
            if i in self.out_indices:
                # 直接输出每个阶段的两个特征图
                outs.append(x1)
                outs.append(x2)

        return tuple(outs)

# 测试模型
if __name__ == '__main__':
    model = IA_ResNet(depth=18, interaction_cfg=(
        None,
        dict(type='SpatialExchange', p=1 / 2),
        None,
        dict(type='ChannelExchange', p=1 / 2),
       ),
        spa_config=(
            dict(type='SA', c=64),
            None,
            None,
            None,
        ))

    x1 = torch.randn(4, 3, 256, 256)
    x2 = torch.randn(4, 3, 256, 256)

    outputs = model(x1, x2)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    print(outputs[3].shape)
    print(outputs[4].shape)
    print(outputs[5].shape)
    print(outputs[6].shape)
    print(outputs[7].shape)
    # for out in outputs:
    #     print(out.shape)
