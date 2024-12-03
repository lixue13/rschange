import torch 
import torch.nn as nn
from rscd.models.backbones.resnet import get_resnet18, get_resnet50_OS32, get_resnet50_OS8
from rscd.models.backbones.swintransformer import *
from rscd.models.backbones.changer_res import *
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.backbone = get_resnet18(pretrained=True)

    def forward(self, xA, xB):
        _, xA1, xA2, xA3, xA4 = self.backbone(xA)
        _, xB1, xB2, xB3, xB4 = self.backbone(xB)

        return [xA1, xA2, xA3, xA4, xB1, xB2, xB3, xB4]
    
class Swin(nn.Module):
    def __init__(self):
        super(Swin, self).__init__()
        self.backbone = swin_tiny(True)

    def forward(self, xA, xB):
        xA1, xA2, xA3, xA4 = self.backbone(xA)
        xB1, xB2, xB3, xB4 = self.backbone(xB)

        return [xA1, xA2, xA3, xA4, xB1, xB2, xB3, xB4]

class ChangerRes(nn.Module):
    def __init__(self):
        super(ChangerRes, self).__init__()
        self.backbone = IA_ResNet(depth=18, interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=1 / 2),
            dict(type='ChannelExchange', p=1 / 2),
            dict(type='ChannelExchange', p=1 / 2)))

    def forward(self, xA, xB):
        outs = self.backbone(xA, xB)
        xA1, xA2, xA3, xA4 = outs[0], outs[2], outs[4], outs[6]
        xB1, xB2, xB3, xB4 = outs[1], outs[3], outs[5], outs[7]
        return [xA1, xA2, xA3, xA4, xB1, xB2, xB3, xB4]
