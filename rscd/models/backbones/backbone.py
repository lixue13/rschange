import torch 
import torch.nn as nn
from rscd.models.backbones.seaformer import *
from rscd.models.backbones.resnet import get_resnet18, get_resnet50_OS32, get_resnet50_OS8
from rscd.models.backbones.swintransformer import *
from rscd.models.backbones.changer_res import *
from rscd.models.mambnet.ex_vssd import Backbone_VMAMBA2
from rscd.models.backbones.msanet_backbone import build_backbone

class Base(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'Seaformer':
            self.backbone = SeaFormer_L(pretrained=True)
        elif name == 'Resnet18':
            self.backbone = get_resnet18(pretrained=True)
        elif name == 'Swin':
            self.backbone = swin_tiny(True)
        elif name ==  'changer_res':
            self.backbone = IA_ResNet(pretrained=False)
        elif name == 'Backbone_VSSM':
            self.backbone = Backbone_VMAMBA2(linear_attn_duality=False, ssd_chunk_size=16)
        elif name == 'build_backbone':
            self.backbone = build_backbone('resnet18',output_stride=16,BatchNorm=nn.BatchNorm2d,in_c=3)

    def forward(self, xA, xB):
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)

        return [featuresA, featuresB]

class Base1(nn.Module):
    def __init__(self, name):
        super().__init__()

        if name ==  'changer_res':
            self.backbone = IA_ResNet(depth=18,
                                      pretrained = True,
                                      interaction_cfg=(
        None,
        dict(type='ChannelExchange', p=1 / 2),
        dict(type='ChannelExchange', p=1 / 2),
        dict(type='ChannelExchange', p=1 / 2),

       ),
        spa_config=(
            dict(type='SA', c=64),
            None,
            None,
            None,
        ))

    def forward(self, xA, xB):
        features = self.backbone(xA,xB)


        return features
