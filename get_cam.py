#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
------------------------------------
# @FileName    :get_cam.py
# @Time        :2024/10/26 10:20
# @Author      :xieyuanzuo
# @description :
------------------------------------
"""
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from utils.config import Config
from train import myTrain, build_dataloader
import argparse

net = 'net.decoderhead.lightdecoder'

def get_args():
    parser = argparse.ArgumentParser(description="Change detection of remote sensing images")
    parser.add_argument("-c", "--config", type=str, default="configs/LDNet_Back.py")
    parser.add_argument("--ckpt", type=str,
                        default="work_dirs/CLCD_BS4_epoch200/LDNet_Back/version_5_backbone_LEVIRCD_消融_无直接进入编码器/ckpts/test/test_change_f1/last.ckpt")
    parser.add_argument("--output_dir", type=str, default="output/12")
    parser.add_argument("--layer_name", type=str, required=False, default=net,help='fram')
    return parser.parse_args()


def register_hook(layer, features):
    """ 注册钩子来捕获指定层的输出特征 """
    def hook(module, input, output):
        features.append(output)
    layer.register_forward_hook(hook)

def show_feature_map(feature_map, heatmap_path):
    feature_map = feature_map.squeeze(0)  # [64, 55, 55]
    # print(feature_map.shape)
    # feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    # upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    # feature_map = upsample(feature_map)
    # feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])  #[64,256,256]
    # feature_map = torch.mean(feature_map, dim=0).unsqueeze(0) # 压缩成[1,256,256],通道数变为1
    # # print(feature_map.shape)
    feature_map_num = feature_map.shape[0] # 通道数
    row_num = np.ceil(np.sqrt(feature_map_num)).astype(int)  # 行数

    plt.figure()
    for idx in range(0,feature_map_num-1):
        plt.subplot(row_num, row_num, idx + 1)
        plt.imshow(feature_map[idx].cpu().numpy(), cmap='jet')
        plt.axis('off')

    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)
    ckpt = args.ckpt or cfg.test_ckpt_path
    assert ckpt, "未提供检查点路径"

    base_dir = args.output_dir or os.path.dirname(ckpt)
    masks_output_dir = os.path.join(base_dir, net)
    os.makedirs(masks_output_dir, exist_ok=True)

    # 加载模型并设置为评估模式
    model = myTrain.load_from_checkpoint(ckpt, cfg=cfg).to('cuda').eval()
    for name, module in model.named_modules():
        print(name)
    # 注册钩子捕获指定层的特征图
    features = []
    layer = dict([*model.named_modules()])[args.layer_name]
    register_hook(layer, features)

    # 处理测试集
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='test')
        for batch_idx, input in enumerate(tqdm(test_loader)):
            img_in1, img_in2, img_id = input[0].cuda(), input[1].cuda(), input[3]
            model(img_in1, img_in2)  # 前向传播，钩子会捕获特征图
            # 保存捕获的特征图
            for i in range(min(len(features), len(img_id))):
                heatmap_path = os.path.join(masks_output_dir, f"{img_id[i]}.png")
                show_feature_map(features[i], heatmap_path)

                # heatmap_path = os.path.join(masks_output_dir, f"{img_id[i]}.png")
                # show_feature_map(feature_map, heatmap_path)
                # print(wwwwwwwww)
            # 清空特征缓存，避免重复叠加
            features.clear()
