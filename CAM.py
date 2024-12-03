"""
------------------------------------
# @FileName    :111.py
# @Time        :2024/10/9 15:05
# @Author      :xieyuanzuo
# @description :此代码生成热力图但是在测试的数据集上batchsize必须设置为1，才能够将数据集完全生成
必须在输出的文件夹下提前建立一个fea文件
------------------------------------
"""
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import prettytable
import time
import os
import multiprocessing.pool as mpp
import multiprocessing as mp
from torchvision import transforms
from train import *

import argparse
from utils.config import Config
from tools.mask_convert import mask_save
def get_args():
    parser = argparse.ArgumentParser('description=Change detection of remote sensing images')
    parser.add_argument("-c", "--config", type=str, default="configs/LDNet_Back.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/CLCD_BS4_epoch200/LDNet_Back/version_5_backbone_LEVIRCD_消融_无直接进入编码器/ckpts/test/test_change_f1/last.ckpt")
    parser.add_argument("--output_dir", type=str, default="output/12")
    return parser.parse_args()
def show_feature_map(feature_map,heatmap_path):
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])
    # print(feature_map.shape)

    # feature_map = torch.mean(feature_map, dim=0)  # 压缩成[1,256,256],通道数变为1
    # print(feature_map.shape)
    # 以下4行，通过双线性插值的方式改变保存图像的大小(没必要)
    # feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    # upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    # feature_map = upsample(feature_map)
    # feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])  #[64,256,256]
    # print(feature_map.shape)

    feature_map_num = feature_map.shape[0] # 返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8

    plt.figure()

    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
            #
            # plt.subplot(int(row_num), int(row_num), index)
            # # plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
            # # 将上行代码替换成，可显示彩色
            # plt.imshow(transforms.ToPILImage()(feature_map[index - 1].cpu().numpy()),cmap='jet')#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        plt.imsave(heatmap_path, feature_map[index - 1].cpu().numpy(), cmap='jet')




if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = cfg.test_ckpt_path
    assert ckpt is not None

    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.dirname(ckpt)
    masks_output_dir = os.path.join(base_dir, "fea")

    # 加载模型并设置为评估模式
    model = myTrain.load_from_checkpoint(ckpt, cfg=cfg)
    model = model.to('cuda')
    model.eval()
    batch_idx = 0
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='test')
        # 打印测试集的样本个数
        for input in tqdm(test_loader):
            img_in1, img_in2 = input[0].cuda(), input[1].cuda()
            # # 获取模型的输出和特征图
            raw_predictions, img_id = model(img_in1, img_in2), input[3]
            output = raw_predictions[-1]  # 修改模型支持特征返回
            # print(output.shape)

            for i in range(raw_predictions.shape[0]):
                heatmap_path = os.path.join(masks_output_dir, f"{img_id[i]}.png")
                show_feature_map(output, heatmap_path)






