"""
Inner_data bs=16  worker = 8  train
Inner_data bs=8  worker = 8
Inner_data bs=16  worker = 8  val
Inner_data bs=8  worker = 8
Inner_data bs=16  worker = 8  test
Inner_data bs=8  worker = 8

LEVIR_CD bs=16  worker = 10  train 14s
LEVIR_CD bs=8  worker = 10          16s
LEVIR_CD bs=16  worker = 10  val
LEVIR_CD bs=8  worker = 10
LEVIR_CD bs=16  worker = 10  test
LEVIR_CD bs=8  worker = 10

WHU_CD bs=16  worker = 10  train  6s
WHU_CD bs=8  worker = 10          6.5s
WHU_CD bs=16  worker = 10  val
WHU_CD bs=8  worker = 10
WHU_CD bs=16  worker = 10  test
WHU_CD bs=8  worker = 10


"""


import os
from PIL import Image
import multiprocessing as mp
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from time import time

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据集的根目录（包含 A, B 和 labels 文件夹）
        :param transform: 数据预处理变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.labels = []

        # 遍历文件夹，生成图像路径对和标签路径
        A_dir = os.path.join(root_dir, "A")
        B_dir = os.path.join(root_dir, "B")
        labels_dir = os.path.join(root_dir, "label")

        for file_name in sorted(os.listdir(A_dir)):
            A_path = os.path.join(A_dir, file_name)
            B_path = os.path.join(B_dir, file_name)
            label_path = os.path.join(labels_dir, file_name)

            if os.path.exists(A_path) and os.path.exists(B_path) and os.path.exists(label_path):
                self.image_pairs.append((A_path, B_path))
                self.labels.append(label_path)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        获取单个样本
        """
        A_path, B_path = self.image_pairs[idx]
        label_path = self.labels[idx]

        # 读取图像
        A_image = Image.open(A_path).convert("RGB")
        B_image = Image.open(B_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # 单通道

        if self.transform:
            A_image = self.transform(A_image)
            B_image = self.transform(B_image)
            label = transforms.ToTensor()(label)  # 转换为张量

        return A_image, B_image, label


def main():
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整大小
        transforms.ToTensor(),         # 转为张量
    ])

    # 加载数据集
    train_dataset = ChangeDetectionDataset(root_dir="data/Inner_data/val", transform=transform)

    # 使用 DataLoader
    print(f"num of CPU: {mp.cpu_count()}")
    for i in  range(1,5):
        for num_workers in range(6, 16, 2):
            train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, num_workers=num_workers, batch_size=8, pin_memory=True,drop_last=False
            )
            start = time()

            for epoch in range(1, 3):
                for i, data in enumerate(train_loader, 0):
                    pass
            end = time()

            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


# 必须加上这部分
if __name__ == "__main__":
    main()
