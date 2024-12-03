#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
------------------------------------
# @FileName    :1.py
# @Time        :2024/11/18 19:57
# @Author      :xieyuanzuo
# @description :
------------------------------------
"""
import os


def delete_different_images(folder1, folder2):
    # 获取两个文件夹中的所有文件名
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 找到两个文件夹中的不同文件名
    different_files_in_folder1 = files1 - files2
    different_files_in_folder2 = files2 - files1

    # 删除 folder1 中不同的文件
    for file in different_files_in_folder1:
        file_path = os.path.join(folder1, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted from {folder1}: {file}")

    # 删除 folder2 中不同的文件
    for file in different_files_in_folder2:
        file_path = os.path.join(folder2, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted from {folder2}: {file}")


# 使用示例
folder1 = "DSFIN/train/B"
folder2 = "DSFIN/train/label"
delete_different_images(folder1, folder2)
