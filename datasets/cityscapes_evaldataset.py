# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityscapesEvalDataset(MonoDataset):
    """Cityscapes评估数据集 - 在此我们加载的是原始图像，而不是预处理过的三联图像，因此裁剪操作需要在get_color方法中完成。
    """
    RAW_HEIGHT = 1024  # 设置原始图像的高度为1024像素
    RAW_WIDTH = 2048  # 设置原始图像的宽度为2048像素

    def __init__(self, *args, **kwargs):
        # 调用父类MonoDataset的初始化方法
        super(CityscapesEvalDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """
        将数据集中的索引转换为文件夹名、帧索引及其他相关信息。

        该方法从数据集索引中获取城市名称和帧名，并返回相应的信息。

        示例txt文件格式：
            aachen aachen_000000 4
        """
        city, frame_name = self.filenames[index].split()  # 从文件名中分割出城市名称和帧名
        side = None  # 对于评估数据集，不涉及图像的“侧面”信息
        return city, frame_name, side  # 返回城市名称、帧名称和侧面信息

    def check_depth(self):
        return False

    def load_intrinsics(self, city, frame_name):
        # 该方法用于加载相机内参，来自sfmlearner的改编版本
        split = "test"  # 根据当前的训练状态设置数据集划分，通常为"test"或"val"
        # 通过拼接路径获取相机内参文件的路径
        camera_file = os.path.join(self.data_path, 'camera',
                                   split, city, frame_name + '_camera.json')
        # 打开并读取相机内参文件
        with open(camera_file, 'r') as f:
            camera = json.load(f)
        # 从相机内参文件中提取fx、fy、u0和v0等值
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']
        # 创建4x4的相机内参矩阵
        intrinsics = np.array([[fx, 0, u0, 0],
                               [0, fy, v0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).astype(np.float32)
        # 将内参矩阵的像素坐标归一化到图像的实际尺寸
        intrinsics[0, :] /= self.RAW_WIDTH  # 宽度归一化
        intrinsics[1, :] /= self.RAW_HEIGHT * 0.75  # 高度归一化（考虑到裁剪）
        return intrinsics  # 返回相机内参矩阵

    def get_color(self, city, frame_name, side, do_flip, is_sequence=False):
        # 该方法用于加载城市景观（Cityscapes）数据集中的颜色图像，并进行裁剪与预处理
        # 如果传入了side，抛出错误，因为当前版本的数据集不支持对不同视角（side）的处理
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides yet")
        # 使用loader加载指定路径的图像
        color = self.loader(self.get_image_path(city, frame_name, side, is_sequence))
        # 获取图像的宽度和高度
        w, h = color.size
        # 裁剪图像，只保留顶部的3/4部分，以适应Cityscapes数据集的尺寸
        crop_h = h * 3 // 4
        color = color.crop((0, 0, w, crop_h))
        return color  # 返回处理后的图像

    def get_offset_framename(self, frame_name, offset=2):
        # 该方法用于计算相对当前帧的偏移帧名称
        # 参数frame_name是当前帧的名称，格式为 city_seq_frame_num
        # 参数offset是偏移量，默认为-2，表示获取前两帧。如果为正数，则表示获取后几帧
        # 将当前帧的名称按照'_'分割，获取城市(city)、序列(seq)和帧编号(frame_num)
        city, seq, frame_num = frame_name.split('_')
        # 根据给定的偏移量，调整帧编号
        frame_num = int(frame_num)
        # 确保帧编号为6位数字，不足时在前面填充0
        frame_num = str(frame_num).zfill(6)
        # 返回新的帧名称，格式为 city_seq_frame_num
        return '{}_{}_{}'.format(city, seq, frame_num)

    def get_colors(self, city, frame_name, side, do_flip):
        # 该方法用于获取当前帧以及前两帧的图像数据，并存储在字典中返回
        # 如果side参数不为None，抛出错误，因为当前的Cityscapes数据集不处理不同视角（side）
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")
        # 获取当前帧的图像数据
        color = self.get_color(city, frame_name, side, do_flip)
        # 获取前两帧的图像数据
        prev_name = self.get_offset_framename(frame_name, offset=-2)
        prev_color = self.get_color(city, prev_name, side, do_flip, is_sequence=True)
        # 将当前帧和前两帧的图像数据存储到字典inputs中
        inputs = {}
        inputs[("color", 0, -1)] = color  # 当前帧的图像
        inputs[("color", -1, -1)] = prev_color  # 前两帧的图像
        # 返回包含图像数据的字典
        return inputs

    #获得数据的部分
    def get_image_path(self, city, frame_name, side, is_sequence=False):
        # 该方法用于构造并返回给定城市、帧名称、视角（side）和是否为序列（is_sequence）情况下的图像路径
        # 如果是序列图像，使用"leftImg8bit_sequence"文件夹，否则使用"leftImg8bit"文件夹
        # 这里注释掉了选择不同文件夹的代码，默认只处理"leftImg8bit_sequence"文件夹
        folder = "leftImg8bit"
        # 设定数据集的拆分方式，当前设置为"test"，表示测试集
        split = "test"
        # 构造图像路径，路径格式为：data_path/leftImg8bit_sequence/test/城市名/帧名称_leftImg8bit.png
        image_path = os.path.join(
            self.data_path, folder, split, city, frame_name + '_leftImg8bit.png')
        # 返回构造好的图像路径
        return image_path
