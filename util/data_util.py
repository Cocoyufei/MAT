import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)  #该函数使用 SA.create 创建一个与 var 形状和数据类型相同的数组 x
    x[...] = var[...]  #然后将 var 数组的所有元素复制到 x 中，并将 x 的 writeable 属性设置为 False,以确保不能更改 x 数组中的值。
    x.flags.writeable = False
    return x


def collate_fn(batch):  #这是一个用于数据加载器的 collate_fn 函数，它接收一个批次的数据 batch，
    coord, feat, label = list(zip(*batch))  # 该函数首先使用 zip 函数将 batch 中的三个部分分别打包为 coord、feat 和 label，
    offset, count = [], 0
    for item in coord:  #然后使用 for 循环遍历 coord 中的每个样本，并记录下每个样本的坐标数量。这是为了在下一步中将所有样本的坐标连接成一个大的坐标张量时，能够知道每个样本的坐标在新张量中的位置。
        count += item.shape[0]  #
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset) #接下来，该函数使用 torch.cat 函数将所有样本的坐标张量、特征张量和标签张量拼接成一个大的张量，并使用 torch.IntTensor 函数将 offset 列表转换为整数张量。

#这是一个用于数据预处理的函数 data_prepare
def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform: #如果传入了变换函数 transform，该函数将使用它对坐标、特征和标签进行变换。
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:   #如果传入了 voxel_size，该函数将使用 voxelize 函数将坐标点云进行体素化，并将特征和标签对应的坐标点也进行相应变换。
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:   #如果传入了 voxel_max，且点云的数量超过了 voxel_max，该函数将随机选择一个点作为初始点，并使用 np.argsort 函数对每个点与初始点之间的距离进行排序，然后选择距离最近的前 voxel_max 个点作为点云的子集
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index: #如果传入了 shuffle_index，该函数将使用 np.random.shuffle 函数随机打乱点云的顺序。 该函数最终将坐标、特征和标签转换为 PyTorch 张量，并将特征张量除以 255 进行归一化处理，最后返回处理后的坐标、特征和标签张量。
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
