'''
 trainX.npy
○ 裡面總共有 8500 張 RGB 圖片，大小都是 32*32*3
○ shape 為 (8500, 32, 32, 3)
● valX.npy
○ 請不要用來訓練
○ 裡面總共有 500 張 RGB 圖片，大小都是 32*32*3
○ shape 為 (500, 32, 32, 3)
● valY.npy
○ 請不要用來訓練
○ 對應 valX.npy 的 label
○ shape為 (500,)
'''
import numpy as np
from torch.utils.data import Dataset
import random
import torch


# 数据预处理
# 将图片里pixel的值由[0,255]转为[-1,1]
def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list


# 数据集类
class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images


