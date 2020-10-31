import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import time


# load data
def read_file(path, lable=True):
    path_dir = os.listdir(path)
    x = np.zeros((len(path_dir), 128,128, 3))
    y = np.zeros(len(path_dir))
    for i, f in enumerate(path_dir):
        img = cv2.imread(os.path.join(path,f))
        x[i, :, :] = cv2.resize(img, (128,128))
        if lable:
            y[i] = int(f.split("_")[0])
    if lable:
        return x, y
    return x


# 制作数据集
class ImageDataset(Dataset):
    def __init__(self, transformer, x, y=None):
        self.x = x
        self.y = y
        self.transformer = transformer
        if y is not None:
            self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        X = self.x[item]
        if self.transformer:
            X = self.transformer(X)
        if self.y:
            Y = self.y[item]
            return X, Y
        else:
            return X
