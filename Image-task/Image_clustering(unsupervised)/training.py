import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data_process
from Model import autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读data
trainX = np.load('./data/trainX.npy')
trainX_preprocessed = data_process.preprocess(trainX)
img_dataset = data_process.Image_Dataset(trainX_preprocessed)
img_dataloader = DataLoader(img_dataset, 64, shuffle=True)
print("加载训练集：", len(trainX))

model = autoencoder().to(device)
loss = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
epochs = 50

print("开始训练...")
max_loss = 0
for epoch in range(epochs):
    epoch_loss = 0

    for img in img_dataloader:
        img = img.to(device)
        _, output = model(img)  # 预测
        loss_ = loss(output, img)  # 计算重构的图片和原始图片的loss值
        optim.zero_grad()  # 计算梯度之前先将上一次的梯度清0
        loss_.backward()  # 计算梯度
        optim.step()  # 参数更新

        epoch_loss += loss_.item()
    print('epoch:{},loss:{}'.format(epoch, epoch_loss))
    torch.save(model.state_dict(), './model/checkpoint_{}.pth'.format(epoch + 1))
