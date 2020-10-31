# Image clustering

## 任务目标

判定给定的图片是否为风景

给定训练集的图片都是 32\* 32 \* 3的，没有任何lable

## 数据集

trainX.npy：8500张RGB图片，shape为：（8500，32，32，3）

valX.npy：500张RGB图片，shape为：（500，32，32，3）

valY.npy：valX.npy对应的lable，shape为：（500，）

**注：**valX.npy、valY.npy是验证模型性能的，非训练集

## 模型

* 先使用autoencoder模型对图片进行编码
  * 包含2部分encoder、decoder，2部分是对应着的
  * encoder：
    * 3个卷积层：Conv2d(3, 64, 3, 1, 1)、Conv2d(64, 128, 3, 1, 1)、Conv2d(128, 256, 3, 1, 1)
    * 3个池化层：MaxPool2d(2， return_indices=True)
  * decoder：
    * 3个逆卷积层：ConvTranspose2d(256, 128, 3, 1, 1)、ConvTranspose2d(128, 64, 3, 1, 1)、ConvTranspose2d(64, 3, 3, 1, 1)
    * 3个逆池化层：MaxUnpool2d(2, 2)

* 将图片拉成一维向量，使用PCA降维2次，先后降维到500、200维
* 用TSEN降到2维
* 计算准确率：训练50轮，平均准确率为0.65

