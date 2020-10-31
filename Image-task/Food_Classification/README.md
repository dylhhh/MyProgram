# Food Classifification

## 1、任务描述

* 对食物照片进行分类，共有11类：Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.

* 数据集为在网上收集的食物照片：
  * Training set: 9866张
  * Validation set: 3430张
  * Testing set: 3347张



## 2、模型介绍

模型采用了ResNet结果：

* 图片处理为（3，128，128）的格式
* 经过 卷积层 + 4个残差块 + 全连接层
* 最后得到11维的向量，采用交叉熵作为损失函数

## 3、运行
直接运行run.py文件
