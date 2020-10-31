# Text Sentiment Classifification

任务介绍：训练模型对 Twitter上的推文 进行情绪分类

## 1、数据集介绍

**数据集**为 Twitter 上收集到的推文，每则推文都会被标注为正面或负面，如：

[![BFAoXn.png](https://s1.ax1x.com/2020/10/22/BFAoXn.png)](https://imgchr.com/i/BFAoXn)

除了 labeled data 以外，还额外提供了 120 万条左右的 unlabeled data

● labeled training data ：20万

● unlabeled training data ：120万 

● testing data ：20万



## 2、模型介绍

### word embedding

采用 skip-gram 的方法在 unlabeled training data 上训练word2vec词向量

### 模型

采用了双向的GRU

### 训练方式（Semi-supervised Learning）

* 先用 labeled training data 训练一个模型
* 用训练好的模型在unlabeled training data进行预测
* 选取一部分unlabeled training data加入到训练集中，重复进行训练
  * 挑选规则是：预测大于0.6的作为正类，小于0.4的作为负类
* 重复2、3步

## 3、结果

提交到[kaggle](https://www.kaggle.com/c/ml2020spring-hw4)上，预测结果为：0.77514

