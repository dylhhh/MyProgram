import numpy as np
import torch
from torch.utils.data import DataLoader

import data_process
import Model
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans


def test(X, model, batch_size=256):
    X = data_process.preprocess(X)  # 数据预处理
    dataset = data_process.Image_Dataset(X)  # 建立数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x).to(device)
        vec, img = model(x)
        # print(vec.shape, img.shape)
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
            # detach(）的主要用途是将有梯度的变量变成没有梯度的，即requires grad=True变成requires grad=False.
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis=0)
    print('Latents Shape:', latents.shape)
    return latents


def predict(latents):
    # 先用PCA降到500维
    transformer = KernelPCA(n_components=500, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # 再用PCA降到500维
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(kpca)
    print('First Reduction Shape:', kpca.shape)

    # 用TSEN降到2维
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Kmeans聚类
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

# 计算准确率
def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """

    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因为是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return acc, 1 - acc


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valX = np.load('./data/valX.npy')
    valY = np.load('./data/valY.npy')

    model = Model.autoencoder().to(device)
    model.load_state_dict(torch.load('./model/checkpoint_40.pth'))

    latent = test(valX, model)
    pred, Xembedding = predict(latent)

    a, b = cal_acc(valY, pred)
    print(a, b)
