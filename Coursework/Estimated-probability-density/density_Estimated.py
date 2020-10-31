#  非参数估计概率密度
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

class DensityEstimated:
    def __init__(self, N, h):
        self.n = N  # 样本数
        self.k_n = int(math.sqrt(N))  # (Kn近邻估计使用的)每个小舱固定的样本数
        self.h = h  # (parzen窗使用）每个小舱的棱长（方窗）
        self.sigma = h/int(math.sqrt(N))  # 高斯窗的方差
        self.data = []

    # 随机产生满足要求的正态分布的一维样本点
    def generateRandData(self, mean=0, sigma=0.1):
        self.data = np.random.normal(mean, sigma, self.n)
        self.data = sorted(self.data)
        return self.data

    # Kn近邻估计法
    def KnEstimated(self):

        p = []
        for i in self.data:
            # 计算点i 到其他所有点的距离
            dist = []
            for j in self.data:
                dist.append(abs(i - j))  # 一维数据距离
            dist = sorted(dist)  # 排序，离x最近的kn个点 dist[0:k_n+1]
            # 小舱体积为：2dist[self.k_n]
            p.append(self.k_n/self.n/(2*dist[self.k_n]))

        return p

    # Parzen窗估计：
    def parzenEstimated(self):

        p_Gaussian = []
        p_square = []
        for i in self.data:
            p1, p2 = 0, 0
            for j in self.data:
                # 高斯窗
                p1 += 1/(math.sqrt(2*3.14)*self.sigma) * math.exp(-(i-j)**2/(2*self.sigma**2))
                # 方窗
                if abs(i-j) <= self.h/2:
                    p2 += 1/self.h
            p_Gaussian.append(p1/self.n)
            p_square.append(p2/self.n)
        return p_Gaussian, p_square

if __name__ == '__main__':
    solu = DensityEstimated(256, 0.1)
    data = solu.generateRandData()
    p_k = solu.KnEstimated()
    p_Gaussian, p_square = solu.parzenEstimated()

    # 解决中文显示问题
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    ax1 = plt.subplot(211)
    plt.plot(data, p_k)
    plt.hist(data, bins=100, density=True)

    ax2 = plt.subplot(223)
    plt.plot(data, p_Gaussian)
    plt.hist(data, bins=100, density=True)

    ax3 = plt.subplot(224)
    plt.plot(data, p_square)
    plt.hist(data, bins=100, density=True)

    ax1.set_title('kn近邻: k_n = 16')
    ax2.set_title('高斯窗: h = 0.1')
    ax3.set_title('方窗: h = 0.1')
    # 调整每隔子图之间的距离
    plt.tight_layout()
    plt.show()
