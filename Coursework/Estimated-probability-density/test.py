import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#用subplot()方法绘制多幅图形
plt.figure(figsize=(6,6),dpi=80)
#创建第一个画板
plt.figure(1)
#将第一个画板划分为2行1列组成的区块，并获取到第一块区域
ax1 = plt.subplot(211)

#在第一个子区域中绘图
plt.scatter([1,3,5],[2,4,5],marker="v",s=50,color="r")
#选中第二个子区域，并绘图
ax2 = plt.subplot(212)
plt.plot([2,4,6],[7,9,15])


#创建第二个画板
plt.figure(2)
x = np.arange(4)
y = np.array([15,20,18,25])
#在第二个画板上绘制柱状图
plt.bar(x,y)
#为柱状图添加标题
plt.title("第二个画板")

#切换到第一个画板
plt.figure(1)

#为第一个画板的第一个区域添加标题
ax1.set_title("第一个画板中第一个区域")
ax2.set_title("第一个画板中第二个区域")

# 调整每隔子图之间的距离
plt.tight_layout()
plt.show()