# _*_ coding:utf-8 _*_
# by '林雪糕'
# time: 2022/10/11 10:46
# filename: test2.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def make_segments(x, y):
    '''
    利用x和y坐标创建线段列表，格式为LineCollection的正确格式。
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    绘制 x,y 的彩色线条
    '''
    #  z 参数为默认情况，颜色在[0,1]上等差排列
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    #  z 参数是单个数字，单色着色:
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def clear_frame(ax=None):
    '''
    隐藏坐标轴标签等内容，让图片看起来“干净”
    '''
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # 隐藏图形框的四条边框线
    for spine in ax.spines.values():
        spine.set_visible(False)

data=np.load('./traj.npy')  # 加载文件
num = data.shape[0]         # 每条线的数据量
line_num = data.shape[1]    # 线的条数

min_t = 1.5
max_t = 5                           # 哪边要粗，哪边要细呢？
t = np.linspace(min_t, max_t, num)  # 控制粗细
min_a = 0.2
max_a = 0.8
a = np.linspace(min_a, max_a, num)  # 控制透明度

fig, axes = plt.subplots()

for i in range(line_num):
    color = np.linspace(0.1*i,0.1*(i+1), num)
    line_data = data[:, i, :]
    colorline(data[:,i,0], data[:,i,1], z=color, alpha=a, linewidth=t, cmap='jet')

x_min = np.min(data[:,:,0])
x_max = np.max(data[:,:,0])
y_min = np.min(data[:,:,1])
y_max = np.max(data[:,:,1])
plt.xlim(x_min - 0.1, x_max + 0.1)
plt.ylim(y_min - 0.1, y_max + 0.1)
clear_frame()
plt.show()

