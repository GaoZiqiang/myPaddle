import matplotlib.pyplot as plt

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer
# %matplotlib inline 我把这里进行了注释，不影响使用

with fluid.dygraph.guard():
    # 创建初始化权重参数w
    w = np.array([1, 0, -1], dtype='float32')
    # 将权重参数调整成维度为[cout, cin, kh, kw]的四维张量
    w = w.reshape([1, 1, 1, 3])
    # 创建卷积算子，设置输出通道数，卷积核大小，和初始化权重参数
    # filter_size = [1, 3]表示kh = 1, kw=3
    # 创建卷积算子的时候，通过参数属性param_attr，指定参数初始化方式
    # 这里的初始化方式时，从numpy.ndarray初始化卷积参数
    conv = Conv2D(num_channels=1, num_filters=1, filter_size=[1, 3],
            param_attr=fluid.ParamAttr(
              initializer=NumpyArrayInitializer(value=w)))
    
    # 创建输入图片，图片左边的像素点取值为1，右边的像素点取值为0
    img = np.ones([50,50], dtype='float32')
    img[:, 30:] = 0.
    # 将图片形状调整为[N, C, H, W]的形式
    x = img.reshape([1,1,50,50])
    # 将numpy.ndarray转化成paddle中的tensor
    x = fluid.dygraph.to_variable(x)
    # 使用卷积算子作用在输入图片上
    y = conv(x)
    # 将输出tensor转化为numpy.ndarray
    out = y.numpy()

f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img, cmap='gray')

f = plt.subplot(122)
f.set_title('output featuremap', fontsize=15)
# 卷积算子Conv2D输出数据形状为[N, C, H, W]形式
# 此处N, C=1，输出数据形状为[1, 1, H, W]，是4维数组
# 但是画图函数plt.imshow画灰度图时，只接受2维数组
# 通过numpy.squeeze函数将大小为1的维度消除
plt.imshow(out.squeeze(), cmap='gray')
plt.show()

