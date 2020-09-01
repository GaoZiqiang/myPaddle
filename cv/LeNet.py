# 导入需要的包
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

# 定义 LeNet 网络结构
class LeNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = Conv2D(num_channels=1, num_filters=6, filter_size=5, act='sigmoid')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=6, num_filters=16, filter_size=5, act='sigmoid')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 创建第3个卷积层
        self.conv3 = Conv2D(num_channels=16, num_filters=120, filter_size=4, act='sigmoid')
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc1 = Linear(input_dim=120, output_dim=64, act='sigmoid')
        self.fc2 = Linear(input_dim=64, output_dim=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 输入数据形状是 [N, 1, H, W]
# 这里用np.random创建一个随机数组作为输入数据
x = np.random.randn(*[3,1,28,28])
x = x.astype('float32')
with fluid.dygraph.guard():
    # 创建LeNet类的实例，指定模型名称和分类的类别数目
    m = LeNet(num_classes=10)
    # 通过调用LeNet从基类继承的sublayers()函数，
    # 查看LeNet中所包含的子层
    print(m.sublayers())
    # 将numpy.ndarray转化成paddle中的tensor
    x = fluid.dygraph.to_variable(x)
    for item in m.sublayers():
        # item是LeNet类中的一个子层
        # 查看经过子层之后的输出数据形状
        try:
            x = item(x)
        except:
            x = fluid.layers.reshape(x, [x.shape[0], -1])
            x = item(x)
        if len(item.parameters())==2:
            # 查看卷积和全连接层的数据和参数的形状，
            # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
            print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
        else:
            # 池化层没有参数
            print(item.full_name(), x.shape)
