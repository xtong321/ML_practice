"""
搭建神经网络 (model.py)
我们首先创建model.py文件，这个文件只负责一件事：
定义神经网络的结构。这是一种良好的工程实践，它让模型结构
与训练逻辑分离，使代码更清晰。


代码参数讲解 (model.py)
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding):
    in_channels=3: 输入的通道数。对于CIFAR-10这种RGB彩色图像，通道数是3（红、绿、蓝）。
    out_channels=32: 输出的通道数。这个数值也代表了卷积核（或称滤波器）的数量。这里我们用了32个卷积核，所以会产生32张特征图（feature map）。
    kernel_size=5: 卷积核的大小。这里是5x5的卷积核。
    stride=1: 步长，即卷积核每次在图像上滑动的距离。1表示一次移动一个像素。
    padding=2: 填充。在图像的边界周围添加额外的像素（这里是2圈）。公式 Output_size = (Input_size - Kernel_size + 2*Padding) / Stride + 1，代入数值 (32 - 5 + 2*2)/1 + 1 = 32。设置padding=2的目的是为了让卷积后的特征图尺寸与输入保持不变。

nn.MaxPool2d(kernel_size):
    kernel_size=2: 池化窗口的大小。这里是2x2的窗口。它会从输入的2x2区域中取出最大的那个值作为输出，从而将特征图的高度和宽度都缩小一半（例如，从32x32缩小到16x16）。

nn.Flatten():
    这是一个没有参数的层。它的唯一作用就是将输入的多维张量“压平”成一个一维向量。例如，一个[64, 64, 4, 4]的张量（batch_size, channels, height, width）会被转换成[64, 1024]的张量（batch_size, features），其中 1024 = 64 * 4 * 4。

nn.Linear(in_features, out_features):
    in_features=1024: 输入特征的维度（神经元数量）。这个数值必须与前一层Flatten的输出维度完全匹配。
    out_features=10: 输出特征的维度。在最后一层，这个数值必须等于我们任务的类别总数。CIFAR-10有10个类别，所以这里是10。
"""

import torch
from torch import nn

class Tudui(nn.Module):
    """
    一个针对CIFAR-10数据集(3x32x32)的卷积神经网络模型。
    该结构参考了经典的CIFAR-10模型设计。
    """
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
        # 第一个卷积层
        # 输入形状: [batch_size, 3, 32, 32]
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
        # 经过卷积后，形状变为: [batch_size, 32, 32, 32] (因为 stride=1, padding=2 保持了尺寸)
        # 第一个最大池化层
        nn.MaxPool2d(kernel_size=2),
        # 经过池化后，形状变为: [batch_size, 32, 16, 16] (高度和宽度减半)
        # 第二个卷积层
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        # 形状仍为: [batch_size, 32, 16, 16]
        # 第二个最大池化层
        nn.MaxPool2d(kernel_size=2),
        # 形状变为: [batch_size, 32, 8, 8]
        # 第三个卷积层
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
        # 形状变为: [batch_size, 64, 8, 8]
        # 第三个最大池化层
        nn.MaxPool2d(kernel_size=2),
        # 形状变为: [batch_size, 64, 4, 4]
        # 压平层，为全连接层做准备
        nn.Flatten(),
        # 形状变为: [batch_size, 64 * 4 * 4], 即 [batch_size, 1024]
        # 第一个全连接层
        nn.Linear(in_features=1024, out_features=64),
        # 形状变为: [batch_size, 64]
        # 第二个全连接层 (输出层)
        nn.Linear(in_features=64, out_features=10)
        # 最终输出形状: [batch_size, 10]
        )

    def forward(self, x):
        """定义数据的前向传播过程"""
        x = self.model(x)
        return x

# --- 用于验证模型正确性的代码 ---
if __name__ == '__main__':
    tudui = Tudui()
    # 创建一个假的输入张量来测试网络
    # torch.ones创建一个全为1的张量
    # (64, 3, 32, 32) 表示一个批次包含64张图片，每张图片3个通道，高和宽都是32像素
    input_tensor = torch.ones((64, 3, 32, 32))
    output_tensor = tudui(input_tensor)
    # 打印输出的形状，以验证网络结构是否按预期工作
    print(output_tensor.shape) # 期望输出: torch.Size([64, 10])