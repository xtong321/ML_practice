"""
pytorch training demo
https://www.cnblogs.com/ycfenxi/p/19195640

Complete model training routine
The goal of this chapter is to integrate all the previously discussed topics—data loading, network setup, loss functions, and optimizers—into a standardized, reusable neural network training and testing workflow. We will start from scratch and model.pybuild train.pya complete project by writing two core files.

Overall Approach
A standard model training script has a fixed and clear logical flow, which can be divided into the following eight core steps:

Prepare the dataset : Load the dataset from disk or download it from the network, and split it into training and test sets. Use DataLoadera wrapper to enable batch loading.
Building a network modelmodel.py : Clearly define the structure of the neural network in a separate file.
Creating a model instance : In the main training script train.py, instantiate the network model we have defined.
Define the loss function and optimizer : Select the appropriate loss function and optimization algorithm based on the task type (such as classification, regression).
Setting up the training loop : The main body of the code is a nested loop. The outer loop controls the total number of training epochs, and the inner loop is responsible for iterating through each batch in the dataset.
Core training steps : In the inner loop, strictly follow the four-step training process:
Forward propagation : Inputting data into the model to obtain prediction results.
Calculate the loss : Use the prediction results and the true labels to calculate the loss value.
Backpropagation : Calls to loss.backward()calculate the gradient.
Update parameters : Call optimizer.step()the update model weights function.
Add a testing step : After each epoch of training, use a separate test set to evaluate the model's performance. This helps us monitor the model's generalization ability and determine if overfitting has occurred.
Model saving and visualization : During training, the model's checkpoints are saved periodically, and tools such as TensorBoard are used to record changes in key metrics such as loss and accuracy, thus enabling visualization of the training process.
We will strictly follow this approach in building our code.
"""

## ----------------------------------------------
##1. Building a neural network ( model.py)
# We first create model.pya file that has only one purpose: to define the structure of the neural network. This is good engineering practice, as it separates the model structure from the training logic, making the code clearer.
# 文件: model.py
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


"""
Detailed explanation of code parameters ( model.py)
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding):

in_channels=3: The number of input channels. For RGB color images like CIFAR-10, the number of channels is 3 (red, green, and blue).
out_channels=32: Number of output channels. This value also represents the number of convolutional kernels (or filters). Here we used 32 convolutional kernels, so 32 feature maps will be generated.
kernel_size=5: The size of the convolution kernel. Here it is a 5x5 convolution kernel.
stride=1Stride size, which is the distance the convolution kernel slides across the image each time. 1 indicates moving one pixel at a time.
padding=2Padding. Adds extra pixels (in this case, two circles) around the boundaries of the image. Formula Output_size = (Input_size - Kernel_size + 2*Padding) / Stride + 1, substitute the values (32 - 5 + 2*2)/1 + 1 = 32. padding=2The purpose of this setting is to ensure that the size of the feature map after convolution remains the same as the input.
nn.MaxPool2d(kernel_size):

kernel_size=2: Size of the pooling window. Here it is a 2x2 window. It takes the largest value from the 2x2 region of the input as the output, thus reducing the height and width of the feature map by half (e.g., from 32x32 to 16x16).
nn.Flatten():

This is a parameterless layer. Its sole purpose is to "flatten" the input multidimensional tensor into a one-dimensional vector. For example, a [64, 64, 4, 4]tensor of size (batch_size, channels, height, width) will be transformed into a [64, 1024]tensor of size (batch_size, features), where 1024 = 64 * 4 * 4...
nn.Linear(in_features, out_features):

in_features=1024: The dimension of the input features (number of neurons). This value mustFlatten exactly match the output dimension of the previous layer .
out_features=10: The dimension of the output features. In the last layer, this value must equal the total number of categories in our task. CIFAR-10 has 10 categories, so it is 10 here.
"""



## ----------------------------------------------
##2. Complete training and testing script ( train.py)
#This file is the core of the project; it will call model.pythe model defined in it and execute the complete training and testing process.
# 文件: train.py
# 导入所有需要的库
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 从我们自己写的 model.py 文件中，导入我们定义的 Tudui 模型类
from model import Tudui

# -------------------- 1. 准备数据集 --------------------
# 使用 torchvision.datasets.CIFAR10 加载CIFAR-10的训练数据集
train_data = torchvision.datasets.CIFAR10(
    root="./data",        # 数据集下载后存放的根目录
    train=True,           # 指定这是训练集 (如果为False，则表示加载测试集)
    transform=torchvision.transforms.ToTensor(), # 创建一个转换，将图像数据转换为PyTorch张量，并自动将像素值从[0, 255]归一化到[0.0, 1.0]
    download=True         # 如果在'root'目录下找不到数据集，则自动从网上下载
)
# 加载CIFAR-10的测试数据集
test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,          # 指定这是测试集
    transform=torchvision.transforms.ToTensor(),
    download=True
)
# 获取训练集和测试集的大小，用于后续计算（如准确率）
train_data_size = len(train_data)
test_data_size = len(test_data)
# 使用f-string打印信息，更直观
print(f"训练数据集的长度为: {train_data_size}") # 输出: 50000
print(f"测试数据集的长度为: {test_data_size}")   # 输出: 10000
# 使用DataLoader将数据集封装成可迭代对象，实现批量加载
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# -------------------- 2. 创建网络模型实例 --------------------
# 实例化我们从model.py中导入的Tudui模型
tudui = Tudui()
# -------------------- 3. 定义损失函数 --------------------
# 使用交叉熵损失函数，它在多分类问题中非常常用
# 它内部已经包含了Softmax操作，所以我们的模型输出层不需要加Softmax
loss_fn = nn.CrossEntropyLoss()
# -------------------- 4. 定义优化器 --------------------
# 定义学习率，这是训练中最重要的超参数之一
learning_rate = 0.01
# 创建一个SGD(随机梯度下降)优化器
# 第一个参数 tudui.parameters() 是告诉优化器，模型中所有需要更新的参数都在这里
# 第二个参数 lr=learning_rate 是设置学习率
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
# -------------------- 5. 设置训练网络的一些超参数 --------------------
total_train_step = 0 # 定义一个计数器，记录总的训练步数（一个batch算一步）
total_test_step = 0  # 定义一个计数器，记录总的测试轮数（一个epoch算一轮）
epoch = 10           # 定义训练的总轮数
# -------------------- 6. 添加TensorBoard用于可视化 --------------------
# 创建一个SummaryWriter实例，它会将日志数据写入到'./logs_train'文件夹
writer = SummaryWriter("./logs_train")
# -------------------- 7. 开始训练循环 --------------------
# 外层循环控制训练的轮数(epoch)
for i in range(epoch):
    print(f"-------- 第 {i+1} 轮训练开始 --------")
    # --- 训练步骤 ---
    # 调用 tudui.train()，将模型设置为训练模式。
    # 这对于包含Dropout或BatchNorm层的模型是必需的，以确保它们在训练时正常工作。
    tudui.train()
    # 内层循环遍历训练数据加载器，每次取出一个批次(batch)的数据
    for data in train_dataloader:
        # 从data中解包出图像数据(imgs)和对应的标签(targets)
        imgs, targets = data
        # 1. 前向传播：将图像数据输入到模型中，得到预测输出
        outputs = tudui(imgs)
        # 2. 计算损失：使用损失函数比较预测输出和真实标签，得到损失值
        loss = loss_fn(outputs, targets)
        # 3. 反向传播：这是PyTorch自动求导的核心
        # 首先，清除上一轮计算残留的梯度
        optimizer.zero_grad()
        # 然后，调用loss.backward()计算当前损失相对于模型所有参数的梯度
        loss.backward()
        # 最后，调用optimizer.step()，优化器会根据计算出的梯度来更新模型的参数
        optimizer.step()
    # 更新总训练步数
    total_train_step += 1
    # 为了避免打印过于频繁，我们设置每训练100步打印一次信息
    if total_train_step % 100 == 0:
        # loss是一个张量，loss.item()可以从中获取其数值
        print(f"训练步数: {total_train_step}, Loss: {loss.item()}")
        # 使用writer将训练损失记录到TensorBoard，方便可视化
        writer.add_scalar("train_loss", loss.item(), total_train_step)
# --- 测试步骤 ---
# 在每轮训练结束后，进行一次测试来评估模型的性能
# 调用 tudui.eval()，将模型设置为评估模式。
# 这会关闭Dropout层，并让BatchNorm层使用全局统计数据，确保测试结果的确定性。
tudui.eval()
total_test_loss = 0  # 初始化测试集上的总损失
total_accuracy = 0   # 初始化测试集上的总正确数
# 使用 with torch.no_grad(): 块，暂时禁用所有梯度计算。
# 这可以节省内存并加快计算速度，因为在测试时我们不需要进行反向传播。
with torch.no_grad():
    # 遍历测试数据加载器
    for data in test_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 累加每个批次的损失
        total_test_loss += loss.item()
        # 计算这个批次的正确预测数
        # outputs.argmax(1) 会返回在维度1上最大值的索引，即模型预测的类别
        # (outputs.argmax(1) == targets) 会得到一个布尔张量，预测正确的位置为True
        # .sum() 会将所有True（计为1）加起来，得到正确预测的数量
        accuracy = (outputs.argmax(1) == targets).sum()
        # 累加每个批次的正确数
        total_accuracy += accuracy
        # 打印本轮训练结束后，在整个测试集上的性能指标
        print(f"本轮训练结束，在测试集上的总Loss为: {total_test_loss}")
        print(f"本轮训练结束，在测试集上的总正确率为: {total_accuracy / test_data_size}")
        # 使用writer将测试损失和准确率记录到TensorBoard
        writer.add_scalar("test_loss", total_test_loss, i) # x轴使用轮数i
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, i)
        # 保存当前轮次的模型状态
        torch.save(tudui, f"tudui_epoch_{i}.pth")
        print(f"模型 tudui_epoch_{i}.pth 已保存")
        # 所有训练轮数结束后，关闭SummaryWriter
        writer.close()

"""
Detailed explanation of code parameters ( train.py)
torchvision.datasets.CIFAR10(...):

root="./data"Specify a folder path, and PyTorch will download and extract the dataset to that location.
train=True/ False: Boolean value, Trueindicating loading the training set, or Falseindicating loading the test set.
transform=torchvision.transforms.ToTensor()The loaded image is preprocessed. ToTensorTwo core tasks are performed: 1. Converting the PIL Image format image or NumPy array to a normalized format torch.FloatTensor. 2. Scaling the image's pixel values ​​from [0, 255]an integer range to [0.0, 1.0]a floating-point range. Normalization is a crucial step in neural network training , enabling the model to converge faster and more stably.
download=TrueIf rootthe dataset cannot be found in the specified path, it will be automatically downloaded from the internet.
DataLoader(dataset, batch_size):

dataset: The dataset object to load, i.e., the one created above train_dataor test_data.
batch_size=64Batch size: This indicates that 64 samples are taken from the dataset each time and packaged into a batch. This is a trade-off between training efficiency and memory consumption.
tudui.train()andtudui.eval() :

train()Switch the model to training mode . This enables Dropoutlayer and BatchNormlayer training behavior (e.g., BatchNormit calculates and updates the mean and variance for each batch).
eval()Switch the model to evaluation/test mode . This disables Dropoutlayers and forces BatchNormthem to use a fixed mean and variance learned across the entire training set, ensuring deterministic test results. Switching between these two modes between training and testing is absolutely necessary; otherwise, inconsistent or erroneous results may occur.
with torch.no_grad()::

This is a context manager that tells PyTorch that all computations within this code block do not require gradient calculation . During the testing phase, we only care about the forward propagation results of the model and do not need to perform backpropagation, so disabling gradients can: 1. save a lot of memory ; 2. significantly speed up computation .
outputs.argmax(1):

outputsThe shape is such that [64, 10]it represents 64 samples, each corresponding to the raw scores (logits) of 10 categories.
argmax(1)Find the index of the maximum value along dimension 1 (i.e., the category dimension) . For example, if a sample has a score of [0.1, 2.5, 0.3, ...], argmaxthe index is returned 1, indicating that the model predicts that the sample belongs to class 1.
Therefore, outputs.argmax(1)it will return a [64]tensor of shape containing the predicted class for the 64 samples in this batch.
"""
