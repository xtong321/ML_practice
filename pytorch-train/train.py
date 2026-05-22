"""
这个文件是项目的核心，它将调用model.py中定义的模型，并执行完整的训练和测试流程

代码参数超详细讲解 (train.py)
torchvision.datasets.CIFAR10(...):
    root="./data": 指定一个文件夹路径，PyTorch会把数据集下载并解压到这里。
    train=True / False: 布尔值，True表示加载训练集，False表示加载测试集。
    transform=torchvision.transforms.ToTensor(): 对加载的图像进行预处理。ToTensor做了两件核心的事：1. 将PIL Image格式的图像或Numpy数组转换为torch.FloatTensor。2. 将图像的像素值从[0, 255]的整数范围，缩放到[0.0, 1.0]的浮点数范围。归一化是神经网络训练中至关重要的一步，能让模型收敛得更快、更稳定。
    download=True: 如果root路径下找不到数据集，就自动从网上下载。

DataLoader(dataset, batch_size):
    dataset: 要加载的数据集对象，即上面创建的train_data或test_data。
    batch_size=64: 批处理大小。表示每次从数据集中取出64个样本打包成一个批次。这是训练效率和内存消耗之间的一个权衡。
    
tudui.train() 和 tudui.eval():
    train(): 将模型切换到训练模式。这会启用Dropout层和BatchNorm层的训练行为（例如，BatchNorm会计算并更新每个批次的均值和方差）。
    eval(): 将模型切换到评估/测试模式。这会禁用Dropout层，并让BatchNorm层使用在整个训练集上学习到的固定的均值和方差，确保测试结果是确定性的。在训练和测试之间切换这两种模式是绝对必要的，否则会导致结果不一致或错误。

with torch.no_grad()::
    这是一个上下文管理器，它告诉PyTorch在这个代码块内部的所有计算都不需要计算梯度。在测试阶段，我们只关心模型的前向传播结果，不需要进行反向传播，所以禁用梯度可以：1. 节省大量内存；2. 显著加快计算速度。

outputs.argmax(1):
    outputs的形状是[64, 10]，代表64个样本，每个样本对应10个类别的原始得分（logits）。
    argmax(1)沿着维度1（即类别维度）查找最大值的索引。例如，如果一个样本的得分是[0.1, 2.5, 0.3, ...]，argmax会返回索引1，代表模型预测这个样本属于第1类。
    所以 outputs.argmax(1) 会返回一个形状为[64]的张量，包含了对这个批次中64个样本的预测类别。
"""

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
    root="../../datasets",        # 数据集下载后存放的根目录
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

    # 内层循环遍历训练数据加载器，每次取出一个批次(batch)的数据`
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