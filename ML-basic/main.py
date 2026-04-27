"""
https://github.com/soapisnotfat/pytorch-cifar10/blob/master/main.py
"""
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import argparse
from models import *
from misc import progress_bar

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs to train')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        
        # self.model = LeNet().to(self.device)
        self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = GoogLeNet().to(self.device)
        # self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        # self.model = resnet152().to(self.device)
        # self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        # self.model = WideResNet(depth=28, num_classes=10).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1) # second para "1" means the dim to be reduced
            total += target.size(0)

            #train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy()==target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.2f%% (%d/%d)'
            % (train_loss / (batch_num + 1), 100.*train_correct/total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n==> epoch: %03d/%03d" % (epoch, self.epochs))
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("==> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


# my personal implementation of 2D conv
import torch
import torch.nn as nn
def pth_conv(input, kernel):
    # input - N C H W (batch_size, Channels, H x W)
    # in_channels, out_channels, bias, stride, padding
    # kernel - out, in, kh, kw
    out_channels, in_channles, kh, kw = kernel.shape
    conv_layer = nn.Conv2d(in_channles, out_channels, kernel_size = (kh, kw), bias=False,stride=1,padding=0)
    conv_layer.weight = nn.Parameter(kernel)
    return conv_layer(input)

def pth_conv_test():
    input = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]).unsqueeze(0).unsqueeze(0).float()
    kernel = torch.tensor([[1,0],[0,1]]).unsqueeze(0).unsqueeze(0).float()

    output = pth_conv(input, kernel)
    print(output)
    assert torch.equal(output, torch.tensor([[[[6,8],[12,14]]]]))


if __name__ == '__main__':
    """import torch

    # Input tensors
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # Operation
    z = x * y

    # Gradients
    z.backward()  # Triggers gradient computation for z with respect to x and y

    # Accessing gradients
    print(x.grad)  # Prints 3.0, which is equal to y
    print(y.grad)  # Prints 2.0, which is equal to x
    print(z.grad)  # None, as z is a scalar. Its gradient is replaced by grad_fn, the function that generated z.
    """
    main()
    #pth_conv_test()