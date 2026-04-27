import torch
import torch.nn as nn

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    
    def __init__(self, classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # layer-1
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # layer-2
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            #layer-3
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            #layer-4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            #layer-5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # FC
        self.classifier = nn.Sequential(
            # layer-6
            nn.Dropout(),            
            nn.Linear(256*1*1, 4096),
            nn.ReLU(inplace=True),

            # layer-7
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            #layer-8
            nn.Linear(4096, classes),

        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

        
def alexnet(**kwargs):
    #r"""AlexNet model architecture
    #"""
    model = AlexNet(**kwargs)
    return model