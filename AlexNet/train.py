from fnmatch import translate
import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cude.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transform.Compose([transforms.RandomResizeCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
        "val": transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize()])
    }