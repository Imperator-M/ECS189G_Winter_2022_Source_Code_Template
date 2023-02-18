import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


if 1:
    f = open("MNIST", "rb")
    data = pickle.load(f)
    f.close()
    print("Training Set Size:", len(data["train"]))
    print("Testing Set Size:", len(data["test"]))

class CNN(nn.module):
    def __init__(self):
        super().__init()
        self.conv1 = nn.Conv2d()
