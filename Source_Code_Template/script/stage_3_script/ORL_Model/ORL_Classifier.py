import torch.nn as nn
import torch
from torch import flatten

# Based on CIFAR Architecture: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

class ORL_Classifier_CNN(nn.Module):
    def __init__(self):
        super(ORL_Classifier_CNN, self).__init__()
        self.Matrix1 = nn.Linear(112*92, 1024)
        self.Matrix2 = nn.Linear(1024, 128)
        self.Matrix3 = nn.Linear(128,40)
        self.R = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 112*92)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()



        
