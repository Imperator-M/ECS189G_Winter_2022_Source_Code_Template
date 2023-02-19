import torch.nn as nn
from torch import flatten

# Based on LeNet Architecture: https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

class CIFAR_Classifier_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_Classifier_CNN, self).__init__()
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

        
