import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as func
from Dataset_Loader import loader
from Tensor_Dataset import TensorDataset
from Torch_Loader import tloader
from torch.utils.data import Dataset, DataLoader
#from MNIST_Classifier import MNIST_Classifier_CNN
#from Training_Testing_Classifier import train_model
#from Training_Testing_Classifier import eval_model

# Load the data
data = loader("CIFAR")

# Convert data into tensors
x_train = [] 
y_train = []
x_test = []
y_test = []

x_train, y_train, x_test, y_test = tloader(x_train, y_train, x_test, y_test, data)

# Create datasets
trainingDS = TensorDataset(x_train, y_train)
testingDS = TensorDataset(x_test, y_test)

# Load dataset
trainingDL = DataLoader(trainingDS, batch_size=50)

# Load CNN
mnist_cnn = MNIST_Classifier_CNN()

# Train model
train_model(trainingDL, mnist_cnn, 20)
