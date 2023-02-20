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
from ORL_Classifier import ORL_Classifier_CNN
from Training_Testing_Classifier import train_model
from Training_Testing_Classifier import eval_model

# Load the data
data = loader("ORL")

# Convert data into tensors
x_train = [] 
y_train = []
x_test = []
y_test = []

x_train, y_train, x_test, y_test = tloader(x_train, y_train, x_test, y_test, data)

#print(y_train[0])

# Create datasets
trainingDS = TensorDataset(x_train, y_train)
testingDS = TensorDataset(x_test, y_test)

# Load dataset
trainingDL = DataLoader(trainingDS, batch_size=25)
testingDL = DataLoader(testingDS, batch_size=25)

# Load CNN
orl_cnn = ORL_Classifier_CNN()

# Train model

epochs, losses, trained_model = train_model(trainingDL, orl_cnn, 25)
'''
plt.plot(epochs.reshape(20,-1).mean(axis=1), losses.reshape(20,-1).mean(axis=1), "o--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.savefig("cifar_training_loss")

trained_model = torch.load("checkpointCIFAR.pth")
eval_model(testingDS, trained_model)
'''