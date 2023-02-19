'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Normalize, Resize
import torch.nn.functional as F
class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(34944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 40)
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')


        for epoch in range(self.max_epoch):
            # print(X)
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # print(y_pred.size())

            #minus 1 here because our class label starts from 1
            y_true = torch.LongTensor(np.array(y))-1
            # print(y_pred.max(1)[1])
            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 1 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
    
    def test(self, X):
        y_pred = self.forward(X)
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.data['train']['X'] = torch.FloatTensor(np.array(self.data['train']['X']))
        #we have three channels but we only need one
        self.data['train']['X'] = self.data['train']['X'][:, :, :, 0:1]
        #pytorch format data as [batch, num_channel, image_height, image_width]
        #our data is [batch, image_height, image_width, num_channel]
        #permute will do the change
        self.data['train']['X'] = self.data['train']['X'].permute(0, 3, 1, 2)
        self.data['train']['y'] = torch.LongTensor(np.array(self.data['train']['y']))

        self.data['test']['X'] = torch.FloatTensor(np.array(self.data['test']['X']))
        self.data['test']['X'] = self.data['test']['X'][:, :, :, 0:1]
        self.data['test']['X'] = self.data['test']['X'].permute(0, 3, 1, 2)
        self.data['test']['y'] = torch.LongTensor(np.array(self.data['test']['y']))

        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            