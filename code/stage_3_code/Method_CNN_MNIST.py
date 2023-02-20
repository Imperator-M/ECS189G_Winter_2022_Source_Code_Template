'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    # Defines the max rounds to train the model.
    max_epoch = 1
    # Defines the learning rate for gradient descent based optimizer for model learning.
    learning_rate = 1e-3

    # Defines the the CNN model architecture.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y_pred = self.softmax(x)
        return y_pred

    # Backward error propagation is an implemented function in PyTorch,
    # so we don't need to define it here.

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        # For evaluating training accuracy.
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # It's be an iterative gradient updating process.
        # We don't do mini-batch; we use the whole input as one batch.
        # You can try to split X and y into smaller-sized batches.
        for epoch in range(self.max_epoch):  # You can do an early stop if self.max_epoch is too much...
            # Get the output; we need to covert X into torch.tensor so pytorch algorithm can operate on it.
            #num_channels = 1
            #1 => grayscale
            #print(torch.FloatTensor(np.array([X])).shape)
            X_chan = np.array(X).reshape(np.array(X).shape[0], 1, np.array(X).shape[1], np.array(X).shape[2])
            y_pred = self.forward(torch.FloatTensor(X_chan))
            # Convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            print('y_pred:', y_pred.shape)
            print('y_true:', y_true.shape)
            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()

            # Do error backpropagation to calculate the gradients.
            train_loss.backward()
            # Update the variables according to the optimizer and gradients calculated by the above loss.backward function.
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        # Do the testing, and result the result.
        X_chan = np.array(X).reshape(np.array(X).shape[0], 1, np.array(X).shape[1], np.array(X).shape[2])
        y_pred = self.forward(torch.FloatTensor(X_chan))
        # Instances get labeled according to the largest.
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}