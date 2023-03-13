'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from code.stage_5_code.layers import GraphConvolution

import matplotlib.pyplot as plt
from torch.autograd import Variable



class Method_Pubmed(method, nn.Module):
    data = None
    # used the recommended architecture outlined in https://arxiv.org/pdf/1609.02907.pdf as posted by the professor
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-2
    decay_rate = 5e-4
    input_dim = 500
    hidden_dim = 8
    output_dim = 3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = GraphConvolution(self.input_dim, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = GraphConvolution(self.hidden_dim, self.output_dim)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(p=.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        x = self.relu2(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def trainer(self, graph, train_test_val):

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.decay_rate)

        #store information for plotting purposes
        epochNum = []
        train_loss = []

        # Set the model to train mode
        self.train()

        # Loop over the training data for a specified number of epochs
        for epoch in range(self.max_epoch):
            # Get the training data
            features = graph['X']
            labels = graph['y'][train_test_val['idx_train']]
            adj = graph['utility']['A']

            # Forward pass
            output = self(features, adj)

            # Compute the loss
            loss = loss_function(output[train_test_val['idx_train']], labels)
            train_loss.append(loss.item())
            epochNum.append(epoch)
            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss at each epoch
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.max_epoch, loss.item()))
        plt.plot(epochNum, train_loss, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


    def test(self, graph, train_test_val):
        self.eval()
        # Get the test set indices
        idx_test = train_test_val['idx_test']

        # Make predictions on the test set
        with torch.no_grad():
            output = self(graph['X'], graph['utility']['A'])
            _, predicted = torch.max(output, dim=1)

        return predicted[idx_test]


    def run(self):
        print('method running...')
        print('--start training...')

        self.trainer(self.data['graph'], self.data['train_test_val'])
        print('--start testing...')
        pred_y = self.test(self.data['graph'], self.data['train_test_val'])

        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_test']]}

