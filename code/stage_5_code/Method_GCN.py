"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.layers import GraphConvolution
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


class Method_GCN(method, nn.Module):
    data = None
    max_epochs = 100
    message_epoch = 25
    learning_rate = 1e-2
    weight_decay = 5e-4

    # Specific to the Cora dataset
    num_feat = 1433
    num_hid_feat = 16
    num_classes = 7

    # ARCHITECTURE
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.gconv1 = GraphConvolution(self.num_feat, self.num_hid_feat)
        self.gconv2 = GraphConvolution(self.num_hid_feat, self.num_classes)

    # ACTUAL IMPLEMENTATION OF ARCHITECTURE
    def forward(self, x, adj):
        '''Forward propagation'''
        x = self.gconv1(x, adj)
        x = F.relu(x)
        x = self.gconv2(x, adj)
        # Returns predicted y
        return F.log_softmax(x, dim=1)

    # TRAINING PROCESS, WHICH USES FORWARD IN COMBINATION WITH OPTIMIZATION FUNCTION AND STUFF
    def train(self, graph, train_test_val):
        # DECIDE FUNCTIONS
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        epoch_history = []
        #accuracy_history = []
        loss_history = []

        # LET THE COMPUTER KEEP USING THE ARCHITECTURE AND UPDATING THE WEIGHTS
        for epoch in range(self.max_epochs):
            features = graph['X']
            labels = graph['y'][train_test_val['index_train']]
            adj = graph['utility']['A']

            # Forward pass
            output = self(features, adj)

            # Compute the loss
            loss = F.nll_loss(output[train_test_val['index_train']], labels)
            loss_history.append(loss.item())
            epoch_history.append(epoch)

            # Backpropogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss at each epoch
            if (epoch+1) % self.message_epoch == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.max_epochs, loss.item()))


        plt.plot(epoch_history, loss_history, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()



    # APPLY ARCHITECTURE TO THE UNUSED TESTING SET TO SEE THE ACTUAL ACCURACY
    def test(self, graph, train_test_val):

        # Get the test set indices
        index_test = train_test_val['index_test']

        # Make predictions on the test set
        with torch.no_grad():
            output = self(graph['X'], graph['utility']['A'])
            _, predicted = torch.max(output, dim=1)

        return predicted[index_test]

    # TAPE EVERYTHING TOGETHER SO WE JUST HAVE TO USE CALL ONE FUNCTION
    def run(self):
        print('method running...')
        print('--start training...')

        self.train(self.data['graph'], self.data['train_test_val'])
        print('--start testing...')
        pred_y = self.test(self.data['graph'], self.data['train_test_val'])

        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][self.data['train_test_val']['index_test']]}
