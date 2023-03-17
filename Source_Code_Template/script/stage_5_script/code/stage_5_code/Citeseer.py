# Edited source code provided from PyGCN repo

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import *
from code.stage_5_code.layers import *

import matplotlib.pyplot as plt

class Citeseer_Method(method, nn.Module):
    data = None
    epoch = 500
    lr = 0.05
    dr = 0.0001
    num_input = 3703
    num_dim = 256
    output = 6

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # number of features first then number of hidden
        self.gconv1 = GraphConvolution(self.num_input, self.num_dim)
        self.gconv2 = GraphConvolution(self.num_dim, self.output)
        self.dropout = 0.50
    
    def forward(self, x, adj):
        x = F.leaky_relu(self.gconv1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.gconv2(x, adj)
        return F.log_softmax(x, dim=1)
    
    def trainer(self, graph, train_test_val):
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.dr)
        optimizer.zero_grad()

        losses = []

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.epoch):
            output = self(graph["X"], graph["utility"]["A"])
            loss = loss_fn(output[train_test_val["idx_train"]], graph["y"][train_test_val["idx_train"]])
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch:", epoch, "Loss:", loss.item())
        

        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Citeseer Training Loss")
        plt.savefig("citeseer_training_loss_2.png")
    
    def test(self, graph, train_test_val):
        self.eval()

        idx_test = train_test_val["idx_test"]

        with torch.no_grad():
            output = self(graph["X"], graph["utility"]["A"])
            _, predicted = torch.max(output, dim=1)

            return predicted[idx_test]
    
    def run(self):
        print('method running...')
        print('--start training...')

        self.trainer(self.data['graph'], self.data['train_test_val'])
        print('--start testing...')
        pred_y = self.test(self.data['graph'], self.data['train_test_val'])

        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_test']]}

