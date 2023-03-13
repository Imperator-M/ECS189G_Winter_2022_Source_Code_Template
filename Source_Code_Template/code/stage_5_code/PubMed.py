# Edited source code provided by Kipf & Welling

import torch
from torch import nn
import torch.nn.functional as F

from code.base_class.method import method
from Evaluate_Accuracy import *
from layers import *

class PubMed_Method(method, nn.Module):
    
    def __init__(self, num_features, num_hidden, num_classes, dropout):
        super(PubMed_Method, self).__init__()

        self.gc1 = GraphConvolution(num_features, num_hidden)
        self.gc2 = GraphConvolution(num_hidden, num_classes)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

