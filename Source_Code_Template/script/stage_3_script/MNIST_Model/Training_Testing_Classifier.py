import numpy as np
import torch.nn as nn
from torch.optim import SGD

def train_model(trainingDL, model, n_epochs=20):
    opt = SGD(model.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    losses = []
    epochs = []

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(trainingDL)
        for i, (x, y) in enumerate(trainingDL):
            opt.zero_grad()
            loss_value = L(model(x), y)
            loss_value.backward()
            opt.step()
            
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)

def eval_model(testingDL, model):
    predictions = []
    for (x,y) in testingDL:
        pred = model(x)
        