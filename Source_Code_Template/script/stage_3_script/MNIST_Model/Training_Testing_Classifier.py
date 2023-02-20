import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn import metrics

def train_model(trainingDL, model, n_epochs=20):
    opt = SGD(model.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    losses = []
    epochs = []

    for epoch in range(n_epochs):
        model.train()
        print("Epoch", epoch, end=" ")
        N = len(trainingDL)
        running_loss = 0.0
        for i, (x, y) in enumerate(trainingDL):
            opt.zero_grad()
            loss_value = L(model(x), y)
            loss_value.backward()
            opt.step()
            
            epochs.append(epoch+i/N)
            running_loss += loss_value.item()
            #losses.append(loss_value.item())
            if i % 3000 == 2999:
                print('loss: %.3f' % (running_loss / 3000))
                losses.append((running_loss / 3000))
                running_loss = 0.0
    
    #print(epoch+1, epoch)
    torch.save(model, "checkpoint.pth")
    return np.array(epochs), np.array(losses), model

def eval_model(testingDS, model):
    
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        for (x, y) in testingDS:
            pred = model(x)
            y_pred.append(torch.argmax(pred))
            y_true.append(torch.argmax(y))

    
    print(metrics.classification_report(y_true, y_pred, digits=3))
    #print(len(testingDS))
    #print("Accuracy on entire test dataset: ", ((numCorrect/len(testingDS))*100))
