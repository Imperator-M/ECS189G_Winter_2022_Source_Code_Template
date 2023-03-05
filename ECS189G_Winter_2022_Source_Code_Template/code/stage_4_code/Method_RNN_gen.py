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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.embedding_dim = 128
        self.LSTM_size = 128
        self.num_layers = 5

        totalNumWords = 5193

        self.embedding = nn.Embedding(embedding_dim=self.embedding_dim, num_embeddings=totalNumWords)
        self.lstm = nn.LSTM(input_size=self.LSTM_size, hidden_size=self.LSTM_size, num_layers=self.num_layers)
        self.fc = nn.Linear(self.LSTM_size, totalNumWords)

    def forward(self, x, prevState):
        output, state = self.lstm(self.embedding(x), prevState)
        return self.fc(output), state

    def init_state(self):
        sequence_length = 5
        hidden = torch.zeros(self.num_layers, sequence_length, self.LSTM_size)
        cell = torch.zeros(self.num_layers, sequence_length, self.LSTM_size)
        return hidden, cell

class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 2
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)

    def train(self, model, dataloader):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        loss_temp = []
        epoch_temp = []
        for epoch in range(self.max_epoch):
            hidden, cell = model.init_state()

            for x, y in dataloader:
                optimizer.zero_grad()
                y_pred, (hidden, cell) = model(x, (hidden, cell))
                loss = loss_function(y_pred.transpose(1, 2), y)
                hidden = hidden.detach()
                cell = cell.detach()
                loss.backward()
                optimizer.step()

            if epoch%50 == 0 or epoch == self.max_epoch-1:
                # print('22')
                # print(y)
                # print('33')
                # print(y_pred)
                # print(y_pred.transpose(1, 2))
                # print(y_pred.transpose(1, 2).max(1)[1])

                # accuracy_evaluator.data = {'true_y': y, 'pred_y': y_pred.transpose(1, 2).max(1)[1]}
                # print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', loss)
                epoch_temp.append(epoch)
                loss_temp.append(loss.item())
                print('Epoch:', epoch, 'Loss:', loss.item())
        plt.plot(epoch_temp, loss_temp, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


    def predictWord(self, model, wordToIndexMap, indexToWordMap, inputPrompt):
        model.eval()
        words = inputPrompt.split(' ')
        hidden, cell = model.init_state()
        for i in range(0, 50):
            x = torch.tensor([[wordToIndexMap[w] for w in words[i:]]])
            y_pred, (hidden, cell) = model(x, (hidden, cell))

            #p = torch.nn.functional.softmax(y_pred[0][-1], dim=0)
            # print(np.argmax(p))
            # word_index = np.argmax(p)
            # print(len(p))
            # print(sum(p))

            word_index = torch.argmax(y_pred[0][-1]).item()
            wordToAdd = indexToWordMap[word_index]
            if wordToAdd == '':
                print(' '.join(words))
                return
            words.append(wordToAdd)
        print(' '.join(words))

    def run(self):
        print('method running...')
        print('--start training...')
        dataset = self.data['dataset']
        model = RNNModel()
        self.train(model, DataLoader(dataset, batch_size=256))
        while 1:
            prompt = input("Enter 5-word-long prompt ('exit' to quit): ")
            if prompt == 'exit':
                break
            if len(prompt.split(' ')) != 5:
                continue
            invalidKey = False
            for word in prompt.split(' '):
                if word not in self.data['wToi']:
                    print('invalid word:', word)
                    invalidKey = True
                    break
            if invalidKey:
                continue

            self.predictWord(model, self.data['wToi'], self.data['iTow'], prompt)

        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
        return
