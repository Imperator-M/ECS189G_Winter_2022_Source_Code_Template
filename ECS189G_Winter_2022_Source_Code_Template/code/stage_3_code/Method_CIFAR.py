'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn

import matplotlib.pyplot as plt
from torch.autograd import Variable



class Method_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 25
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = .01
    batch_size = 20

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Convolution 1

        # Training set size: 50,000, testing set size: 10,000, number of classes: 10. Each instance is a
        # 32x32 color image, and will have one single label denoted by an integer from {0, 1, 2, …, 9}.

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(2, 2)


        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(2, 2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=.3)

        # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)

        # Max pool 3
        out = self.maxpool3(out)

        # Convolution 4
        out = self.cnn4(out)
        out = self.relu4(out)

        # Max pool 4
        out = self.maxpool4(out)

        out = out.reshape(out.size(0), -1)

        # Linear function (readout)
        #out = self.fc1(out)
        out = self.dropout(self.relu1(self.fc1(out)))
        out = self.dropout(self.relu2(self.fc2(out)))
        out = self.dropout(self.relu3(self.fc3(out)))
        out = self.fc4(out)

        return out

    def train(self, train_data, test_data, model):

        #data loader to implement batch strategy. We pass in the entire trainging set and give it a batch size. It converts it into a tensor
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                     batch_size= self.batch_size,
                                                     shuffle=True)

        loss_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        epochNum = []
        loss = []
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        for epoch in range(self.max_epoch):

            correct = 0
            T1 = torch.Tensor([])  # tensor to hold all predictions for use in evaluation

            for i, (images, labels) in enumerate(train_loader):
                images = images.permute(0, 3, 1, 2)  #needed to convert tensor from torch.Size([batch, 32, 32, 3]) to torch.Size([batch, 3, 32, 32])
                images = images.type(torch.float32) #needed to convert tensor from float64 to float32


                images = Variable(images)
                labels = Variable(labels)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()

                # Forward pass to get predicted output
                outputs = model(images)

                # calculate the training loss
                train_loss = loss_function(outputs, labels)

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()

                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                # Total correct predictions
                predicted = torch.max(outputs.data, 1)[1]
                correct += (predicted == labels).sum()  # used to calculate

                # T1 = torch.cat((T1, predicted))
                #
                # accuracy_evaluator.data = {'true_y': T1, 'pred_y': self.data['test_y']}
                # accuracy, precision, recall, f1 = accuracy_evaluator.evaluate()

                if i % 50000 == 0:  # 50*32 = 1600 per
                    epochNum.append(epoch)
                    loss.append(train_loss.item())
                    print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                        epoch, i * len(images), len(train_loader.dataset),
                               100. * i / len(train_loader), train_loss.item(),
                               float(correct * 100) / float(self.batch_size * (i + 1))))

        plt.plot(epochNum, loss, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        return model

    def test(self, test_data, model):
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                    batch_size=self.batch_size,
                                                    shuffle=False)
        total = 0
        correct = 0

        T1 = torch.Tensor([]) #tensor to hold all predictions for use in evaluation

        for images, labels in test_loader:
            images = images.permute(0, 3, 1, 2)  # needed to convert tensor from torch.Size([batch, 32, 32, 3]) to torch.Size([batch, 3, 32, 32])
            images = images.type(torch.float32)
            images = Variable(images)


            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            T1 = torch.cat((T1, predicted))
            # Total number of labels
            total += labels.size(0)

            correct += (predicted == labels).sum()

        print("Test accuracy:{:.3f}% ".format(float(correct) / (len(test_loader) * self.batch_size)))

        return T1


    def run(self):
        print('method running...')
        print('--start training...')
        model = Method_CIFAR('convolutional neural network', '') #I was having problems getting the model from the script so I just added it manually here


        model_fit = self.train(self.data['train'], self.data['test'], model)
        print('--start testing...')
        pred_y = self.test(self.data['test'], model_fit)

        return {'pred_y': pred_y, 'true_y': self.data['test_y']}

