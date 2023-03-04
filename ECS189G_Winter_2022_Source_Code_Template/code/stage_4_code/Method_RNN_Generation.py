'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import numpy as np
from torch import nn
from code.stage_4_code.Dataset import Dataset
from code.stage_4_code.JokeDataset import JokeDataset
import torchtext


class Method_RNN_Generation(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    batch_size = 128
    sequence_length = 5
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    vocab_size = 288

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.input = self.vocab_size
        self.hidden_dim = 128
        self.embedding_dim = 100
        self.num_layers = 3
        self.dropout = .3

        self.embedding = nn.Embedding(
            num_embeddings=self.input,
            embedding_dim=self.embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_dim, self.input)

    def forward(self, text):
        # text shape: (seq_len, batch_size)
        embedded = self.dropout(self.embedding(text))
        # embedded shape: (seq_len, batch_size, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: (seq_len, batch_size, hidden_dim)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # cell shape: (num_layers * num_directions, batch_size, hidden_dim)
        prediction = self.fc(self.dropout(output))
        # prediction shape: (seq_len, batch_size, vocab_size)
        return prediction.transpose(0, 1)  # transpose to get (batch_size, seq_len, vocab_size)

    def trainer(self, train_dataloader, val_dataloader):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose


        #train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size= self.batch_size)

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            total_loss = 0
            self.train()  # Set model to training mode
            for input_seq, target_seq in train_dataloader:

                # print("input_seq shape:", input_seq.shape)
                # print("target_seq shape:", target_seq.shape)
                optimizer.zero_grad()
                output = self(input_seq)
                target_seq = target_seq.view(-1, 1)  # flatten target_seq to (batch_size*sequence_length, 1)

                # print("output: in trainer", output.reshape(-1, self.vocab_size).shape)
                # print("target_seq: in trainer", target_seq.shape)

               # loss = loss_function(output.reshape(self.batch_size, self.sequence_length, self.vocab_size), target_seq.view(self.batch_size, self.sequence_length))
                loss = loss_function(output.reshape(-1, self.vocab_size), target_seq.squeeze())

                loss.backward()
                optimizer.step()
                total_loss += loss.item()


            train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}: train loss = {train_loss:.4f}")

            # Evaluate model
            self.eval()  # Set model to evaluation mode
            total_loss = 0
            with torch.no_grad():
                for input_seq, target_seq in val_dataloader:
                    output = self(input_seq)
                    #loss = loss_function(output.view(-1, self.vocab_size), target_seq.view(-1))
                    loss = loss_function(output.reshape(-1, self.vocab_size), target_seq.view(-1))
                    total_loss += loss.item()

            val_loss = total_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1}: val loss = {val_loss:.4f}")

        return self

    def generate_joke(self, start_string, vocab, max_length=10):
        # Tokenize the starting string
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        tokens = tokenizer(start_string.lower())
        print([vocab.get_stoi()[token] for token in tokens])
        # Convert the tokens to a tensor of word indices
        start_tensor = torch.tensor([vocab.get_stoi()[token] for token in tokens]).unsqueeze(1)

        # Set the model to evaluation mode
        self.eval()

        # Initialize the hidden state of the LSTM
        hidden = None

        # Generate the joke word by word
        for i in range(max_length):
            # Get the model's predicted probability distribution over the next word
            with torch.no_grad():
                output = self(start_tensor[-1].unsqueeze(0))
                probabilities = torch.softmax(output[0], dim=0)

            # Sample the next word from the probability distribution
            word_index = torch.multinomial(probabilities, num_samples=1).item()

            print(word_index)
            print(len(vocab))
            # Convert the word index back to its corresponding word
            word = vocab.get_itos()[word_index]

            # Add the word to the tensor of word indices
            start_tensor = torch.cat((start_tensor, torch.tensor([[word_index]])))

            # If the model generates the <eos> token, stop generating
            if word == '<eos>':
                break

        # Convert the tensor of word indices back to a string
        output_tokens = [vocab.get_itos()[word_index] for word_index in start_tensor.squeeze()]
        output_string = ' '.join(output_tokens)

        # Return the generated joke
        return output_string.capitalize()

    def run(self):
        print('method running...')
        print('--start training...')
        model = Method_RNN_Generation('recurrent neural network', '') #I was having problems getting the model from the script so I just added it manually here
        text = 'What did the bartender say'
        dataset = Dataset()

        train_dataset = JokeDataset(self.data['train'], self.data['vocab'])
        val_dataset = JokeDataset(self.data['validate'], self.data['vocab'])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)


        model = self.trainer(train_dataloader, val_dataloader)
        print('--start generating...')
        print(self.generate_joke(text, self.data['vocab']))

        #haven't implemented an evaluation function yet, program still runs.
        #return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

