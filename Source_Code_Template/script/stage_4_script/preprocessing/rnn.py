import torch

class RNN(torch.nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)        
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        hidden.squeeze_(0)
        output = self.fc(hidden)
        return output