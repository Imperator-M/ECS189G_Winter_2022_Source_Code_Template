import torch
from torch.utils.data import Dataset

class JokeDataset(Dataset):
    def __init__(self, data, vocab, seq_length=5):
        self.vocab = vocab
        self.seq_length = seq_length
        self.data = self.process_data(data)

    def process_data(self, data):
        # Convert each joke text into a numerical sequence
        numerical_data = []
        for tokenized_sentence in data:
            numerical_sentence = []
            for token in tokenized_sentence:
                if token in self.vocab:
                    numerical_sentence.append(self.vocab[token])
                else:
                    numerical_sentence.append(self.vocab['<unk>'])
            numerical_sentence.append(self.vocab['<eos>'])
            numerical_data.append(numerical_sentence)

        # Split each numerical sequence into input/target pairs
        input_seqs = []
        target_seqs = []
        for numerical_sentence in numerical_data:
            for i in range(0, len(numerical_sentence) - self.seq_length):
                input_seq = numerical_sentence[i:i + self.seq_length]
                target_seq = numerical_sentence[i+1:i + self.seq_length+1]
                input_seqs.append(input_seq)
                target_seqs.append(target_seq)

        return input_seqs, target_seqs

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.data[0][idx])
        target_seq = torch.tensor(self.data[1][idx])
        # print("input_seq:", self.data[0][idx])
        # print("target_seq:", self.data[1][idx])
        # print("input_seq shape:", input_seq.shape)
        # print("target_seq shape:", target_seq.shape)

        return input_seq, target_seq

    def __len__(self):
        return len(self.data[0])