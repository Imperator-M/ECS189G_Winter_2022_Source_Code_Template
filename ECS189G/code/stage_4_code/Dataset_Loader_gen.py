'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        sentences = []
        all_words = set()
        for line in f:
            line = line.split('"')[1]
            word_list = re.findall(r"[\w']+|[.,!?;]", line)
            for word in word_list:
                all_words.add(word)
            sentences.append(word_list)
        all_words.add('')

        word_dict = {w: i for i, w in enumerate(all_words)}
        number_dict = {i: w for i, w in enumerate(all_words)}

        df = pd.DataFrame(columns=['1word', '2word', '3word', '4word', '5word', '6word'])
        for sen in sentences:
            for i in range(len(sen) - 4):
                if i + 5 == len(sen):
                    df.loc[len(df.index)] = [word_dict[sen[i]], word_dict[sen[i + 1]], word_dict[sen[i + 2]],
                                             word_dict[sen[i + 3]], word_dict[sen[i + 4]], word_dict['']]
                else:
                    df.loc[len(df.index)] = [word_dict[sen[i]], word_dict[sen[i + 1]], word_dict[sen[i + 2]],
                                             word_dict[sen[i + 3]], word_dict[sen[i + 4]], word_dict[sen[i + 5]]]


        # train, test = train_test_split(df, test_size=0.2)
        # X_train = train[['1word', '2word', '3word', '4word']].to_numpy()
        # y_train = train['correct1'].to_numpy()
        # X_test = test[['1word', '2word', '3word', '4word']].to_numpy()
        # y_test = test['correct1'].to_numpy()
        # return {'X_train': X_train, 'y_train': y_train, 'X_test':X_test, 'y_test':y_test, 'wToi':word_dict, 'iTow':number_dict}

        torchDataset = df_to_pyTorchDataset(df)
        return {'dataset': torchDataset, 'wToi':word_dict, 'iTow':number_dict}

class df_to_pyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.x = df.iloc[:, 0:5].to_numpy()
        self.y = df.iloc[:, 1:6].to_numpy()
        self.x = torch.from_numpy(self.x).long()
        self.y = torch.from_numpy(self.y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]