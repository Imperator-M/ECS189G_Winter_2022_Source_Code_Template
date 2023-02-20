'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np



class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):

        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        print('training set size:', len(data['train']), 'testing set size:', len(data['test']))

        if (self.dataset_source_file_name == 'ORL'):
            train_data = []
            for pair in data["train"]:
                train_data.append((pair["image"], pair["label"] - 1))

            test_data = []
            test_data_y = []
            for pair in data["test"]:
                test_data.append((pair["image"], pair["label"] - 1))
                test_data_y.append(pair["label"] - 1)
        else:
            train_data = []
            for pair in data["train"]:
                train_data.append((pair["image"], pair["label"]))

            test_data = []
            test_data_y = []
            for pair in data["test"]:
                test_data.append((pair["image"], pair["label"]))
                test_data_y.append(pair["label"])

        return {'train': train_data, 'test': test_data, 'test_y': test_data_y}