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
        # print('loading data...')
        # f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        # data = pickle.load(f)
        # f.close()
        # print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
        # X_train = []
        # y_train = []
        #
        # X_test = []
        # y_test = []
        #
        # for pair in data["train"]:
        #     X_train.append(pair["image"])
        #     y_train.append(pair["label"])
        #
        # for pair in data["test"]:
        #     X_test.append(pair["image"])
        #     y_test.append(pair["label"])
        #
        # print(np.shape(X_train))
        # print(np.shape(y_train))
        # print(np.shape(X_test))
        # print(np.shape(y_test))
        print('loading data...')
        f = open('../../data/stage_3_data/MNIST', 'rb')
        data = pickle.load(f)
        f.close()
        print('training set size:', len(data['train']), 'testing set size:', len(data['test']))

        train_data = []
        for pair in data["train"]:
            train_data.append((pair["image"], pair["label"]))

        test_data = []
        test_data_y = []
        for pair in data["test"]:
            test_data.append((pair["image"], pair["label"]))
            test_data_y.append(pair["label"])

        return {'train': train_data, 'test': test_data, 'test_y': test_data_y}
        # for pair in data['train']:
        #     # for pair in data['test']:
        #     plt.imshow(pair['image'], cmap="Greys")
        #     plt.show()
        #     print(pair['label'])


        # df = pd.read_csv(self.training_dataset_source_folder_path + self.training_dataset_source_file_name, header=None)
        # X_train = df.iloc[:, 1:]
        # y_train = df[df.columns[0]]
        #
        # df = pd.read_csv(self.testing_dataset_source_folder_path + self.testing_dataset_source_file_name, header=None)
        # X_test = df.iloc[:, 1:]
        # y_test = df[df.columns[0]]
        #
        # return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
