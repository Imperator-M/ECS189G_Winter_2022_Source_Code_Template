'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self, show=False):
        # Load data
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        # Print data images
        if show:
            print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
            for pair in data['train']:
            # for pair in data['test']:
                plt.imshow(pair['image'], cmap="Greys")
                plt.show()
                print(pair['label'])

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for pair in data['train']:
            X_train.append(pair['image'])
            y_train.append(pair['label'])

        for pair in data['test']:
            X_test.append(pair['image'])
            y_test.append(pair['label'])

        #print('X_test:', np.array(X_test).shape)

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}