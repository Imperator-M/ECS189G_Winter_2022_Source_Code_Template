'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    train_data = None
    train_dataset_source_folder_path = None
    train_dataset_source_file_name = None

    test_data = None
    test_dataset_source_folder_path = None
    test_dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        f = open(self.train_dataset_source_folder_path + self.train_dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_train.append(elements[1:])
            y_train.append(elements[0])
        f.close()

        X_test = []
        y_test = []
        f = open(self.test_dataset_source_folder_path + self.test_dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_test.append(elements[1:])
            y_test.append(elements[0])
        f.close()

        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}