'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import csv

class Dataset_Loader(dataset):
    data = None

    train_dataset_source_folder_path = None
    train_dataset_source_file_name = None

    test_dataset_source_folder_path = None
    test_dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X_train = []
        y_train = []

        with open(self.train_dataset_source_folder_path + self.train_dataset_source_file_name) as f:
            reader_obj = csv.reader(f)
            for line in reader_obj:
                intLine = [int(ele) for ele in line]
                X_train.append(intLine[1:])
                y_train.append(intLine[0])

        X_test = []
        y_test = []
        with open(self.test_dataset_source_folder_path + self.test_dataset_source_file_name) as f:
            reader_obj = csv.reader(f)
            for line in reader_obj:
                intLine = [int(ele) for ele in line]
                X_test.append(intLine[1:])
                y_test.append(intLine[0])
        f.close()

        return {'X_train': X_train, 'y_train': y_train, 'X_test':X_test, 'y_test':y_test}