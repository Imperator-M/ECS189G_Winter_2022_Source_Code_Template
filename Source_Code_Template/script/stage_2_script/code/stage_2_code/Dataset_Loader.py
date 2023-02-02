'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import time

class Dataset_Loader(dataset):
    data = None
    
    dataset_train_folder_path = None
    dataset_train_file_name = None
    
    dataset_test_folder_path = None
    dataset_test_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        
        X_train = []
        y_train = []
        
        X_test = []
        y_test = []

        f = open(self.dataset_train_folder_path + self.dataset_train_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            line = line.replace(',', ' ')
            #print(line)
            elements = [int(i) for i in line.split(' ')]
            X_train.append(elements[1:])
            y_train.append(elements[0])
            #print("Here's X: ", X_train)
            #print("Here's Y: ", y_train)
        f.close()

        f = open(self.dataset_test_folder_path + self.dataset_test_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            line = line.replace(',', ' ')
            #print(line)
            elements = [int(j) for j in line.split(' ')]
            X_test.append(elements[1:])
            y_test.append(elements[0])
            #print("Here's X: ", X_train)
            #print("Here's Y: ", y_train)
        f.close()


        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}