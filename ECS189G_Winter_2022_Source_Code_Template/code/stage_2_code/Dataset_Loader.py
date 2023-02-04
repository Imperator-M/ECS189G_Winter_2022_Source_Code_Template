'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import csv
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class Dataset_Loader(dataset):
    data = None
    training_dataset_source_folder_path = None
    training_dataset_source_file_name = None

    testing_dataset_source_folder_path = None
    testing_dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        df = pd.read_csv(self.training_dataset_source_folder_path + self.training_dataset_source_file_name, header=None)
        X_train = df.iloc[:, 1:]
        y_train = df[df.columns[0]]

        df = pd.read_csv(self.testing_dataset_source_folder_path + self.testing_dataset_source_file_name, header=None)
        X_test = df.iloc[:, 1:]
        y_test = df[df.columns[0]]

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
