'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Setting_Train_Test_Split(setting):
    fold = 3

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        tokenized_data, vocab = loaded_data['tokenized_data'], loaded_data['vocabulary']
        train_ratio = 0.8
        train_size = int(train_ratio * len(tokenized_data))
        val_size = len(tokenized_data) - train_size
        train_data, val_data = tokenized_data[:train_size], tokenized_data[train_size:]

        # run MethodModule
        self.method.data = {'train': train_data, 'validate': val_data, 'vocab': vocab}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

