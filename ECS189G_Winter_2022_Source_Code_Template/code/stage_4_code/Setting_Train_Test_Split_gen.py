'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        # X_train = loaded_data['X_train']
        # X_test = loaded_data['X_test']
        # y_train = loaded_data['y_train']
        # y_test = loaded_data['y_test']
        # wToi = loaded_data['wToi']
        # iTow =loaded_data['iTow']
        # run MethodModule
        # self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test},
        #                     'wToi':wToi, 'iTow':iTow}

        dataset = loaded_data['dataset']
        wToi = loaded_data['wToi']
        iTow = loaded_data['iTow']
        self.method.data = {'dataset': dataset, 'wToi': wToi, 'iTow': iTow}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        # self.evaluate.data = learned_result
        
        return None, None

        