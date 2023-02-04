'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import numpy as np


class Setting(setting):
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X_train, X_test = np.array(loaded_data['train']['X']), np.array(loaded_data['test']['X'])
        y_train, y_test = np.array(loaded_data['train']['y']), np.array(loaded_data['test']['y'])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        score = self.evaluate.evaluate()

        return score