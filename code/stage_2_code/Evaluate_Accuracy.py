'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

    def classification_report(self):
        return classification_report(self.data['true_y'], self.data['pred_y'], labels=np.unique(self.data['true_y']))