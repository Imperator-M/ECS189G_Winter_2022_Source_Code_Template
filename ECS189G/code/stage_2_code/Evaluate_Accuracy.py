'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return (accuracy_score(self.data['true_y'], self.data['pred_y']),
                precision_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                recall_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                f1_score(self.data['true_y'], self.data['pred_y'], average='macro'))
        