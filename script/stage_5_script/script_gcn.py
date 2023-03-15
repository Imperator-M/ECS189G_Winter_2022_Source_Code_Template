from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Method_GCN import Method_GCN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting import Setting
import numpy as np
import torch

# ---- Graph Convolution Net script ----

# ---- parameter section -------------------------------
np.random.seed(42)
torch.manual_seed(42)
# ------------------------------------------------------

# ---- objection initialization section ---------------
data_obj = Dataset_Loader('node classification', '')
data_obj.dataset_source_folder_path = '../../data/stage_5_data/cora/'

method_obj = Method_GCN('graph convolutional net', '')

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_'
result_obj.result_destination_file_name = 'prediction_result'

setting_obj = Setting('setting', '')

evaluate_obj = Evaluate_Accuracy('accuracy', '')
# ------------------------------------------------------

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.dataset.dataset_name = 'cora'
setting_obj.print_setup_summary()
score = setting_obj.load_run_save_evaluate()
print('************ Overall Performance ************')
print('GCN Accuracy: ' + str(score))
print('************ Finish ************')
# ------------------------------------------------------
