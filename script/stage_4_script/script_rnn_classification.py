from code.stage_4_code.Dataset_Loader_classification import Dataset_Loader
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting import Setting
import numpy as np
import torch

#---- Recurrent Neural Net script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('classification', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'

    method_obj = Method_RNN('recurrent neural net', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('setting', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(score))
    print('************ Finish ************')
    # ------------------------------------------------------