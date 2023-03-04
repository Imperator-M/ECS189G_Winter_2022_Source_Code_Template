from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split

import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train', '')
    data_obj.training_dataset_source_folder_path = '../../data/stage_4_data/'
    data_obj.training_dataset_source_file_name = 'train.csv'

    data_obj.testing_dataset_source_folder_path = '../../data/stage_4_data/'
    data_obj.testing_dataset_source_file_name = 'test.csv'

    method_obj = Method_RNN_Classification('recurrent neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result_Classification'

    ##################################################################
    setting_obj = Setting_Train_Test_Split("", '')  ##################
    ##################################################################

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(mean_score[0]) + ' +/- ' + str(std_score))
    print('RNN Precision: ' + str(mean_score[1]) + ' +/- ' + str(std_score))
    print('RNN Recall: ' + str(mean_score[2]) + ' +/- ' + str(std_score))
    print('RNN F1: ' + str(mean_score[3]) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------


