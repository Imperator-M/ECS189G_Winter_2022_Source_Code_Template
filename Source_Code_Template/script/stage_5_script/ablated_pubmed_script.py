from code.stage_5_code.Dataset_Loader import *
from code.stage_5_code.Ablated_PubMed import *
from code.stage_5_code.Result_Saver import *
from code.stage_5_code.Evaluate_Accuracy import *
from code.stage_5_code.Setting_Train_Test_Split import *

import numpy as np
import torch

if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('pubmed', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/pubmed'
    data_obj.dataset_source_file_name = 'pubmed'

###############################################################################################
    method_obj = Ablated_PubMed_Method('gcn', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_'
    result_obj.result_destination_file_name = 'prediction_result_ablated_Pubmed'

    ##################################################################
    setting_obj = Setting_Train_Test_Split("", '')
    ##################################################################

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('GCN Accuracy: ' + str(mean_score[0]))
    print('GCN Precision: ' + str(mean_score[1]))
    print('GCN Recall: ' + str(mean_score[2]))
    print('GCN F1: ' + str(mean_score[3]))
    print('************ Finish ************')