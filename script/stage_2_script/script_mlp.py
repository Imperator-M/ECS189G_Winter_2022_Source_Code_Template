from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting import Setting
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train and test files', '')
    data_obj.train_dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.train_dataset_source_file_name = 'train.csv'
    data_obj.test_dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.test_dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('no cross validation', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(score))
    print('MLP Classification Report:')
    print(setting_obj.evaluate.classification_report())
    # Plot Performance
    fig, (acc_ax, loss_ax) = plt.subplots(2, figsize=(8, 6))
    # Plot performance over training data
    acc_ax.plot(setting_obj.method.accuracy_history, label='train_data')
    loss_ax.plot(setting_obj.method.loss_history, label='train_data')
    # Axis Lables
    acc_ax.set_ylabel('accuracy')
    loss_ax.set_ylabel('loss')
    loss_ax.set_xlabel('epochs')
    # Graph Legends
    acc_ax.legend()
    loss_ax.legend()
    #Show plot
    plt.show()

    print('************ Finish ************')
    # ------------------------------------------------------