from code.stage_5_code.Result_Loader import Result_Loader

result_obj = Result_Loader('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_'
result_obj.result_destination_file_name = 'prediction_result'

result_obj.load()
print('Result:', result_obj.data)