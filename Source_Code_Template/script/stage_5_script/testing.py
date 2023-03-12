from Dataset_Loader_Node_Classification import *

data_obj = Dataset_Loader("cora-small")
data_obj.dataset_name = "cora-small"
data_obj.dataset_source_folder_path = "../../../stage_5_data/stage_5_data/cora/"
data_obj.dataset_source_file_name = "link"

dictionary = data_obj.load()

print("Done!")