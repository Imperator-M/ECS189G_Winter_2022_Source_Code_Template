from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func

class TensorDataset(Dataset):
    def __init__(self, x, y):
        # Init
        self.x = x
        self.y = y

        # Normalize between 0 and 1
        self.x = self.x / 255.

        #One Hot Encoding
        #self.y = func.one_hot(self.y.long(), num_classes=10).to(float)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.shape[0]