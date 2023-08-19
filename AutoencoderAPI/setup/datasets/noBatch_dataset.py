import torch
from torch.utils.data import Dataset
from os import listdir
import numpy as np

class build_dataset(Dataset):
    """

    """
    def __init__(self, config):
        """

        """
        super().__init__()

        skip = config['train']["skip_elements"]
        folder = f"{config['files']['dataset']}"
        size = config['files']['input_dimension']
        files = listdir(folder)

        X = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])
        if skip > 1: X = X[:, 1::skip]

        self.data = torch.from_numpy(X).view(-1, 1, int(size / skip)).float().to(config['internal']['device'])
            

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, index):
        """

        """
        return self.data[index]
    



