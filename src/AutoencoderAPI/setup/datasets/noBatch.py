import torch
from torch.utils.data import Dataset
from os import listdir
import numpy as np

class build_dataset(Dataset):

    def __init__(self, config):
        """
        # build_dataset

        Pytorch class structure for datasets.
        Only use if the dataset if small enough to be loaded all in one instance.
        Batching is not available for this structure but allows for true randomization of the dataset.

        TODO:
        Combine Pytorch class structure for batching and tru random.

            Parameters
            ----------
            - config : dict 
                    - Dictionary containing the experiment parameters. 
            
            Returns
            -------
            - None
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
        """
        # __len__

        Number of elements in the dataset.

        Parameters
        ----------
        - None
        
        Returns
        -------
        - Length : int 

        """
        return len(self.data)

    def __getitem__(self, index):
        """
        # __getitem__

        Access item with its index.

            Parameters
            ----------
            - index : int  
                - Index of the desired sample.
            
            Returns
            -------
            - sample : torch.tensor
                - Dataset sample.
        """
        return self.data[index]
    



