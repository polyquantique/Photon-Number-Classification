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

        skip = config['network']["skip_elements"]
        folder = f"{config['files']['dataset']}"
        size = config['files']['input_dimension']
        files = listdir(folder)

        X = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])
        if skip > 1: X = X[:, 1::skip]

        self.data = torch.from_numpy(X).view(-1, 1, int(size / skip)).float().to(config['internal']['device'])
        self.keys = config.keys()

        if 'transformer' in config.keys():
            self.encoder_seq_len = config['transformer']['encoder_seq_len']
            #self.decoder_seq_len = decoder_seq_len
            #self.target_seq_len = target_seq_len
            

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, index):
        """

        """
        sample = self.data[index]

        if 'transformer' in self.keys:
            src, trg, trg_y = self.get_src_trg(
                sequence = sample,
                enc_seq_len = self.encoder_seq_len,
                )
            return src, trg, trg_y
        
        return sample
    

    def get_src_trg(self, sequence: torch.Tensor, enc_seq_len: int):
        """

        """
        target_seq_len = len(sequence) - enc_seq_len
        
        src = sequence[:enc_seq_len] 
        trg = sequence[enc_seq_len-1:len(sequence)-1]
        trg_y = sequence[-target_seq_len:]

        return src, trg, trg_y.squeeze(-1)




