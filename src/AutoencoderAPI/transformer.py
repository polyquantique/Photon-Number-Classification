from os import makedirs, listdir
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm

import torch

from .setup.networks.transformerAutoencoder import build_autoencoder
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .setup.train.generic import train as train_MSE
from .setup.train.triplet import train as train_Triplet
from .setup.validation.tripletValidation import validation
from .setup.validation.autoencoderValidation import validation
from .utils.files import save_all

#torch.use_deterministic_algorithms(True)
#torch.manual_seed(42)

class transformer():

    def __init__(self):
        self.device = None

    def setup(self, config):
        """
        # setup
        
        Load dataset from files and define the log file name.

        Parameters
        ----------
        - config : dict 
                - Dictionary containing the experiment parameters. 
        
        Returns
        -------
        - data : torch.tensor
            - Dataset.
        - log_path : str
            - Path where the results of the experiment are stored.

        Example
        -------

        Exemple of config :

        ```
        config_Transformer = {
            'files' : {
                    'dataset'                  : "Datasets/NIST (250)",
                    'path_save'                : 'Autoencoder Log/',
                    'input_dimension'          : 250
                    },
            'network' : {
                    'nhead'                    : 250,
                    'dropout'                  : 0.1,
                    'sequence_len'             : 1       
                    },
            'train' : {
                    'optimizer'                : 'Adam',
                    'criterion'                : 'MSELoss', 
                    'epochs'                   : 8,
                    'learning_rate'            : 1e-6
                    }
            }
        ```
        """
        try:
            if config['sweep']:
                folder_name = ""
        except:
            folder_name = "/run-" + datetime.now().strftime(r"%Y-%m-%d-%H-%M")

        log_path = f"{config['files']['path_save']}{folder_name}/fold 0"

        # Define device and runs on Cuda if is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define dataset
        sequence = config['network']["sequence_len"]
        size = config['files']['input_dimension']
        config['network']["embed_dim"] = int(size / sequence)

        folder = f"{config['files']['dataset']}"
        files = listdir(folder)

        X = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])

        data = torch.from_numpy(X).view(-1, sequence, size).float().to(self.device) #config['network']["embed_dim"]

        return config, data, log_path


    def split_dataset(self, data):
        """
        # split_dataset
        
        Split the dataset into a training, validation and testing set.
        The index of the sets are given as an output.

        Repartition of the original dataset:

        - Train : 50 %
        - Validation : 25 %
        - Test : 25 %


        Parameters
        ----------
        - data : torch.tensor 
                - Total dataset used for training, validating and testing the model.
  
        
        Returns
        -------
        - train_index : torch.tensor 
            - Train indexes
        - validation_index : torch.tensor 
            - Validation indexes
        - test_index : torch.tensor
            - Test indexes
        """
        len_ = data.size(0)
        index = torch.randperm(len_)
        
        index_split = torch.split(index, len_//4)
        train_index = torch.cat((index_split[0], index_split[1]))
        validation_index = index_split[2]
        test_index = index_split[3]

        return train_index, validation_index, test_index
                

    def run(self, config):
        """
        # run

        run(config)

        Execute a neural network experiment by creating an autoencoder neural network and training it to reproduce the it's input signal.
        Once it is trained, the encoder portion is used to associate each signal to a singular value. 
        This way, the network acts as a dimensionality reduction technique.

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
        
        Returns
        -------
        - None


        Example
        -------

        Exemple of config :

        ```
        config_Transformer = {
            'files' : {
                    'dataset'                  : "Datasets/NIST (250)",
                    'path_save'                : 'Autoencoder Log/',
                    'input_dimension'          : 250
                    },
            'network' : {
                    'nhead'                    : 250,
                    'dropout'                  : 0.1,
                    'sequence_len'             : 1       
                    },
            'train' : {
                    'optimizer'                : 'Adam',
                    'criterion'                : 'MSELoss', 
                    'epochs'                   : 8,
                    'learning_rate'            : 1e-6
                    }
            }
        ```
        """
        # Initialization of loss and result arrays
        loss = {'train_loss'        : [], 
                'validation_loss'   : [],
                'test_loss'         : []
                }

        config, data, log_path = self.setup(config)
        train_index, validation_index, test_index = self.split_dataset(data)

        network = build_autoencoder(config).float().to(self.device)
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        for epoch in tqdm(range(config['train']['epochs']) , desc="Epoch"):
        
            train_loss = train_MSE(network, data[train_index], optimizer, criterion)
            validation_loss = validation(network, data[validation_index], criterion)

            loss['train_loss'].append(train_loss) 
            loss['validation_loss'].append(validation_loss) 

        test_loss, results = validation(network, data[test_index], criterion, store=True)
        loss['test_loss'].append(test_loss)

        makedirs(log_path)
        save_all(log_path, network, results, loss, config)