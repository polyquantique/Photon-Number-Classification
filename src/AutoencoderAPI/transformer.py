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
                log_path = f"{config['files']['path_save']}/fold 0"
        except:
            folder_name = "/run-" + datetime.now().strftime(r"%Y-%m-%d-%H-%M")
            log_path = f"{config['files']['path_save']}{folder_name}/fold 0"
        

        config['internal'] = {}
        # Define device and run on Cuda if is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define dataset
        folder = f"{config['files']['dataset']}"
        size = config['files']['input_dimension']
        
        files = listdir(folder)

       
        # Define dataset
        sequence = config['network']["sequence_len"]
        config['network']["embed_dim"] = int(size / sequence)

        
        try:
            interval = config['train']["interval"]
            config['internal']['size_network'] = interval[1] - interval[0]
            interval1 = interval[0]
            interval2 = interval[1]
        except:
            config['internal']['size_network'] = config['files']['input_dimension']
            interval1 = 0
            interval2 = config['files']['input_dimension']

        try:
            skip = config['train']['skip_elements']
            if skip < 1: skip = 1
            config['internal']['size_network'] = int(config['internal']['size_network']/skip)
        except:
            skip = 1

        try:
            if config['files']['folder_type'] == 'npy':
                X = -1 *np.concatenate([np.load(f"{folder}/{file_name}").reshape((-1,size))[:,interval1:interval2:skip] for file_name in files])
            else:
                #X = -1 * np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size))[:w,interval1:interval2:skip] for w, file_name in zip(file_weight, files)]).astype("double")
                X = -1 * np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size))[:,interval1:interval2:skip] for file_name in files]).astype("double")
        except Exception as ex:
            print(ex)
            X = -1 * np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size))[:,interval1:interval2:skip] for file_name in files]).astype("double")

        try:
            X = (X - config['internal']['mean'])/config['internal']['std']
        except:
            config['internal']['mean'] = np.mean(np.copy(X))
            config['internal']['std'] = np.std(X)
            X = (X - config['internal']['mean'])/config['internal']['std']


        #X_ = np.copy(X)
        #for i in range(3,6):
        #    X_noise1 = [X__ + np.random.normal(0, 0.001* i, config['internal']['size_network']) for X__ in X_]
        #    X = np.concatenate([X, X_noise1])

        #X = X[np.max(X, axis=1) > 0]
        #condition = np.min(X, axis=1) < -1.5
        #X = X[condition]
        #condition = X[:,100] > -1.5
        #X = X[condition]

            
        data = torch.from_numpy(X).view(-1, 1, config['internal']['size_network']).float().to(self.device)







        try:
            if config['sweep']:
                folder_name = ""
        except:
            folder_name = "/run-" + datetime.now().strftime(r"%Y-%m-%d-%H-%M")

        log_path = f"{config['files']['path_save']}{folder_name}/fold 0"

    


       

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