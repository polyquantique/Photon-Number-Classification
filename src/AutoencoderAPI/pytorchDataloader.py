from os import makedirs
from datetime import datetime
from tqdm.notebook import tqdm

from sklearn.model_selection import KFold

import torch
import torch.onnx
from torch.utils.data import SubsetRandomSampler, DataLoader

from .setup.datasets.noBatch import build_dataset 
from .setup.networks.autoencoder import build_autoencoder 
from .setup.train.generic import train
from .setup.validation.autoencoderValidation import validation
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .utils.files import save_all

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

class pytorchDataloader:

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
        data = build_dataset(config)

        return data, log_path


    def run(self, config):
        """
        # run

        Execute a neural network experiment by creating an autoencoder neural network and training it to reproduce the it's input signal.
        Once it is trained, the encoder portion is used to associate each signal to a singular value. 
        This way, the network acts as a dimensionality reduction technique.

        Parameters
        ----------
        - build_autoencoder : class
                - Pytorch neural network class with a `__init__` definition and `forward` process.
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary.
        
        Returns
        -------
        - None
        """
        data, log_path = self.setup(config)

        fold = KFold(n_splits=config['train']['k-fold'], shuffle=True, random_state=42).split(data)

        for fold_index, (train_index, test_index) in tqdm(enumerate(fold), 
                                                        desc="Fold", 
                                                        total=config['train']['k-fold']):

            # Initialization of loss and result arrays
            loss = {'train_loss'        : [], 
                    'validation_loss'   : [],
                    'test_loss'         : [],
                    }
    
            network = build_autoencoder(config).float().to(config['internal']['device'])
            optimizer = build_optimizer(network, config)
            criterion = build_criterion(config)

            train_sampler = SubsetRandomSampler(train_index)
            validation_sampler = SubsetRandomSampler(test_index[range(0, len(test_index), 2)])
            test_sampler = SubsetRandomSampler(test_index[range(1, len(test_index), 2)])
        
            train_loader = DataLoader(data, batch_size=1, sampler=train_sampler)  #batch_size=config['train']['batch_number']
            validation_loader = DataLoader(data, batch_size=1, sampler=validation_sampler)
            test_loader = DataLoader(data, batch_size=1, sampler=test_sampler)

            for epoch in tqdm(range(config['train']['epochs']) , desc="Epoch"):
                
                train_loss = train(network, train_loader, optimizer, criterion)
                validation_loss = validation(config, network, validation_loader, criterion)

                loss['train_loss'].append(train_loss)
                loss['validation_loss'].append(validation_loss)

            test_loss, results = validation(config, network, test_loader, criterion, store=True)
            loss['test_loss'].append(test_loss)
            
            fold_path = f"{log_path}/fold {fold_index}"
            makedirs(fold_path)

            save_all(fold_path, network, results, loss, config)



 
    