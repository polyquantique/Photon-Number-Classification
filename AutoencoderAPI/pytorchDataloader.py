from os import makedirs
import matplotlib.pyplot as plt
from datetime import datetime
import random
from tqdm.notebook import tqdm

from sklearn.model_selection import KFold

import torch
import torch.onnx
from torch.utils.data import SubsetRandomSampler, DataLoader

from .setup.datasets.noBatch_dataset import build_dataset 
from .setup.networks.autoencoder import build_autoencoder 
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .utils.files import save_all

plt.style.use("seaborn-pastel")

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

class pytorchDataloader:

    def __init__(self) -> None:
        pass


    def train_epoch(self, network, X_train, optimizer, criterion):
        """
        # train_epoch

        train_epoch(config, network, X_train, optimizer, criterion)

        Training process executed for every epoch. The actions consists of setting the gradients to zero, 
        making predictions for the batch, computing the loss and its gradient and updating the weights and biases.

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary
        - network : Pytorch sequential : 
                - Autoencoder neural network that is trained to reproduce its input signal.
        - X_train : torch.tensor
                - Input samples used to train the autoencoder.
        - optimizer : Pytorch optimizer
                - Optimizer used for training.
        - criterion : Pytorch criterion
                - Criterion used for training.
        
        Returns
        -------
        - Average loss : float
                - Average loss of the training process (loss of one epoch).
        """
        cumu_loss = 0
        network.train()
        list_ = range(X_train.__len__())
        for input_ in X_train:
            
            # Zero gradient
            optimizer.zero_grad()
            # Forward
            output_ = network(input_)
            # Criterion
            loss = criterion.forward(output_, input_, X_train, network, list_)
            # Backward
            loss.backward()
            optimizer.step()
            # Loss
            cumu_loss += loss.item()

        return cumu_loss / len(X_train)
    

    def validation_test(self, config, network, X, criterion, store=False):
        """
        # validation_test

        validation_test(config, network, X, criterion, store=False)

        Validation or testing of the network.
        This action consists of a forward pass of the network using the desired samples.
        In this step the intermediate results can be stored in a `results` dictionary.
        The results consists of the input, the encoder output and the decoder output.


        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary.
        - network : Pytorch sequential : 
                - Autoencoder neural network that is trained to reproduce its input signal.
        - X : torch.tensor
                - Input samples used to validate or test the autoencoder.
        - criterion : Pytorch criterion
                - Criterion used for training.
        - store : bool
                - If `True` the intermediate results are stored in the `results` dictionary.
        
        Returns
        -------
        - store = `True` : 
            - Average loss : float
                - Average loss of the training process (loss of one epoch).
            - results : dict
                - Dictionary containing the intermediate results of the process 
                  (input, encoder output and decoder output)
        - store = `False` : 
            - Average loss : float
                - Average loss of the training process (loss of one epoch).
        """
        cumu_loss = 0
        list_ = range(X.__len__())

        if store:
            results = {'encode' : [],
                       'input'  :  [],
                       'decode' : []
                       }

        network.eval()
        with torch.no_grad():
            for index, data in enumerate(X):

                if store:
                    encode = network(data, encoding=True)
                    decode = network(encode, decoding =True)

                    save_encode = torch.clone(encode).numpy()
                    results[f'encode'].append(save_encode[0,0,0])

                    if index < 2:
                        results['input'].append(data.clone().view(-1).numpy())
                        results['decode'].append(decode.clone().view(-1).numpy())

                else:
                    decode = network(data)
                    
                loss = criterion.forward(decode, data, X, network, list_)
                cumu_loss += loss.item()

        if store:
            return cumu_loss / len(X), results
        
        return cumu_loss / len(X)
    

    def setup(self, config):

        # log path and folder creation to store results
        if config['sweep']['sweep_name'] is not None:
            log_path = f"{config['files']['path_save']}/{config['sweep']['sweep_name']}/sweep {str(config['internal']['sweep_index']).rjust(config['internal']['number_size'], '0')}"
        else:
            config['internal'] = {}
            folder_name = datetime.now().strftime(r"%Y-%m-%d-%H-%M")
            log_path = f"{config['files']['path_save']}/run-{folder_name}"

        # Define device and runs on Cuda is available
        config['internal']['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define dataset
        data = build_dataset(config)

        return config, data, log_path


    def run(self, config):
        """
        # run

        run(build_autoencoder, config)

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
        config, data, log_path = self.setup(config)

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
                
                train_loss = self.train_epoch(network, train_loader, optimizer, criterion)
                validation_loss = self.validation_test(config, network, validation_loader, criterion)

                loss['train_loss'].append(train_loss)
                loss['validation_loss'].append(validation_loss)

            test_loss, results = self.validation_test(config, network, test_loader, criterion, store=True)
            loss['test_loss'].append(test_loss)
            
            fold_path = f"{log_path}/fold {fold_index}"
            makedirs(fold_path)

            save_all(fold_path, network, results, loss, config)



 
    