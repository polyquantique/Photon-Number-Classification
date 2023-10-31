from os import makedirs, listdir
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm

import torch

from .setup.networks.autoencoder import build_autoencoder
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .setup.train.generic import train as train_MSE
from .setup.train.triplet import train as train_Triplet
from .setup.validation.tripletValidation import validation
from .utils.files import save_all
from .utils.clustering.kernelDensity import kernel_density

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

class recurrentTriplet():

    def __init__(self):
        self.device = None

    def setup(self, config):
        """
        Load dataset from files and define the folder parameters where 
        the results are stored.

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

        # Define device and run on Cuda if is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define dataset
        skip = config['train']["skip_elements"]
        folder = f"{config['files']['dataset']}"
        size = config['files']['input_dimension']
        files = listdir(folder)

        X = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])
        if skip > 1: 
            X = X[:, ::skip]
            X= X[:,:int(size / skip)]
        else : skip = 1

        
        
        data = torch.from_numpy(X).view(-1, 1, int(size / skip)).float().to(self.device)

        return data, log_path



    def split_dataset(self, data):
        """
        Split the dataset into a training, validation and testing set.
        The index of the sets is given as an output.

        Separation of the original dataset:

        - Train : 80 %
        - Validation : 10 %
        - Test : 10 %

        Parameters
        ----------
        data : torch.tensor 
            Total dataset used for training, validating and testing the model.
        
        Returns
        -------
        train_index : torch.tensor 
            Train indexes
        validation_index : torch.tensor 
            Validation indexes
        test_index : torch.tensor 
            Test indexes
        """
        len_ = data.size(0)
        index = torch.randperm(len_)
        
        index_split = torch.split(index, len_//10)
        train_index = torch.cat((index_split[:8]))
        validation_index = index_split[8]
        test_index = index_split[9]

        return train_index, validation_index, test_index



    def update_cluster(self, network, X, bw = (-5, -2, 20)):
        """    
        Update the labels associated with each sample of X considering the autoencoder dimensionality reduction.
        Kernel density estimation is used to assign the labels.
        The bandwidth is defined using a grid search over an array of possible values.

        Parameters
        ----------
        network : Pytorch sequential 
            Autoencoder neural network that is trained to reproduce its input signal.
        X : torch.tensor
            Input data.
        bw : tuple or numpy.array
            If bw is a tuple, it represents the parameters inside np.logspace(\*bw).
            Otherwise, an array can be used, this represents an array containing all 
            the possible bandwidth used in the kernel density estimation.

        Returns
        -------
        labels : torch.tensor
            Tensor of labels for every element in X.

        """

        with torch.no_grad():
            X_low_dim = network(X, encoding=True)
            X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)

            cl = kernel_density(X_low_dim, bw)

            return cl.labels

                

    def run(self, config):
        """
        Generate an autoencoder neural network and train it to reproduce the its input signal.
        Once it is trained, the encoder portion is used to associate each signal to a singular value. 
        This way, the network acts as a dimensionality reduction technique.

        In this context the triplet loss requires labeling of the low dimensional data in an usupervised scheme.
        An initial labeling of the data is done by training the network using the MSE loss and reaching a maximum accuracy.
        The labeling is updated every epoch using kernel density estimation.

        Parameters
        ----------
        config : dict
            Dictionary containing the experiment parameters. 
        
        Returns
        -------
        - None

        """
        # Initialization of loss and result arrays
        loss = {'train_loss'        : [], 
                'validation_loss'   : [],
                'test_loss'         : []
                }

        data, log_path = self.setup(config)
        learning_rate = config['train']['learning_rate']
        train_index, validation_index, test_index = self.split_dataset(data)
        network = build_autoencoder(config).float().to(self.device)
        
        
        config['train']['criterion'] = 'MSELoss'
        config['train']['learning_rate'] = 1e-4
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        for epoch in tqdm(range(4) , desc="Epoch MSE"):
            train_loss = train_MSE(network, data[train_index], optimizer, criterion)
        
        
        config['train']['criterion'] = 'TripletMSE'
        config['train']['learning_rate'] = learning_rate
        bw_cst = config['network']['bw_cst']
        
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        n_cluster = range(17,25)
        train_dataset = data[train_index]
        validation_dataset = data[validation_index]
        for epoch in tqdm(range(config['train']['epochs']) , desc="Epoch Triplet"):
            #, train_dataset
            train_labels = self.update_cluster(network, train_dataset, bw_cst)
            train_loss = train_Triplet(config, network, train_dataset, optimizer, criterion, train_labels)

            validation_labels = self.update_cluster(network, validation_dataset, bw_cst)
            validation_loss = validation(config['train']['alpha'], network, validation_dataset, criterion, validation_labels)

            loss['train_loss'].append(train_loss) # Triplet
            loss['validation_loss'].append(validation_loss) # Triplet

        test_labels = self.update_cluster(network, data[test_index], bw_cst)
        test_loss, results = validation(config['train']['alpha'], network, data[test_index], criterion, test_labels, store=True)
        loss['test_loss'].append(test_loss)

        makedirs(log_path)
        save_all(log_path, network, results, loss, config)
