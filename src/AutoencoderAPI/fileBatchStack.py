import numpy as np
from os import listdir, makedirs
from datetime import datetime

import torch
from sklearn.model_selection import KFold, train_test_split

from tqdm.notebook import tqdm
import warnings

from .setup.networks.autoencoder import build_autoencoder 
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .utils.files import save_all, open_object, save_object
from .autoencoderGaussianMixture import autoencoder_gaussianMixture

#torch.use_deterministic_algorithms(True)
#torch.manual_seed(42)



class fileBatchStack:

    def __init__(self):
        pass


    def custom_Kfold(self, config):
        """
        Create a K-fold cross validation set up by creating a list of training, validation and test files.
        The test files are meant to be used to test the model after the training and validation steps.
        The test files stay the same across the K folds. 
        The training and validation are defined to create K folds and each fold can be separated into batches.

        Parameters
        ----------
        config : dict
            Dictionary containing the experiment parameters. 

        Returns
        -------
        train_files : list
            List of numpy arrays containing the name of the training files used in the batches and folds.
            The list is organised so each element of the list is associated to a fold and the every sub array 
            is a batch.
        validation_files : list
            List of numpy arrays containing the name of the validation files used in the batches and folds.
            The list is organised so each element of the list is associated to a fold and the every sub array 
            is a batch.
        test_files : list
            List containing all the test files.
        config : dict
            Updated dictionary. Define `input_dimension` of the autoencoder
        """

        dB = np.load('Datasets/SNSPD/Paderborn/db_shuffled.npy')

        folder = f"{config['files']['dataset']}"
        files = [file_dB for index, file_dB in enumerate(listdir(folder)) if dB[index] == config['dB']]

        fold = KFold(n_splits=config['train']['k-fold'],shuffle=True,random_state=42) #config['train']['k-fold']
        train_validation_files, test_files = train_test_split(files,train_size=0.9,shuffle=True)
        splits = fold.split(train_validation_files)

        train_files = []
        validation_files = []

        for train_index, validation_index in splits:
            train_files.append(np.take(train_validation_files, train_index))
            validation_files.append(np.take(train_validation_files, validation_index))

        train_batch_number = validation_batch_number = config['train']['batch_number']
        
        batch_max = len(train_files[0])
        if train_batch_number >= batch_max:
            warnings.warn(f"Batch number too high, was instead set to {batch_max} (maximum)")
            train_batch_number = batch_max
        
        batch_max = len(validation_files[0])
        if validation_batch_number >= batch_max:
            validation_batch_number = batch_max
        
        train_files = [np.array_split(train_fold, train_batch_number) for train_fold in train_files]
        validation_files = [np.array_split(validation_fold, validation_batch_number) for validation_fold in validation_files]

        return train_files, validation_files, test_files, config

    

    def custom_dataloader(self, config, files):
        """

        Creates a Pytorch tensor containing all the batch samples of a specific fold.

        Parameters
        ----------
        config : dict
                Dictionary containing the experiment parameters. 
        files : list
                List of files used in the batch. 
                All the samples inside the files will be stored as a Pytorch tensor.
                To reduce the memory requirements increase the batch number in the configuration dictionary.

        Returns
        -------
        samples : torch.tensor
                Three-dimensional tensors containing the batch samples.
                Tensor of shape (N,0,S), where N is the number of samples and S is the size of each sample.    
        """
        
        folder = f"{config['files']['dataset']}"
        size = config['files']['input_dimension']
        size_network = config['internal']['size_network']

        try:
            if config['files']['folder_type'] == 'npy':
                TES = np.concatenate([np.load(f"{folder}/{file_name}").reshape((-1,size)) for file_name in files])
            else:
                TES = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])
        except Exception as ex:
            TES = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])

        try:
            interval = config['train']["interval"]
            TES = TES[:, interval[0]:interval[1]]
        except:
            pass
    
        try:
            skip = config['train']["skip_elements"]
            if skip > 1: 
                TES = TES[:, 1::skip]
            else : skip = 1
        except:
            skip = config['train']["skip_elements"] = 1

        try:
            TES = (TES - config['internal']['mean'])/config['internal']['std']
        except:
            config['internal']['mean'] = np.mean(TES)
            config['internal']['std'] = np.std(TES)
            TES = (TES - config['internal']['mean'])/config['internal']['std']
        
        
        #if self.loop > 2:
        TES = TES[np.min(TES, axis=1) < -0.5]
        TES = TES[np.max(TES, axis=1) > 1] #####
        #self.loop += 1

        np.random.shuffle(TES)
        
        return torch.from_numpy(TES).view(-1, 1, size_network).float()

        

    def train_epoch(self, config, network, X_train, optimizer, criterion):
        """

        Training process executed for every epoch. The action consists of setting the gradients to zero, 
        making predictions for the batch, computing the loss and its gradient and updating the weights and biases.

        Parameters
        ----------
        config : dict
                Dictionary containing the experiment parameters. 
        network : Pytorch sequential : 
                Autoencoder neural network that is trained to reproduce its input signal.
        X_train : torch.tensor
                Input samples used to train the autoencoder.
        optimizer : Pytorch optimizer
                Optimizer used for training.
        criterion : Pytorch criterion
                Criterion used for training.
        
        Returns
        -------
        Average loss : float
                Average loss of the training process (loss of one epoch).
        """
        cumu_loss = 0
        network.train()
        list_ = range(X_train.size(0))
        for input_ in tqdm(X_train, total=len(X_train)): #X_train:
            # Use cuda if available
            input_ = input_.float().to(config['internal']['device'])
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

        return cumu_loss, len(X_train)

    

    def validation_test(self, config, network, X, criterion, store=False):
        """

        Validation or testing of the network.
        This action consists of a forward pass of the network using the desired samples.
        In this step the intermediate results can be stored in a `results` dictionary.
        The result consists of the input, the encoder output and the decoder output.


        Parameters
        ----------
        config : dict
                Dictionary containing the experiment parameters. 
        network : Pytorch sequential : 
                Autoencoder neural network that is trained to reproduce its input signal.
        X : torch.tensor
                Input samples used to validate or test the autoencoder.
        criterion : Pytorch criterion
                Criterion used for training.
        store : bool
                If `True` the intermediate results are stored in the `results` dictionary.
        
        Returns
        -------
        store = `True` : 
            Average loss : float
                Average loss of the training process (loss of one epoch).
            results : dict
                Dictionary containing the intermediate results of the process 
                (input, encoder output and decoder output)
        store = `False` : 
            Average loss : float
                Average loss of the training process (loss of one epoch).
        """
        cumu_loss = 0
        list_ = range(X.size(0))

        if store:
            results = {'encode' : [],
                       'input'  : [],
                       'decode' : []
            }

        network.eval()
        with torch.no_grad():
            for index, input_ in enumerate(X):
                # Use cuda if available
                input_ = input_.float().to(config['internal']['device'])

                if store:
                    encode = network(input_, encoding=True)
                    decode = network(encode, decoding =True)

                    results['encode'].append(encode.cpu().clone().numpy()[0])

                    if index < 2:
                        results['input'].append(input_.cpu().clone().numpy()[0])
                        results['decode'].append(decode.cpu().clone().numpy()[0])

                else:
                    decode = network(input_)
                    
                loss = criterion.forward(decode, input_, X, network, list_)
                cumu_loss += loss.item()

        if store:
            return cumu_loss / len(X), results

        return cumu_loss, len(X)

    
    def setup(self, config):

        config['internal'] = {}

        try:
            if config['stack']:
                config['stack'] = True
            else:
                config['stack'] = False
        except:
            config['stack'] = False

        # log path and folder creation to store results
        try:
            if config['files']['run_name'] != None:  
                log_path = f"{config['files']['path_save']}/{config['files']['stack']}"
            else:          
                folder_name = datetime.now().strftime(r"%Y-%m-%d-%H-%M")
                log_path = f"{config['files']['path_save']}/run-{folder_name}"
        except:
            folder_name = datetime.now().strftime(r"%Y-%m-%d-%H-%M")
            log_path = f"{config['files']['path_save']}/run-{folder_name}"
            

        try:
            interval = config['train']["interval"]
            config['internal']['size_network'] = interval[1] - interval[0]
        except:
            config['internal']['size_network'] = config['files']['input_dimension']

        try:
            skip = config['train']['skip_elements']
            if skip < 1: skip = 1
            config['internal']['size_network'] = int(config['internal']['size_network']/skip)
        except:
            pass
            
        config['internal']['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        return log_path, config



    def stack_filter(self, config, log_path, X):
        """
        
        """
        #number_sample = len(X)
        #output_label = np.zeros(number_sample)
        #index_samples = range(number_sample)
        
        for index, model in enumerate(config['models']):

            path = f"{log_path}/{model}/fold 0"
            config_load = open_object(f"{path}/log.bin")
            space_separation = open_object(f"{path}/space.bin")
            size_network = config_load['internal']['size_network']

            network = build_autoencoder(config_load).float().to(config['internal']['device'])
            network.load_state_dict(torch.load(f"{path}/model.pt"))

            network.eval()
            with torch.no_grad():
                #X_pytorch = torch.from_numpy(X).view(-1, 1, size_network).float().to(config['internal']['device'])
                X = X.float().to(config['internal']['device'])
                X_low_dim = network(X, encoding=True)
                #X_low_dim = X_low_dim#.numpy().reshape(-1, 1)

                #X_low = self.flip * X_low

            labels = torch.searchsorted(space_separation, X_low_dim)#.cpu()
            condition = labels == 0
            
            #output_label[index_samples[condition]] = index
            
            #index_samples = index_samples[not condition]
            X = torch.masked_select(X, torch.logical_not(condition)).view(-1, 1, size_network)

        return X


    def run(self, config):
        """
        Execute a neural network experiment by creating an autoencoder neural network and training it to reproduce its input signal.
        Once it is trained, the encoder portion is used to associate each signal to a singular value. 
        This way, the network acts as a dimensionality reduction technique.

        Parameters
        ----------
        - build_autoencoder : class
                - Pytorch neural network class with a `__init__` definition and `forward` process.
        - config : dict
                - Dictionary containing the experiment parameters. 
        
        Returns
        -------
        - None
        """
        
        log_path, config = self.setup(config)

        train_files, validation_files, test_files, config = self.custom_Kfold(config)
        
        for fold_index in tqdm(range(config['train']['k-fold']), desc="Fold"):

            # Initialization of loss and result arrays
            loss = {'train_loss'        : [],
                    'validation_loss'   : [],
                    'test_loss'         : [],
                    'average_test_loss' : []
                    }
   
            network = build_autoencoder(config).float().to(config['internal']['device'])
            optimizer = build_optimizer(network, config)
            criterion = build_criterion(config)

            for epoch in tqdm(range(config['train']['epochs']), desc="Epoch"):
                
                train_loss = validation_loss = train_number = validation_number = 0

                for batch_files in train_files[fold_index]:   
                
                    X_train = self.custom_dataloader(config, batch_files)

                    if config['stack']:
                        X_train = self.stack_filter(config, log_path, X_train) 
                    
                    train_loss_, train_number_ = self.train_epoch(config, network, X_train, optimizer, criterion)
                    train_loss += train_loss_
                    train_number += train_number_

                for batch_files in validation_files[fold_index]:

                    X_validation = self.custom_dataloader(config, batch_files)

                    if config['stack']:
                        X_validation = self.stack_filter(config, log_path, X_validation)

                    validation_loss_, validation_number_ = self.validation_test(config, network, X_validation, criterion)
                    validation_loss += validation_loss_
                    validation_number += validation_number_

                loss['train_loss'].append(train_loss/train_number)
                loss['validation_loss'].append(validation_loss/validation_number)

            X_test = self.custom_dataloader(config, test_files)
            if config['stack']:
                        X_test = self.stack_filter(config, log_path, X_test)
            test_loss , results = self.validation_test(config, network, X_test, criterion, store=True)
            loss['test_loss'].append(test_loss)
    
            fold_path = f"{log_path}/{config['files']['run_name']}/fold {fold_index}"
            makedirs(fold_path)
            save_all(fold_path, network, results, loss, config)


            agm = autoencoder_gaussianMixture(fold_path)
            agm.fit(X_train.cpu().numpy(),
                    bins_plot = 5_000,
                    plot_density = True,
                    plot_cluster = True,
                    plot_traces = False,
                    plot_traces_average = False,
                    plot_cross_talk = False,
                    bw_cst = [config['bw']],
                    filter_input = False,
                    skip = 1,
                    flip = config['flip'])#config['flip'])
            
            save_object(agm.mins,f'{fold_path}/space')

        