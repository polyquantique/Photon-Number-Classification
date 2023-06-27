import numpy as np
from os import listdir, makedirs
import matplotlib.pyplot as plt
from datetime import datetime
from  random import choices, choice

import torch
import torch.onnx
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, random_split

from sklearn.model_selection import KFold, train_test_split

from tqdm.notebook import tqdm
import pickle
import warnings

plt.style.use("seaborn-pastel")

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)



class build_autoencoder(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        # build_autoencoder

        __init__(config)

        Build a Pytorch autoencoder based on a CNN (convolution neural network) architecture with desired caracteristics. 

        Parameters
        ----------
        config : dict
                Dictionary containing the CNN desired caracteristics. 
                See the `autoencoder` class for more details on the config dictionary

        Returns
        -------
        None
        """
        super(build_autoencoder, self).__init__()
        
        if config['run']['layer_number'] % 2 != 0:
            print("Invalid number of layer (needs to be even number)")

        # Layer type
        layer_type_dict = {
            "Linear" : nn.Linear
        }
        layer = layer_type_dict[config['network']['layer_type']]

        # Activation function
        activation_dict = {
            "ReLU"       : nn.ReLU,
            "Sigmoid"    : nn.Sigmoid,
            "CELU"       : nn.CELU, 
            "Softmax"    : nn.Softmax,
            "Softmin"    : nn.Softmin,
            "Hardshrink" : nn.Hardshrink,
            "LeakyReLU"  : nn.LeakyReLU,
            "ELU"        : nn.ELU,
            "LogSigmoid" : nn.LogSigmoid,
            "PReLU"      : nn.PReLU,
            "GELU"       : nn.GELU,
            "SiLU"       : nn.SiLU,
            "Mish"       : nn.Mish,
            "Softplus"   : nn.Softplus,
            "Softsign"   : nn.Softsign,
            "Tanh"       : nn.Tanh,
            "GLU"        : nn.GLU,
            "Threshold"  : nn.Threshold,
        }

        # Number of layer
        #encoder_layers = np.linspace(config["input_dimension"], config["output_dimension"], int(config['layer_number'] / 2 + 1), dtype=int)
        #layer_list = np.concatenate((encoder_layers, np.flip(encoder_layers)[1:]))
        skip = config['network']["skip_elements"]
        size = config['files']['input_dimension']
        layer_list = config['run']['layer_list']
        layer_list[0] = layer_list[-1] = int(size / skip)
        activation_list = config['run']['activation_list']
        
        # Build network
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        #self.encoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))
        #self.encoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))
        
        for index, activation_type in enumerate(activation_list):
            if index < len(activation_list) // 2 + 1:
                self.encoder.append(layer(layer_list[index], layer_list[index+1]))
                self.encoder.append(activation_dict[activation_type]())
            else:
                self.decoder.append(layer(layer_list[index], layer_list[index+1]))
                self.decoder.append(activation_dict[activation_type]())
            
        self.decoder.append(layer(layer_list[-2], layer_list[-1]))

        #self.decoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))
        #self.decoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))

    def forward(self, X, encoding=False, decoding=False) -> any:
        """
        # forward

        forward(X, encoding=False, decoding=False)

        Forward pass of the CNN autoencoder.

        Parameters
        ----------
        - X : any
            - Input signal of the autoencoder.
        - encoding : bool
            - If `True` the forward pass will return the encoder output.
        - decoding : bool
            - If `True` the forward pass expects an input X of size equal to 
                the encoder output and returns the encoder output.

        Returns
        -------
        - encoding = `True` (encoder) : Any
            - Encoder output (size of the middle layer of the autoencoder)
        - decoding = `True` (decoder) : Any
            - Decoder output (size of the input layer of the autoencoder)
        - encoding = decoding = `False` : Any
            - Autoencoder output (size of the input layer of the autoencoder)
        """
        if encoding:
            return self.encoder(X)
        elif decoding:
            return self.decoder(X)
        else:
            encode = self.encoder(X)
            decode = self.decoder(encode)
        return decode


class autoencoder:

    def __init__(self) -> None:
        pass


    def save_object(self, save_object, file_name) -> None:
        """
        # save_object

        save_object(save_object, file_name)

        Save an object to a .bin file using the Pickle library.
        Is used in this context for dictionaries.

        Parameters
        ----------
        - save_object : dict
                - Dictionary to save.
        - file_name : str
                - Name of the file to save.

        Returns
        -------
        None
        """
        try:
            with open(f"{file_name}.bin", 'wb') as f:
                pickle.dump(save_object, f)
        except Exception as ex:
            warnings.warn("Error during saving process : ", ex)


    def open_object(self, file_name):
        """
        # open_object

        open_object(file_name)

        Open a file using the Pickle library. 
        Is used in this context for files containing dictionaries.

        Parameters
        ----------
        - file_name : str
                - Name of the file to open.

        Returns
        -------
        None
        """
        try:
            with open(file_name, 'rb') as f:
                dictionary = pickle.load(f)
        except Exception as ex:
            warnings.warn("Error when loading file : ", ex)

        return dictionary
    

    def custom_Kfold(self, config):
        """
        # custom_Kfold

        custom_Kfold(config)

        Create a K-fold cross validation set up by creating a list of training, validation and test files.
        The test files are meant to be used to test the model after the training and validation steps.
        The test files stay the same accross the K folds. 
        The training and validation are defined to create K folds and each fold can be separated into batches.

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary

        Returns
        -------
        - train_files : list
            - List of numpy arrays containing the name of the training files used in the batches and folds.
              The list is organised so each element of the list is associated to a fold and the every sub array 
              is a batch.
        - validation_files : list
            - List of numpy arrays containing the name of the validation files used in the batches and folds.
              The list is organised so each element of the list is associated to a fold and the every sub array 
              is a batch.
        - test_files : list
            - List containing all the test files.
        - config : dict
            - Updated dictionary. Define `input_dimension` of the autoencoder
        """
        folder = f"Datasets/{config['files']['dataset']}"
        files = listdir(folder)

        fold = KFold(n_splits=config['train']['k-fold'],shuffle=True,random_state=42)
        train_validation_files, test_files = train_test_split(files,train_size=config['train']['train_size'],shuffle=True)
        splits = fold.split(train_validation_files)

        train_files = []
        validation_files = []

        for train_index, validation_index in splits:
            train_files.append(np.take(train_validation_files, train_index))
            validation_files.append(np.take(train_validation_files, validation_index))

        train_batch_number = validation_batch_number = config['train']['batch_number']
        
        batch_max = len(train_files[0])
        if train_batch_number >= batch_max:
            warnings.warn(f"Batch number too high, was set to {batch_max} (maximum)")
            train_batch_number = batch_max
        
        batch_max = len(validation_files[0])
        if validation_batch_number >= batch_max:
            validation_batch_number = batch_max
        
        train_files = [np.array_split(train_fold, train_batch_number) for train_fold in train_files]
        validation_files = [np.array_split(validation_fold, validation_batch_number) for validation_fold in validation_files]

        return train_files, validation_files, test_files, config
    

    def custom_dataloader(self, config, files):
        """
        # custom_dataloader

        custom_dataloader(config, files)

        Creates a pytorch tensor containing all the batch samples of a specific fold.

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary
        - files : list
                - List of files used in the batch. 
                  All the samples inside the files will be stored as a pytorch tensor.
                  To reduce the memory requirements increase the batch number in the configuration dictionary.

        Returns
        -------
        - samples : torch.tensor
            - Three dimensional tensor containing the batch samples.
              Tensor of shape (N,0,S), where N is the number of sample and S is the size of each sample.    
        """
        skip = config['network']["skip_elements"]
        folder = f"Datasets/{config['files']['dataset']}"
        size = config['files']['input_dimension']

        TES = np.concatenate([np.fromfile(f"{folder}/{file_name}",dtype=np.uint16).reshape((-1,size)) for file_name in listdir(folder)])
        
        if skip > 1: TES = [i[1::skip] for i in TES]

        return torch.tensor(np.array(TES, dtype="float32")).view(-1,1,int(size / skip))
        
    
    def build_optimizer(self, network, config):
        """
        # build_optimizer

        build_optimizer(network, config)

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary
        - network : Pytorch sequential : 
                - Autoencoder neural network that is trained to reproduce its input signal.

        Returns
        -------
        - optimizer : Pytorch optimizer
                - Optimizer used to train the autoencoder.
        """

        optimizer_dict = {
            "SGD"  : optim.SGD,
            "adam" : optim.Adam
        }

        try:
            optimizer = optimizer_dict[config['train']['optimizer']]
        except Exception as ex:
            optimizer = optimizer_dict["adam"]
            #warn(ex)
            warnings.warn("No optimizer was defined int the configuration dict (was set to adam)")

        return optimizer(network.parameters(), lr=config['train']['learning_rate'])


    def build_criterion(self, config):
        """
        # build_optimizer

        build_optimizer(network, config)

        Parameters
        ----------
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary
        - network : Pytorch sequential : 
                - Autoencoder neural network that is trained to reproduce its input signal.

        Returns
        -------
        - optimizer : Pytorch optimizer
                - Optimizer used to train the autoencoder.
        """
        criterion_dict = {
            "CrossEntropy"       : nn.CrossEntropyLoss(),
            "L1Loss"             : nn.L1Loss(),
            "MSELoss"            : nn.MSELoss(),
            "NLLLoss"            : nn.NLLLoss(),
            "HingeEmbeddingLoss" : nn.HingeEmbeddingLoss(),
            "MarginRankingLoss"  : nn.MarginRankingLoss(),
            "TripletMarginLoss"  : nn.TripletMarginLoss(),
            "KLDivLoss"          : nn.KLDivLoss()
        }

        try:
            criterion = criterion_dict[config['train']['criterion']]
        except Exception as ex:
            criterion = criterion_dict["MSELoss"]
            #warn(ex)
            warnings.warn("No criterion was defined int the configuration dict (was set MSELoss)")
            
        return criterion
    

    def train_epoch(self, config, network, X_train, optimizer, criterion):
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
        for data in X_train:
            # Use cuda if available
            data = data.double().to(config['internal']['device'])
            # Zero gradient
            optimizer.zero_grad()
            # Forward
            loss = criterion(network(data), data)
            # Backward
            loss.backward()
            optimizer.step()
            # Loss
            cumu_loss += loss.item()

        return cumu_loss, len(X_train)
    

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
        lenght = len(X)
        results = {'encode1'     : [],
                'encode2'     : [],
                'decode'      : [],
                'input'       : []
                }

        cumu_loss = 0
        network.eval()
        with torch.no_grad():
            for index, data in enumerate(X):
                # Use cuda if available
                data = data.double().to(config['internal']['device'])

                if store:
                    
                    encode = network(data, encoding=True)
                    decode = network(encode, decoding =True)

                    save_encode = torch.clone(encode).numpy()
                    results["encode1"].append(save_encode[0,0])
                    results["encode2"].append(save_encode[0,1])

                    if index > 3:
                        save_input = torch.clone(data).numpy()
                        results["input"].append(save_input[0])

                        save_encode = torch.clone(decode).numpy()
                        results["decode"].append(save_encode[0])
                else:
                    decode = network(data)
                    
                loss = criterion(decode, data)
                cumu_loss += loss.item()

        if store:
            return cumu_loss / lenght, results
        
        return cumu_loss, lenght
    

    def save_all(self, log_path, network, results, loss, config):
        torch.save(network.state_dict() , f"{log_path}/model.pt")

        self.save_object(results , f"{log_path}/results")
        self.save_object(loss , f"{log_path}/loss")
        self.save_object(config, f"{log_path}/log")


    def run(self, build_autoencoder, config):
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
        # log path and folder creation to store results
        if config['sweep']['sweep_name'] is not None:
            log_path = f"{config['files']['path_save']}/{config['sweep']['sweep_name']}/sweep {config['internal']['sweep_index']}"
        else:
            config['internal'] = {}
            folder_name = datetime.now().strftime(r"%Y-%m-%d-%H-%M")
            log_path = f"{config['files']['path_save']}/run-{folder_name}"

        config['internal']['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_files, validation_files, test_files, config = self.custom_Kfold(config)

        for fold_index in tqdm(range(config['train']['k-fold']), desc="Fold", total=config['train']['k-fold']):

            # Initialization of loss and result arrays
            loss = {'train_loss'        : [], 
                    'validation_loss'   : [],
                    'test_loss'         : [],
                    'average_test_loss' : []
                    }
        
            network = build_autoencoder(config).double().to(config['internal']['device'])
            optimizer = self.build_optimizer(network, config)
            criterion = self.build_criterion(config)

            for epoch in tqdm(range(config['train']['epochs']), desc="Epoch"):
                
                train_loss = validation_loss = train_number = validation_number = 0

                for batch_files in train_files[fold_index]:   
                
                    X_train = self.custom_dataloader(config, batch_files)
                    
                    train_loss_, train_number_ = self.train_epoch(config, network, X_train, optimizer, criterion)
                    train_loss += train_loss_
                    train_number += train_number_

                for batch_files in validation_files[fold_index]:

                    X_validation = self.custom_dataloader(config, batch_files)

                    validation_loss_, validation_number_ = self.validation_test(config, network, X_validation, criterion)
                    validation_loss += validation_loss_
                    validation_number += validation_number_

                loss['train_loss'].append(train_loss/train_number)
                loss['validation_loss'].append(validation_loss/validation_number)
            
            X_test = self.custom_dataloader(config, test_files)
            test_loss , results = self.validation_test(config, network, X_test, criterion, store=True)
            loss['test_loss'].append(test_loss)
    
            fold_path = f"{log_path}/fold {fold_index}"
            makedirs(fold_path)
            self.save_all(fold_path, network, results, loss, config)



    def sweep(self, name, test_number, build_autoencoder, config):
        """
        # Sweep 

        sweep(name, test_number, config)

        Train a series of neural networks (change the activation functions and layer size) 
        using a search process (random search or grid search). 
        The activation function are defined to keep the autoencoder structure.
        Any parameter can be modified in the sweep to experiment.

        TODO
        - Find criteria for dimensionality reduction and add it to the sweep

        name (str) : 
        test_number (int) : Number of model to create and train.
        config (dict) : Configuration parameters to create the autoencoders.

        Parameters
        ----------
        - name : str
                - Name of the folder created to store the runs.
        - test_number : int
                - Number of model to create and train.
        - build_autoencoder : class
                - Pytorch neural network class with a `__init__` definition and `forward` process.
        - config : dict
                - Dictionary containing the experiment parameters. 
                  See the `autoencoder` class for more details on the config dictionary.
        
        Returns
        -------
        - None
        """
        config['internal'] = {}
        config['sweep']['sweep_name'] = name
        config_run = config.copy()
        config_run['internal']['sweep_index'] = 0
        layer_number = config_run['run']['layer_number']

        
        # Creat random test
        if config['sweep']['search_type'] == 'random_search':
            pbar = tqdm(total=test_number)
            while config_run['internal']["sweep_index"] <= test_number:

                if 'layer_possibility' in config['sweep']['search_param']:
                        layer_number = choice(config['sweep']['layer_possibility']['number'])
                        layer_list = choices(config['sweep']['layer_possibility']['size'], k = int(layer_number / 2))
                        layer_list.append(config['network']['output_dimension'])
                        
                        config_run['run']['layer_list'] = layer_list + list(reversed(layer_list))[1:]
                        config_run['run']['layer_number'] = layer_number

                for parameter in config['sweep']['search_param']:

                    if parameter == 'activation_possibilty':
                        activation_list = choices(config['sweep'][parameter], k = int(layer_number / 2))
                        config_run['run']['activation_list'] = activation_list + list(reversed(activation_list))[1:]
                    elif parameter == 'layer_possibility':
                        pass
                    else:
                        config_run['sweep'][parameter] = choice(config[parameter])
                    
                # Train the newly created config
                self.run(build_autoencoder, config_run)
                # Only save the results if the results meet threshold criteria
                config_run['internal']['sweep_index'] += 1
                pbar.update(1)
            pbar.close()
        """
        elif config['sweep']['search_type'] == 'grid_search':
            for parameter in tqdm(config['sweep']['search_param']):
                for value in config[parameter]:
                    config_run[parameter] = value
                    # Train the newly created config
                    threshold = train(config_run)
                    config_run['sweep_index'] +=1
        """

    def load_results(self, file_name):
        """
        # load_results

        load_results(file_name)

        Load a run or a sweep to plot the dimensionality reduction output, the losses, and two 
        inputs compared to the autoencoder output.

        Parameters
        ----------
        - file_name : str
                - Name of the file or path to file inside the `Autoencoder Log` folder.
        
        Returns
        -------
        - None
        """
        warnings.filterwarnings("ignore")
        path = f"Autoencoder Log/{file_name}"
        

        for index, fold in enumerate(listdir(path)):

            fig, axs = plt.subplots(2,2,figsize=(15,8))

            results = self.open_object(f"{path}/{fold}/results.bin")
            
            axs[0,0].scatter(results['encode1'], results['encode2'],label=f"fold {index}",s=5,alpha=0.01)
            axs[0,0].set_xlabel("feature 1")
            axs[0,0].set_ylabel("feature 2")
            leg = axs[0,0].legend()
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

            axs[1,0].plot(results['decode'][0],label=f"Autoencoder output {index}")
            axs[1,0].plot(results['input'][0],label=f"Autoencoder input {index}")
            axs[1,0].set_ylabel("Normalized voltage")
            axs[1,0].set_xlabel("element")
            axs[1,0].legend()

            axs[1,1].plot(results['decode'][1],label=f"Autoencoder output {index}")
            axs[1,1].plot(results['input'][1],label=f"Autoencoder input {index}")
            axs[1,1].set_ylabel("Normalized voltage")
            axs[1,1].set_xlabel("element")
            axs[1,1].legend()

            loss = self.open_object(f"{path}/{fold}/loss.bin")

            axs[0,1].plot(loss['train_loss'],label=f"Train {index}")
            axs[0,1].plot(loss['validation_loss'],label=f"Validation {index}")
            axs[0,1].hlines(loss['test_loss'], 0, len(loss['validation_loss'])-1, linestyles='dashed',label = f"Test {index}")
            axs[0,1].legend()
            axs[0,1].set_ylabel("loss")
            axs[0,1].set_xlabel("epoch")

        config_file = self.open_object(f"{path}/{fold}/log.bin")
        print("Activation list : ", config_file['run']['activation_list'])
        print("Layer list : ", config_file['run']['layer_list'])


