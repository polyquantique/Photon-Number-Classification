import numpy as np
from os import listdir, makedirs
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime
from random import choices, choice, sample

import torch
import torch.onnx
from torch import nn
import torch.optim as optim

from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tqdm.notebook import tqdm
import pickle
import warnings

from AutoencoderAPI import dataset, autoencoder 
from AutoencoderAPI.utils.files import open_object, save_object
from AutoencoderAPI.loss.pytorch_kmeans_silhouette_loss import pytorch_kmeans_silhouette_loss
from AutoencoderAPI.loss.sklearn_kernelDensity_loss import sklearn_kernelDensity_loss
from AutoencoderAPI.loss.sklearn_kmeans_silhouette_loss import sklearn_kmeans_silhouette_loss

plt.style.use("seaborn-pastel")

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)



class function:

    def __init__(self) -> None:
        pass


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
        folder = f"{config['files']['dataset']}"
        files = listdir(folder)

        fold = KFold(n_splits=config['train']['k-fold'],shuffle=True,random_state=42)
        train_validation_files, test_files = train_test_split(files,train_size=0.8,shuffle=True)
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
        folder = f"{config['files']['dataset']}"
        size = config['files']['input_dimension']

        TES = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])

        if skip > 1: TES = TES[:, 1::skip]

        return torch.from_numpy(TES).view(-1, 1, int(size / skip)).float()
        
    
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
            "Adam" : optim.Adam
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
            "CrossEntropy"       : (nn.CrossEntropyLoss() , 0),
            "L1Loss"             : (nn.L1Loss() , 0),
            "MSELoss"            : (nn.MSELoss() , 0),
            "NLLLoss"            : (nn.NLLLoss() , 0),
            "HingeEmbeddingLoss" : (nn.HingeEmbeddingLoss() , 0),
            "MarginRankingLoss"  : (nn.MarginRankingLoss() , 0),
            "TripletMarginLoss"  : (nn.TripletMarginLoss() , 0),
            "KLDivLoss"          : (nn.KLDivLoss() , 0),
            "pytorch_kmeans_silhouette_loss"  : (pytorch_kmeans_silhouette_loss() , 1),
            "sklearn_kernelDensity_loss"      : (sklearn_kernelDensity_loss() , 1),
            "sklearn_kmeans_silhouette_loss"  : (sklearn_kmeans_silhouette_loss() , 1)
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
        list_ = range(len(X_train))
        for _, data in tqdm(enumerate(X_train) , total=len(X_train)): #enumerate(X_train):
            # Use cuda if available
            data = data.float().to(config['internal']['device'])
            # Zero gradient
            optimizer.zero_grad()
            # Forward
            output = network(data)
            # Criterion
            if criterion[1]:
                if _//40 == 0:
                    X_sub = X_train[sample(list_, 10_000)]
                crit = criterion[0]
                loss = crit.forward(output, data, network, X_sub)
            else:
                loss = criterion[0](output, data)

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
        cumu_loss = 0

        if store:
            results = {'encode' : [],
                       'input'  : [],
                       'decode' : []
            }

        network.eval()
        with torch.no_grad():
            for index, data in enumerate(X):
                # Use cuda if available
                data = data.float().to(config['internal']['device'])

                if store:
                    encode = network(data, encoding=True)
                    decode = network(encode, decoding =True)

                    save_encode = torch.clone(encode).numpy()
                    results['encode'].append(save_encode[0])

                    if index < 2:
                        results['input'].append(data.clone().numpy()[0])
                        results['decode'].append(decode.clone().numpy()[0])

                else:
                    decode = network(data)
                    
                loss = criterion(decode, data)
                cumu_loss += loss.item()

        if store:
            return cumu_loss / len(X), results
        
        return cumu_loss, len(X)
    

    def save_all(self, log_path, network, results, loss, config):
        torch.save(network.state_dict() , f"{log_path}/model.pt")

        save_object(results , f"{log_path}/results")
        save_object(loss , f"{log_path}/loss")
        save_object(config, f"{log_path}/log")


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
        # log path and folder creation to store results
        if config['sweep']['sweep_name'] is not None:
            log_path = f"{config['files']['path_save']}/{config['sweep']['sweep_name']}/sweep {str(config['internal']['sweep_index']).rjust(config['internal']['number_size'], '0')}"
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
        
            network = autoencoder.build_autoencoder(config).float().to(config['internal']['device'])
            optimizer = self.build_optimizer(network, config)
            criterion = self.build_criterion(config)

            for epoch in tqdm(range(config['train']['epochs']), desc="Epoch"):
                
                train_loss = validation_loss = train_number = validation_number = 0

                for batch_files in tqdm(train_files[fold_index], desc="Train batch"):   
                
                    X_train = self.custom_dataloader(config, batch_files)
                    
                    train_loss_, train_number_ = self.train_epoch(config, network, X_train, optimizer, criterion)
                    train_loss += train_loss_
                    train_number += train_number_

                for batch_files in tqdm(validation_files[fold_index], desc="Validation batch"):

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



    def sweep(self, name, build_autoencoder, config, test_number=1):
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

                if 'layer_size_possibility' in config['sweep']['search_param']:
                        layer_number = choice(config['run']['layer_number'])
                        layer_list = choices(config['sweep']['layer_size_possibility'], k = int(layer_number / 2))
                        layer_list.append(config['network']['output_dimension'])
                        
                        config_run['run']['layer_list'] = layer_list + list(reversed(layer_list))[1:]

                for parameter in config['sweep']['search_param']:

                    if parameter == 'activation_possibilty':
                        activation_list = choices(config['sweep'][parameter], k = int(layer_number / 2))
                        config_run['run']['activation_list'] = activation_list + list(reversed(activation_list))[1:]
                    elif parameter == 'layer_size_possibility':
                        pass
                    else:
                        config_run['sweep'][parameter] = choice(config[parameter])
                    
                # Train the newly created config
                self.run(build_autoencoder, config_run)
                config_run['internal']['sweep_index'] += 1
                pbar.update(1)
            pbar.close()
        
        elif config['sweep']['search_type'] == 'grid_search':
            parameter_dict = {}
            for parameter in config['sweep']['search_param']:
                parameter_dict[parameter[1]] = config[parameter[0]][parameter[1]]

            parameter_list = list(ParameterGrid(parameter_dict))
            test_number = len(parameter_list)
            config_run['internal']['number_size'] = len(str(test_number))

            pbar = tqdm(total=test_number)
            for parameters in parameter_list:
                for value in parameters:
                    print(f"{value} : " , parameters[value])
                    config_run['train'][value] = parameters[value]
                
                # Train the newly created config
                self.run(build_autoencoder, config_run)
                config_run['internal']['sweep_index'] += 1
                pbar.update(1)
            pbar.close()


    def silhouette_kmean(self, X, max_cluster):

        X = np.array(X).reshape(-1,1)
        scores1 = []
        #scores2 = []
        #scores3 = []

        for cluster_number in tqdm(range(2,max_cluster+1) , desc="Clusters") :
            clusters = KMeans(n_clusters=cluster_number, random_state=42, algorithm='auto').fit_predict(X[::10])
            scores1.append(silhouette_score(X[::10], clusters))
            #scores2.append(calinski_harabasz_score(X[::10], clusters))
            #scores3.append(davies_bouldin_score(X[::3], clusters))

        optimal_cluster = np.argmax(scores1) + 1

        km = KMeans(n_clusters=optimal_cluster, random_state=42)
        fit = km.fit(X)
        
        centroids = fit.cluster_centers_
        labels = fit.labels_
        unique_label = np.unique(labels)

        if len(unique_label) > 1:
            optimal_score = silhouette_score(X, labels)
        else:
            optimal_score = 0

        out = []
        for label in unique_label:
            out.append(X[labels == label])

        centroids, out, labels = zip(*sorted(zip(centroids, out, unique_label)))

        return scores1, optimal_cluster, optimal_score, out, np.mean(labels)
        

    def load_run_results(self, file_name):
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

            fig, axs = plt.subplots(2,2,figsize=(15,10))

            results = open_object(f"{path}/{fold}/results.bin")

            scores1, optimal_cluster, optimal_score, out, average_nb = self.silhouette_kmean(results['encode'], 40)
            print(f"Optimal number : {optimal_cluster}")
            
            bins = np.linspace(min(results['encode']), max(results['encode']), 2000).reshape(-1)

            for index1, cluster in enumerate(out):
                axs[0,0].hist(cluster , bins, alpha = 0.5, label=f"{index1}")
            #axs[0,0].hist(results['encode'] , bins)
            axs[0,0].set_xlabel("feature")
            axs[0,0].set_ylabel("counts")
            axs[0,0].legend(ncol=3)
            axs[0,0].title.set_text(f"Average photon number : {average_nb}")
                

            #axs[1,0].plot(range(2, len(scores1)+2), scores1, label="Silhouette")
            #axs[1,0].plot(range(2, len(scores2)+2), scores2, label="Calinski-Harabasz")
            axs[1,0].plot(range(2, len(scores1)+2), scores1, label="Approx Silhouette")

            axs[1,0].hlines(optimal_score, 2, len(scores1)+2, linestyles='dashed', label="Final Silhouette")
            axs[1,0].set_ylabel("Clustering score")
            axs[1,0].set_xlabel("Number of cluster")
            axs[1,0].legend()
            axs[1,0].title.set_text("3")


            axs[1,1].plot(results['decode'][1],label=f"Autoencoder output {index}")
            axs[1,1].plot(results['input'][1],label=f"Autoencoder input {index}")
            axs[1,1].set_ylabel("Normalized voltage")
            axs[1,1].set_xlabel("element")
            axs[1,1].legend()
            axs[1,1].title.set_text("4")

            loss = open_object(f"{path}/{fold}/loss.bin")

            axs[0,1].plot(loss['train_loss'],label=f"Train {index}")
            axs[0,1].plot(loss['validation_loss'],label=f"Validation {index}")
            axs[0,1].hlines(loss['test_loss'], 0, len(loss['validation_loss'])-1, linestyles='dashed',label = f"Test {index}")
            axs[0,1].legend()
            axs[0,1].set_ylabel("loss")
            axs[0,1].set_xlabel("epoch")
            axs[0,1].title.set_text("2")

        config_file = open_object(f"{path}/{fold}/log.bin")
        print("Activation list : ", config_file['run']['activation_list'])
        print("Layer list : ", config_file['run']['layer_list'])



    def load_sweep_results(self, file_name, parameters):
        warnings.filterwarnings("ignore")
        path = f"Autoencoder Log/{file_name}"

        parameter1 = []
        parameter2 = []
        loss_sweep = []
        min_loss = 1
        
        for sweep in sorted(listdir(path)):
            loss_cum = 0
            fold_list = sorted(listdir(f"{path}/{sweep}"))
            fold_len = len(fold_list)

            for index, fold in enumerate(fold_list):
                loss = open_object(f"{path}/{sweep}/{fold}/loss.bin")
                loss_cum += loss['test_loss'][0]

            config_file = open_object(f"{path}/{sweep}/{fold}/log.bin")
            
            parameter1.append(config_file['train'][parameters[0]])
            parameter2.append(config_file['train'][parameters[1]])
            loss_sweep.append(loss_cum / fold_len)

            if loss_sweep[-1] < min_loss:
                min_loss = loss_sweep[-1]
                min_parameter1 = parameter1[-1]
                min_parameter2 = parameter2[-1]

        
        x=np.unique(parameter1)
        y=np.unique(parameter2)
        X,Y = np.meshgrid(x,y)
        print("min : ", min_loss)
        print(f"{parameters[0]} : ", min_parameter1)
        print(f"{parameters[1]} : ", min_parameter2)

        Z= np.rot90(np.array(loss_sweep).reshape(len(y),len(x)))
        plt.xticks(np.arange(len(x)), labels=x)
        plt.yticks(np.arange(len(y)), labels=y)

        #plt.pcolormesh(X,Y,Z)
        plt.imshow(Z, norm=colors.LogNorm(), interpolation="bilinear")
        plt.xlabel(parameters[0])
        plt.ylabel(parameters[1])
        plt.colorbar()
        plt.show()