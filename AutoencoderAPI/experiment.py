import numpy as np
from os import listdir, makedirs
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime
from  random import choices, choice

import torch
import torch.onnx
from torch import nn
import torch.optim as optim

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from kmeans_pytorch import kmeans

from tqdm.notebook import tqdm
import pickle
import warnings
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset

from AutoencoderAPI import dataset, autoencoder #, transformer

plt.style.use("seaborn-pastel")

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

class function:

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
            "CrossEntropy"       : nn.CrossEntropyLoss(),
            "L1Loss"             : nn.L1Loss(),
            "MSELoss"            : nn.MSELoss(),
            "NLLLoss"            : nn.NLLLoss(),
            "HingeEmbeddingLoss" : nn.HingeEmbeddingLoss(),
            "MarginRankingLoss"  : nn.MarginRankingLoss(),
            "TripletMarginLoss"  : nn.TripletMarginLoss(),
            "KLDivLoss"          : nn.KLDivLoss(),
            "custom"             : nn.MSELoss()
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
        def custom_criterion(output, data, network, X_train):
         
            x = X_train.dataset.__getitem__(range(10_000))
            x_numpy =x.numpy().reshape(-1,250)
            feature = network(x, encoding=True).detach().numpy().reshape(-1,1)

            labels = KMeans(n_clusters=21, random_state=42, n_init='auto').fit_predict(x_numpy)

            davies_loss = davies_bouldin_score(feature, labels)
            mse = nn.MSELoss()

            return davies_loss / 1e4 + mse(output, data)


        cumu_loss = 0
        network.train()
        for _, data in tqdm(enumerate(X_train)):
            # Zero gradient
            optimizer.zero_grad()
            # Forward
            output = network(data)
            #loss = criterion(output, data)
            loss = custom_criterion(output, data, network, X_train)

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
                    
                loss = criterion(decode, data)
                cumu_loss += loss.item()

        if store:
            return cumu_loss / len(X), results
        
        return cumu_loss / len(X)
    

    def save_all(self, log_path, network, results, loss, config):
        torch.save(network.state_dict() , f"{log_path}/model.pt")

        self.save_object(results , f"{log_path}/results")
        self.save_object(loss , f"{log_path}/loss")
        self.save_object(config, f"{log_path}/log")


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
        data = dataset.build_dataset(config)

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

        fold = KFold(n_splits=config['train']['k-fold'], shuffle=True, random_state=42)

        for fold_index, (train_index, test_index) in tqdm(enumerate(fold.split(data)), 
                                                        desc="Fold", 
                                                        total=config['train']['k-fold']):

            # Initialization of loss and result arrays
            loss = {'train_loss'        : [], 
                    'validation_loss'   : [],
                    'test_loss'         : [],
                    }
    
            network = autoencoder.build_autoencoder(config).float().to(config['internal']['device'])
            optimizer = self.build_optimizer(network, config)
            criterion = self.build_criterion(config)

            train_sampler = SubsetRandomSampler(train_index)
            test_sampler = SubsetRandomSampler(test_index)
        
            train_loader = DataLoader(data, batch_size=1, sampler=train_sampler)  #batch_size=config['train']['batch_number']
            test_loader = DataLoader(data, batch_size=1, sampler=test_sampler)

            for epoch in tqdm(range(config['train']['epochs']) , desc="Epoch"):
                
                train_loss = self.train_epoch(config, network, train_loader, optimizer, criterion)
                validation_loss = self.validation_test(config, network, test_loader, criterion)

                loss['train_loss'].append(train_loss)
                loss['validation_loss'].append(validation_loss)

            # Temporary test (test should be different from validation to be unbias)
            test_loss, results = self.validation_test(config, network, test_loader, criterion, store=True)
            loss['test_loss'].append(test_loss)
            
            fold_path = f"{log_path}/fold {fold_index}"
            makedirs(fold_path)

            self.save_all(fold_path, network, results, loss, config)

            #if config['train']['cluster']:

             #   for 




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
                self.run(config_run)
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
                self.run(config_run)
                config_run['internal']['sweep_index'] += 1
                pbar.update(1)
            pbar.close()
        

    def silhouette_kmean(self, X, max_cluster):

        X = np.array(X).reshape(-1,1)
        scores1 = []
        scores2 = []
        scores3 = []

        for cluster_number in tqdm(range(2,max_cluster+1) , desc="Clusters") :
            clusters = KMeans(n_clusters=cluster_number, random_state=42).fit_predict(X[::3])
            #scores1.append(silhouette_score(X[::10], clusters))
            #scores2.append(calinski_harabasz_score(X[::10], clusters))
            scores3.append(davies_bouldin_score(X[::3], clusters))

        optimal_cluster = np.argmin(scores3) + 1

        labels = KMeans(n_clusters=optimal_cluster, random_state=42).fit(X).labels_
        optimal_score = silhouette_score(X, labels)

        out = []
        for label in np.unique(labels):
            out.append(X[labels == label])

        return scores1, scores2, scores3, out, optimal_cluster, optimal_score



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

            fig, axs = plt.subplots(2,2,figsize=(15,8))

            results = self.open_object(f"{path}/{fold}/results.bin")

            scores1, scores2, scores3, X, optimal_cluster, optimal_score = self.silhouette_kmean(results['encode'], 40)
            print(f"Optimal number : {optimal_cluster}")
            
            bins = np.linspace(min(results['encode']), max(results['encode']), 1000)

            for index, cluster in enumerate(X):
                axs[0,0].hist(cluster , bins, alpha = 0.5)
            #axs[0,0].hist(results['encode'] , bins)
            axs[0,0].set_xlabel("feature")
            axs[0,0].set_ylabel("counts")
                

            #axs[1,0].plot(range(2, len(scores1)+2), scores1, label="Silhouette")
            #axs[1,0].plot(range(2, len(scores2)+2), scores2, label="Calinski-Harabasz")
            axs[1,0].plot(range(2, len(scores3)+2), scores3, label="Davies-Bouldin")

            axs[1,0].hlines(optimal_score, 2, len(scores1)+2, label="Final Silhouette")
            axs[1,0].set_ylabel("Silhouette score")
            axs[1,0].set_xlabel("Number of cluster")
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
                loss = self.open_object(f"{path}/{sweep}/{fold}/loss.bin")
                loss_cum += loss['test_loss'][0]

            config_file = self.open_object(f"{path}/{sweep}/{fold}/log.bin")
            
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