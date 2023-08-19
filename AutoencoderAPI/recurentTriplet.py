from os import makedirs, listdir
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm

import torch
import torch.onnx

from .setup.networks.autoencoder import build_autoencoder 
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .utils.files import save_all

from fast_pytorch_kmeans import KMeans as pytorch_kmeans
from pytorch_adapt.layers import SilhouetteScore as pytorch_silhouette_score

from torch import nn

plt.style.use("seaborn-pastel")

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

class recurentTriplet():

    def __init__(self):
        self.device = None
        self.alpha = 0

    def train_triplet_epoch(self, network, X_train, optimizer, criterion, cluster_label):
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
        _ = None
        network.train()
        for index, input_ in enumerate(X_train):#tqdm(enumerate(X_train),total=X_train.size(0) , desc='Train Triplet'):
            
            # Zero gradient
            optimizer.zero_grad()
            # Forward
            output_ = network(input_)
            # Criterion
            current_label = cluster_label[index]
            negative_index = torch.where(cluster_label != current_label)[0]
            rand_index = torch.randint(negative_index.size(0), (1,))
            negative = X_train[negative_index[rand_index]]
            loss = criterion.forward(output_, input_, _, negative.view(1,-1), self.alpha)
            # Backward
            loss.backward()
            optimizer.step()
            # Loss
            cumu_loss += loss.item()

        return cumu_loss / len(X_train)


    def train_MSE_epoch(self, network, X_train, optimizer, criterion):
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
        _ = None
        network.train()
        for input_ in X_train:#tqdm(X_train, total=X_train.size(0), desc='Train MSE'):
            
            # Zero gradient
            optimizer.zero_grad()
            # Forward
            output_ = network(input_)
            # Criterion
            loss = criterion.forward(output_, input_, _, _, _)
            # Backward
            loss.backward()
            optimizer.step()
            # Loss
            cumu_loss += loss.item()

        return cumu_loss / len(X_train)
    

    def validation_test(self, network, X, criterion, cluster_label, store=False):
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
        _ = None

        if store:
            results = {'encode' : [],
                       'input'  :  [],
                       'decode' : [],
                       'MSE'    : []
                       }

        network.eval()
        with torch.no_grad():
            for index, data in enumerate(X):

                if store:
                    encode = network(data, encoding=True)
                    decode = network(encode, decoding =True)

                    save_encode = torch.clone(encode).numpy()
                    results['encode'].append(save_encode[0,0])
                    mse = nn.MSELoss()                           
                    results['MSE'].append(mse(decode , data))

                    if index < 2:
                        results['input'].append(data.clone().view(-1).numpy())
                        results['decode'].append(decode.clone().view(-1).numpy())

                else:
                    decode = network(data)
                
                current_label = cluster_label[index]
                negative_index = torch.where(cluster_label != current_label)[0]
                rand_index = torch.randint(negative_index.size(0), (1,))
                negative = X[negative_index[rand_index]]

                loss = criterion.forward(decode, data, _, negative.view(1,-1), self.alpha)
                cumu_loss += loss.item()

        if store:
            return cumu_loss / len(X), results
        
        return cumu_loss / len(X)
    

    def setup(self, config):

        try:
            if config['sweep']:
                folder_name = ""
        except:
            folder_name = "/run-" + datetime.now().strftime(r"%Y-%m-%d-%H-%M")

        log_path = f"{config['files']['path_save']}{folder_name}/fold 0"

        if 'alpha' in config['train'].keys():
            self.alpha = config['train']['alpha']

        # Define device and runs on Cuda if is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define dataset
        skip = config['train']["skip_elements"]
        folder = f"{config['files']['dataset']}"
        size = config['files']['input_dimension']
        files = listdir(folder)

        X = np.concatenate([np.fromfile(f"{folder}/{file_name}", dtype=np.float16).reshape((-1,size)) for file_name in files])
        if skip > 1: X = X[:, 1::skip]
        

        data = torch.from_numpy(X).view(-1, 1, int(size / skip)).float().to(self.device)

        return config, data, log_path


    def split_dataset(self, data):

        len_ = data.size(0)
        index = torch.randperm(len_)
        
        index_split = torch.split(index, len_//4)
        train_index = torch.cat((index_split[0], index_split[1]))
        validation_index = index_split[2]
        test_index = index_split[3]

        return train_index, validation_index, test_index


    def update_cluster(self, network, X_train, n_cluster, output=False):

        score = 0
        optimal_cluster = 0
        skip = 5
        labels = torch.tensor([])
        network.eval()
        with torch.no_grad():
            encode = network(X_train, encoding=True).reshape(-1,1)

            for cluster_number in n_cluster:#tqdm(n_cluster , desc=f"Clusters {n_cluster}") :

                kmeans = pytorch_kmeans(n_clusters=cluster_number, mode='euclidean', verbose=False)
                temp_labels = kmeans.fit_predict(encode[::skip])
                sil = pytorch_silhouette_score()
                if len(np.unique(temp_labels)) < 2:
                    temp_score = 0
                else:
                    temp_score = sil(encode[::skip], temp_labels)

                if temp_score > score:
                    optimal_cluster = cluster_number
                    score = temp_score

            kmeans = pytorch_kmeans(n_clusters=optimal_cluster, mode='euclidean', verbose=False)
            labels = kmeans.fit_predict(encode)

            if optimal_cluster <= 7: optimal_cluster = 7
            n_cluster = range(optimal_cluster-5 , optimal_cluster+5)

            if output:
                if len(np.unique(temp_labels)) < 2:
                    score = 0
                else:
                    score = sil(encode, labels)

                return score
        
        return labels, n_cluster
                
                

        

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
        # Initialization of loss and result arrays
        loss = {'train_loss'        : [], 
                'validation_loss'   : [],
                'test_loss'         : []
                #'Silhouette'        : 0
                }

        config, data, log_path = self.setup(config)
        learning_rate = config['train']['learning_rate']
        train_index, validation_index, test_index = self.split_dataset(data)
        network = build_autoencoder(config).float().to(self.device)
        
        
        config['train']['criterion'] = 'MSELoss'
        config['train']['learning_rate'] = 1e-4
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        for epoch in tqdm(range(4) , desc="Epoch MSE"):
            train_loss = self.train_MSE_epoch(network, data[train_index], optimizer, criterion)
        
        
        config['train']['criterion'] = 'TripletMSE'
        config['train']['learning_rate'] = learning_rate
        
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        n_cluster = range(4,10)
        for epoch in tqdm(range(config['train']['epochs']) , desc="Epoch Triplet"):
            
            train_labels, n_cluster = self.update_cluster(network, data[train_index], n_cluster)
            train_loss = self.train_triplet_epoch(network, data[train_index], optimizer, criterion, train_labels)

            validation_labels, _ = self.update_cluster(network, data[validation_index], n_cluster)
            validation_loss = self.validation_test(network, data[validation_index], criterion, validation_labels)

            loss['train_loss'].append(train_loss) # Triplet
            loss['validation_loss'].append(validation_loss) # Triplet

        test_labels, _ = self.update_cluster(network, data[test_index], n_cluster)
        test_loss, results = self.validation_test(network, data[test_index], criterion, test_labels, store=True)
        loss['test_loss'].append(test_loss)

        #n_cluster = [6]
        #loss['Silhouette'] = self.update_cluster(network, data[test_index], n_cluster, output=True)
        
        makedirs(log_path)
        save_all(log_path, network, results, loss, config)

