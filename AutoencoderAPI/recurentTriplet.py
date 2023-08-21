from os import makedirs, listdir
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm

import torch

from .setup.networks.autoencoder import build_autoencoder 
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .setup.train.MSE import train as train_MSE
from .setup.train.Triplet import train as train_Triplet
from .setup.validation.TripletValidation import validation
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

    def setup(self, config):

        try:
            if config['sweep']:
                folder_name = ""
        except:
            folder_name = "/run-" + datetime.now().strftime(r"%Y-%m-%d-%H-%M")

        log_path = f"{config['files']['path_save']}{folder_name}/fold 0"

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
            train_loss = train_MSE(network, data[train_index], optimizer, criterion)
        
        
        config['train']['criterion'] = 'TripletMSE'
        config['train']['learning_rate'] = learning_rate
        
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        n_cluster = range(4,10)
        for epoch in tqdm(range(config['train']['epochs']) , desc="Epoch Triplet"):
            
            train_labels, n_cluster = self.update_cluster(network, data[train_index], n_cluster)
            train_loss = train_Triplet(config, network, data[train_index], optimizer, criterion, train_labels)

            validation_labels, _ = self.update_cluster(network, data[validation_index], n_cluster)
            validation_loss = validation(config['train']['alpha'], network, data[validation_index], criterion, validation_labels)

            loss['train_loss'].append(train_loss) # Triplet
            loss['validation_loss'].append(validation_loss) # Triplet

        test_labels, _ = self.update_cluster(network, data[test_index], n_cluster)
        test_loss, results = validation(config['train']['alpha'], network, data[test_index], criterion, test_labels, store=True)
        loss['test_loss'].append(test_loss)

        makedirs(log_path)
        save_all(log_path, network, results, loss, config)

