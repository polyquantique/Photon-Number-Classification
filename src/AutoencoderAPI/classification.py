from os import makedirs, listdir
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm
from warnings import warn
import torch

from .setup.networks.autoencoder import build_autoencoder
from .setup.optimizer import build_optimizer
from .setup.criterion import build_criterion
from .setup.train.generic import train as train_MSE
from .setup.train.triplet import train as train_Triplet
from .setup.validation.tripletValidation import validation
from .utils.files import save_all
from .utils.clustering.GaussianMixture import gaussian_mixture

#torch.use_deterministic_algorithms(True)
#torch.manual_seed(42)

class classification_training():

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
        if config['files']['dB'] != None:
            config['files']['dB'] = [str(i) for i in config['files']['dB']]
            files = [i for i in files if i[67:71] in config['files']['dB']]
        
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

        return data, log_path, config



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



    def update_cluster(self, network, X, config, plot=False):
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
            X_low_dim = X_low_dim.cpu().numpy().reshape(-1, 2)
            X_high_dim = X.cpu().numpy().reshape(-1, config['internal']['size_network'])

            min_ = np.min(X_low_dim)
            max_ = np.max(X_low_dim)

            X_low_dim = X_low_dim - min_ /(max_ - min_)
            gm = gaussian_mixture(X_low_dim,
                                  X_high_dim,
                                  number_cluster = config['train']['number_cluster'],
                                  cluster_iter = 20,
                                  info_sweep = 0,
                                  plot_sweep = False,
                                  label_shift = 0)
                      
            #gm = density_gaussianMixture(X_low_dim, config['network']['bw_cst'], 
            #                             number_cluster=config['train']['number_cluster'], 
            #                             skip=100)
            if plot:
                gm.plot_density(bw_adjust = 0.1)
                gm.plot_cluster()
            

            if len(np.unique(gm.labels)) == 1:
                warn('One class detected : bw_cst might be too large')

            return torch.tensor(gm.labels), torch.tensor(gm.cluster_means)

                

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

        data, log_path, config = self.setup(config)
        train_index, validation_index, test_index = self.split_dataset(data)
        network = build_autoencoder(config).float().to(self.device)
        
        
        config['train']['criterion'] = 'MSELoss'
        config['train']['learning_rate'] = config['train']['learning_rate_MSE']
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        for epoch in tqdm(range(config['train']['epochs_MSE']) , desc="Epoch MSE"):
            train_loss = train_MSE(network, data[train_index], optimizer, criterion)
        

        config['train']['criterion'] = 'TripletMSE'
        config['train']['learning_rate'] = config['train']['learning_rate_triplet']
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        for epoch in tqdm(range(config['train']['epochs_triplet']) , desc="Epoch Triplet"):
            train_labels, train_means = self.update_cluster(network, 
                                                            data[train_index], 
                                                            config,
                                                            plot=False)
            
            train_loss = train_Triplet(config, 
                                       network, 
                                       data[train_index], 
                                       optimizer, 
                                       criterion, 
                                       train_labels, 
                                       train_means, 
                                       self)

            validation_labels, validation_means = self.update_cluster(network, 
                                                                      data[validation_index], 
                                                                      config)
            validation_loss = validation(config['train']['alpha'], 
                                         network, 
                                         data[validation_index], 
                                         criterion, 
                                         validation_means,
                                         self,
                                         validation_labels)

            loss['train_loss'].append(train_loss) # Triplet
            loss['validation_loss'].append(validation_loss) # Triplet

        test_labels, test_means = self.update_cluster(network, 
                                                      data[test_index], 
                                                      config)
        
        test_loss, results = validation(config['train']['alpha'], 
                                        network, 
                                        data[test_index], 
                                        criterion, 
                                        test_means, 
                                        self,
                                        test_labels,
                                        store=True)
        
        loss['test_loss'].append(test_loss)

        makedirs(log_path)
        save_all(log_path, network, results, loss, config)