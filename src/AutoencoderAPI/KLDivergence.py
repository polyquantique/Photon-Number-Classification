from os import makedirs, listdir
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm
from warnings import warn
import torch

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.manifold._utils import _binary_search_perplexity

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

class KL_divergence():

    def __init__(self):

        self.device = None
        # Define the number of available threads
        self.num_threads = _openmp_effective_n_threads()
        # Numerical precision
        self.MACHINE_EPSILON = np.finfo(np.double).eps

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


    def distances_knn(self, X : np.array, perplexity : int):

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(n_neighbors = min(X.shape[0] - 1, int(3.0 * perplexity + 1)), 
                               metric = "euclidean")
        knn.fit(X)
        distances_nn = knn.kneighbors_graph(mode = "distance")
        del knn # Free memory
        distances_nn.data **= 2

        distances_nn.sort_indices()
        n_samples = distances_nn.shape[0]
        distances_data = distances_nn.data.reshape(n_samples, -1)
        distances_data = distances_data.astype(np.float32, copy=False)

        return distances_data, distances_nn.indices, distances_nn.indptr
    

    def knn(ref, query, k):
        ref_c = torch.stack([ref] * query.shape[-1], dim=0).permute(0, 2, 1).reshape(-1, 2).transpose(0, 1)
        query_c = torch.repeat_interleave(query, repeats=ref.shape[-1], dim=1)
        delta = query_c - ref_c
        distances = torch.sqrt(torch.pow(delta, 2).sum(dim=0))
        distances = distances.view(query.shape[-1], ref.shape[-1])
        sorted_dist, indices = torch.sort(distances, dim=-1)
        return sorted_dist[:, :k], indices[:, :k]


    def _joint_probabilities_nn(
            self,
            distances : np.array, 
            indices : np.array, 
            indptr : np.array
        ) -> np.array:

        # Binary search and conditional probability evaluation (in Cython)
        conditional_P = _binary_search_perplexity(distances, self.perplexity, verbose = 0)

        # Symmetrize the joint probability distribution using sparse operations
        P = csr_matrix(
            (conditional_P.ravel(), indices, indptr),
            shape = (self.n_samples, self.n_samples),
        )
        P = P + P.T

        # Normalize the joint probability distribution
        sum_P = np.maximum(P.sum(), self.MACHINE_EPSILON)
        P /= sum_P

        return P
                

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

        # Compute the distances
        distances_nn, indices, indptr = self.distances_knn(X = data[train_index],
                                                           perplexity = config['train']['perplexity'])
        # compute the joint probability distribution for the input space
        P = self._joint_probabilities_nn(distances = distances_nn, 
                                        indices = indices,
                                        indptr = indptr)

        network = build_autoencoder(config).float().to(self.device)
        criterion = build_criterion(config)
        optimizer = build_optimizer(network, config)

        """
        for epoch in tqdm(range(config['train']['epochs_triplet']) , desc="Epoch Triplet"):
            train_loss = train_Triplet(config, 
                                       network, 
                                       data[train_index], 
                                       optimizer, 
                                       criterion, 
                                       train_labels, 
                                       train_means, 
                                       self)

            validation_loss = validation(config['train']['alpha'], 
                                         network, 
                                         data[validation_index], 
                                         criterion, 
                                         validation_means,
                                         self,
                                         validation_labels)

            loss['train_loss'].append(train_loss) # Triplet
            loss['validation_loss'].append(validation_loss) # Triplet

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
        """