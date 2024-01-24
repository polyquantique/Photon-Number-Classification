import torch
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from .utils.files import open_object
#from .utils.clustering.silhouetteGaussianMixture import silhouette_gaussianMixture
from .utils.clustering.densityGaussianMixture import density_gaussianMixture
from .setup.networks.autoencoder import build_autoencoder


class autoencoder_gaussianMixture():
    """
    Load an autoencoder model for dimensionality reduction.
    The feature space is separated for labelling using Gaussian mixture.

    Parameters
    ----------
    model_path : str
        Path to the `run` folder containing the autoencoder.

    Returns
    -------
    None

    """
    def __init__(self, model_path):
        
        config_load = open_object(f"{model_path}/log.bin")
        network = build_autoencoder(config_load)
        network.load_state_dict(torch.load(f"{model_path}/model.pt"))
        
        self.size_network = config_load['internal']['size_network']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)
        self.config_load = config_load
        self.mins = None
        self.predict = None
        self.labels = None
        self.cluster_means = None
        self.cluster_covariance = None
        
    
    def fit(self, X,  
                plot_density = False,
                plot_cluster = False, 
                bins_plot = 5000,
                plot_traces = False,
                plot_traces_average = False,
                plot_cross_talk = False,
                bw_cst = [0.008], 
                density_kernel = 'gaussian',
                skip = 1,
                flip = False,
                filter_input = False,
                filter_threshold = 0.0005,
                size_plot = 10):
        """
        Use the loaded autoencoder to transform the input data into a 
        low-dimensional representation. The labels are assigned to the samples by
        using kernel density estimation to separate the low-dimensional space.
        
        After the feature space is separated, the `get_label` function can 
        be used directly to predict new samples.

        If desired a variety of parameters can be plotted :

        - Kernel density estimation over the feature space.
        - Histogram of these samples in the feature space with their labels.
        - Labelled input traces. 
        
        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.
        plot_cluster : bool 
            If `True` plot the histogram of these samples in the feature space with their labels.
        bins_plot : int 
            Number of bins for the cluster plot.
        plot_traces : bool
            If `True` plot the labelled input traces.
        plot_traces : bool
            If `True` plot the labelled input traces average for every cluster.
        bw_cst : tuple or numpy.array
            If bw is a tuple, it represents the parameters inside np.logspace(*bw).
            Otherwise, an array can be used, this represents an array containing all 
            the possible bandwidth used in the kernel density estimation.
        skip : int
            Number of skip for the kernel density estimation step (folowing the X[skip::] notation)
        flip : bool
            flips the feature space to inverse le labels ordering.
        filter_input : bool
            If `True` filter the traces by evaluating the reconstruction error of the autoencoder following the `filter_threshold`.
        filter_threshold : float
            Value used to reject traces with too low reconstruction errors.

        Returns
        -------
        None
        """
        self.network.eval()
        with torch.no_grad():
            X_pytorch = torch.from_numpy(X).view(-1, 1, self.size_network).float().to(self.device)
            

            if filter_input:
                X_low_dim, X_reconst = self.network(X_pytorch, both=True)
            
                X_reconst = X_reconst.cpu().numpy().reshape(-1, self.size_network)
                X_low_dim = X_low_dim.cpu().numpy().reshape(-1, 1)

                MSE = ((X - X_reconst)**2).mean(axis=1)
                condition = MSE < filter_threshold

                X_low_dim = X_low_dim[condition]
                X = X[condition]
            else:
                X_low_dim = self.network(X_pytorch, encoding=True)
                X_low_dim = X_low_dim.cpu().numpy().reshape(-1, 1)

        dgm = density_gaussianMixture(X_low_dim, 
                                      bw = bw_cst, 
                                      density_kernel = density_kernel,
                                      bins_plot = bins_plot,
                                      flip=flip,
                                      skip=skip,
                                      size_plot=size_plot)
        if plot_density:
            dgm.plot_density()
        if plot_cluster:
            dgm.plot_cluster() 
        if plot_traces:
            dgm.plot_traces(X)
        if plot_traces_average:
            dgm.plot_traces_average(X)
        if plot_cross_talk:
            dgm.plot_cross_talk()

        self.mins = dgm.mins
        self.predict = dgm.predict
        self.labels = dgm.labels
        self.cluster_means = dgm.cluster_means
        self.cluster_covariance = dgm.cluster_covariance



    def get_clusters(self, X, 
                     filter_input = False, 
                     filter_threshold = 0.0005):
        """
        Sort the traces and their low dimensional representation based on initial fit.

        .. warning::
            The function requires initial `fit` to work.
        
        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.
        filter_input : bool
            If `True` filter the traces by evaluating the reconstruction error of the autoencoder following the `filter_threshold`.
        filter_threshold : float
            Value used to reject traces with too low reconstruction errors.
  

        Returns
        -------
        clusters_traces : list
            List of numpy arrays containing the traces for every cluster.
        clusters_low_dim : list
            List of numpy arrays containg the low dimensional representation of the traces for every cluster.
        """
        with torch.no_grad():
            X_pytorch = torch.from_numpy(X).view(-1, 1, self.size_network).float().to(self.device)
            
            if filter_input:
                X_low_dim, X_reconst = self.network(X_pytorch, both=True)
        
                MSE = torch.mean((X_pytorch - X_reconst)**2, 2)
                condition = MSE < filter_threshold

                X_low_dim = X_low_dim[condition]
                X_pytorch = X_pytorch[condition]
            else:
                X_low_dim = self.network(X_pytorch, encoding=True)

        labels = self.predict(X_low_dim).view(-1)
        clusters_traces = []
        clusters_low_dim = []

        for number in torch.unique(labels):
            condition = labels == number
            clusters_traces.append(X_pytorch[condition].view(-1).cpu())
            clusters_low_dim.append(X_low_dim[condition].view(-1).cpu())

        return clusters_low_dim #clusters_traces, clusters_low_dim
        

    def get_label(self, X):
        """
        Transform samples into a low-dimensional space using the predefined autoencoder.
        With this new representation and based on the feature space separation, it 
        assigns a label to every sample.

        .. warning::
            The function requires initial `fit` to work.

        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.

        Returns
        -------
        labels : numpy.array
            Array containing the labels of the input samples.

        """
        X_pytorch = torch.from_numpy(X).view(-1, 1, self.size_network).float().to(self.device)
        self.network.eval()
        with torch.no_grad():
            X_low_dim = self.network(X_pytorch, encoding=True)
        return self.predict(X_low_dim).flatten().cpu()

    

    def get_label_filter(self, X, threshold=0.0005):
        """
        Transform samples into a low-dimensional space using the predefined autoencoder.
        With this new representation and based on the feature space separation, it 
        assigns a label to every sample.

        .. warning::
            The function requires initial `fit` to work.

        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.
        threshold : float
            Value used to reject traces with too low reconstruction errors.

        Returns
        -------
        labels : numpy.array
            Array containing the labels of the input samples.

        """
        X_pytorch = torch.from_numpy(X).view(-1, 1, self.size_network).float().to(self.device)
        self.network.eval()
        with torch.no_grad():
            X_low_dim, X_reconst = self.network(X_pytorch, both=True)
            X_reconst = X_reconst.cpu().numpy().reshape(-1, self.size_network)

            MSE = ((X - X_reconst)**2).mean(axis=1)
            labels = self.predict(X_low_dim)

        return labels[MSE < threshold].cpu()
    
    