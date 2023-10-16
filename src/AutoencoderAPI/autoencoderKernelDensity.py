import torch

from .utils.files import open_object
from .utils.kernelDensity import kernel_density
from .setup.networks.autoencoder import build_autoencoder


class autoencoder_kernelDensity():
    """
    Load an autoencoder model for dimensionality reduction.
    The feature space is separated for labelling using kernel density estimation.

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
        print(config_load)
        if config_load['train']['skip_elements'] <= 1:
            self.size = config_load['files']['input_dimension']
        else:
            self.size = int(config_load['files']['input_dimension'] / config_load['train']['skip_elements'])

        self.network = network
        self.config_load = config_load
        self.fit_ = None
        
    
    def fit(self, X, 
                plot_density = False, 
                plot_cluster = False, 
                plot_traces = False,
                plot_traces_average = False, 
                bw_cst = (-5, -2, 20), 
                flip = False,
                filter_input = False,
                filter_threshold = 0.0005,
                cluster_xlim = None,
                traces_xlim = None,
                skip = 1):
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
        plot_density : bool
            If `True` plot the kernel density estimation over the feature space.
        plot_cluster : bool 
            If `True` plot the histogram of these samples in the feature space with their labels.
        plot_traces : bool
            If `True` plot the labelled input traces.
        bw : tuple or numpy.array
            If bw is a tuple, it represents the parameters inside np.logspace(*bw).
            Otherwise, an array can be used, this represents an array containing all 
            the possible bandwidth used in the kernel density estimation.
        flip : bool
            flips the feature space to inverse le labels ordering.
        cluster_xlim : tuple
            The limits of the horizontal axis when plotting the clusters. Follows the structure (min,max).
        traces_xlim : tuple
            The limits of the horizontal axis when plotting the traces. Follows the structure (min,max).
        skip : int
            Skip a number of elements to define the feature space separation following the [::skip] structure.

        Returns
        -------
        None
        """
        self.network.eval()
        with torch.no_grad():
            X_pytorch = torch.from_numpy(X).view(-1, 1, self.size).float()
            

            if filter_input:
                X_low_dim, X_reconst = self.network(X_pytorch, both=True)
            
                X_reconst = X_reconst.detach().numpy().reshape(-1, self.size)
                X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)

                MSE = ((X - X_reconst)**2).mean(axis=1)
                condition = MSE < filter_threshold

                X_low_dim = X_low_dim[condition]
                X = X[condition]
            else:
                X_low_dim = self.network(X_pytorch, encoding=True)
                X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)

        kd = kernel_density(X_low_dim, bw_cst, skip, flip)
            
        if plot_density:
            kd.plot_density()
        if plot_cluster:
            kd.plot_cluster(cluster_xlim)
        if plot_traces:
            kd.plot_traces(X, traces_xlim)
        if plot_traces_average:
            kd.plot_traces_average(X, traces_xlim)

        self.fit_ = kd.fit

        

    def get_label(self, X):
        """
        Transform samples into a low-dimensional space using the predefined autoencoder.
        With this new representation and based on the feature space separation, it 
        assigns a label to every sample.

        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.

        Returns
        -------
        labels : numpy.array
            Array containing the labels of the input samples.

        """
        X_pytorch = torch.from_numpy(X).view(-1, 1, self.size).float()
        self.network.eval()
        with torch.no_grad():
            X_low_dim = self.network(X_pytorch, encoding=True)
            X_low_dim = X_low_dim.detach().numpy().reshape(-1)

        return self.fit_(X_low_dim)

    

    def get_label_filter(self, X, threshold=0.0005):
        """
        Transform samples into a low-dimensional space using the predefined autoencoder.
        With this new representation and based on the feature space separation, it 
        assigns a label to every sample.

        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.

        Returns
        -------
        labels : numpy.array
            Array containing the labels of the input samples.

        """
        X_pytorch = torch.from_numpy(X).view(-1, 1, self.size).float()
        self.network.eval()
        with torch.no_grad():
            X_low_dim, X_reconst = self.network(X_pytorch, both=True)
            
            X_reconst = X_reconst.detach().numpy().reshape(-1, self.size)
            X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)

            MSE = ((X - X_reconst)**2).mean(axis=1)
            labels = self.fit_(X_low_dim)

        return labels[MSE < threshold]