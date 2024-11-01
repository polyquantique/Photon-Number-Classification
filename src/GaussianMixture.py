from typing import Union
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import rcParams
from sklearn.mixture import GaussianMixture
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from scipy.stats import multivariate_normal, poisson, uniform, gennorm
from scipy.special import gamma
from scipy.signal import butter, filtfilt
from scipy.integrate import simpson, trapezoid

class gaussian_mixture():
    """
    Gaussian mixture clustering for TES samples inside a 1D or 2D latent space.
    The procedure offers ordered clustering based on the area of the average
    signal for every cluster. This is justified since the area is proportional
    to the energy and therefore photon number.

    Parameters
    ----------
    X_low : ndarray
        Low dimensional representation of the samples in X_high.
        Shape : (Number of samples , Dimension of latent space)
    X_high : ndarray
        High dimensional representation of the samples (initial signals).
        Shape : (Number of samples , Dimension of the initial samples)
    number_cluster : int
        Initial guess for the number of clusters in the data.
    label_shift : int
        Label shift for the predictions. Example for `label_shift` = 1 every
        label will be shifted by 1.
        This allows labels to be associated to photon numbers for cases when
        the method ignores 0.
    critical_frequency : float
        Critical frequency of Butterworth filter used to order clusters based on area.
        `Wn` in `scipy.signal.butter` https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    cluster_iter : int
        Number of initializations to perform for the Gaussian Mixture model.
    means_init : Union[None, np.array]
        Mean of clusters used in first iteration of Gaussian Mixture.
    tol : float
        Tolerance used in the Gaussian mixture model.
    seed : int
        Random seed used inGaussian Mixture model.
    info_sweep : int
        Values for the different number of clusters evaluated. Example
        for `info_sweep` = 2 the cluster numbers from
        [`number_cluster` - `info_sweep` , `number_cluster` + `info_sweep`)
        are evaluated.
    plot_sweep : bool
        Boolean to plot the number of cluster analysis.
            - Akaike information criterion (aic)
            - Bayesian information criterion (bic)
            - Silhouette score (silhouette)
    metric_sweep : str
        Metric used to sweep the number of clusters ('silhouette', 'aic', 'bic').
    width_plot : int
        Width of the plots in the class.
    height_plot : int
        Height of the plots in the class.
    dpi : int
        Resolution of the plots in the class.

    Returns
    -------
    None
    """

    def __init__(self, X_low : np.array,
                 X_high : np.array,
                 number_cluster : int = 10,
                 label_shift : int = 0,
                 critical_frequency : float = 0.01,
                 cluster_iter : int = 10,
                 means_init : Union[None,np.array] = None,
                 tol : float = 1e-3,
                 seed : any = 42,
                 info_sweep : int = 0,
                 plot_sweep : bool = False,
                 metric_sweep : str = 'silhouette',
                 width_plot : int = 6,
                 height_plot : int = 3,
                 dpi : int = 100,
                 style : str = r'src/custom.mplstyle',
                 latex : bool = False) -> None:

        # Style
        self.style_name = style#"seaborn-v0_8"  # str
        self.cmap = 'Blues'  # str
        self.text_color = 'k'  # str
        self.levels = 30  # int
        self.width_plot = width_plot  # Union[int, float]
        self.height_plot = height_plot  # Union[int, float]
        self.gridsize_density = 500  # int
        self.min_log = 1e-2  # Bottom ylim when plotting (avoid plotting small values in log)
        self.dpi = dpi  # int
        self.seed = seed  # int
        if latex: rcParams['text.usetex'] = True
                
        # Dataset
        self.X_low = np.array(X_low)
        self.dim = self.X_low.shape[1]  # int
        self.X_high = X_high  # np.array (Number of samples , Dimension of the initial samples)
        self.eps = 1e-10  # To reduce risk of divided by 0 in discrete integral
        self.tol = tol
        self.number_cluster = number_cluster  # int
        self.min = self.X_low.min(axis=0)  # float
        self.max = self.X_low.max(axis=0)  # float
        self.X_low = (X_low - self.min) / (self.max - self.min)  # np.array (Number of samples , Dimension low dim space)

        assert 0 < self.dim <= 2, \
            'The number of dimensions of X_low must be 1 or 2'
        assert self.X_low.shape[0] == self.X_high.shape[0], \
            'The number of sample in X_low and X_high must be equal'
        assert self.number_cluster < self.X_low.shape[0], \
            'The number of sample must be higher than the number of clusters'
        assert self.number_cluster - info_sweep > 1, \
            'The number of cluster must be higher than 1'

        # Clustering iteration
        self.critical_frequency = critical_frequency  # float
        self.metric_sweep = None  # Shape (2*info_sweep)
        self.cluster_iter = cluster_iter  # int

        # Initial clustering iteration (find optimal number of cluster)
        assert metric_sweep in ['silhouette', 'aic', 'bic'], \
            "Metric must be one of the following options: 'silhouette', 'aic', or 'bic'."

        if (means_init is None) and (info_sweep > 0):
            self.clustering_iter(info_sweep = info_sweep,
                             cluster_iter = self.cluster_iter,
                             plot_sweep = plot_sweep,
                             metric_sweep = metric_sweep)
        else:
            pass

        # Initial clustering parameters
        self.cluster_means = None  # shape (self.number_cluster, self.dim)
        self.cluster_covariances = None  # Shape (self.number_cluster, self.dim, self.dim)
        self.cluster_weights = None  # Shape (self.number_cluster)
        self.predict_ = None  # Sklearn object

        # Ordered clustering with initialized means
        self.clustering_order(cluster_iter = self.cluster_iter,
                              means_init = means_init)

        # Labels
        self.label_shift = label_shift
        self.labels = self.predict_(self.X_low) + self.label_shift
        self.unique_labels = np.arange(self.number_cluster) + label_shift

        # Confidence metrics
        self.confidence_1D = None  # Shape (self.number_cluster)
        self.confidence_2D = None  # Shape (self.number_cluster)
        self.confidence_poisson = None

        # Trustworthiness
        self.trustworthiness_eucl = np.zeros(3)

        # g2
        self.g2 = None

    def clustering_iter(self, info_sweep : int = 10,
                            cluster_iter : int = 10,
                            plot_sweep : int = False,
                            metric_sweep : str = 'silhouette') -> None:
        """
        Iteration over interval
        [`number_cluster` - `info_sweep` , `number_cluster` + `info_sweep`)
        to find an optimal number of clusters for Gaussian Mixture.

        Parameters
        ----------
        info_sweep : int
            Values for the different number of clusters evaluated. Example for
            info_sweep = 2 the cluster numbers from
            [`number_cluster` - `info_sweep` , `number_cluster` + `info_sweep`)
            are evaluated.
        cluster_iter : int
            Number of initializations to perform for the Gaussian Mixture model.
        plot_sweep : bool
            Boolean to plot the number of cluster analysis.
                - Akaike information criterion (aic)
                - Bayesian information criterion (bic)
                - Silhouette score (silhouette)
        metric_sweep : str
            Metric used to find the number of clusters.
            Possible options are Akaike information criterion ('aic'), Bayesian
            information criterion ('bic'), or Silhouette score ('silhouette')

        Returns
        -------
        None
        """

        size_sweep = 2 * info_sweep
        self.metric_sweep = np.zeros(size_sweep)
        number_array = np.arange(self.number_cluster - info_sweep,
                                 self.number_cluster + info_sweep)

        for index, n_cluster in tqdm(enumerate(number_array),
                                     desc = 'Searching for number of clusters',
                                     total=size_sweep):

            fit_ = GaussianMixture(n_components = n_cluster,
                                   tol = self.tol,
                                   max_iter = 200,
                                   n_init = -(cluster_iter//-2),
                                   init_params = 'k-means++',
                                   random_state = self.seed).fit(self.X_low)

            if metric_sweep == 'silhouette':
                self.metric_sweep[index] = silhouette_score(self.X_low, fit_.predict(self.X_low))
                selected_index = np.argmax(self.metric_sweep)
                self.number_cluster = number_array[selected_index]
            else:
                if metric_sweep == 'aic':
                    self.metric_sweep[index] = fit_.aic(self.X_low)
                elif metric_sweep == 'bic':
                    self.metric_sweep[index] = fit_.bic(self.X_low)

                second_deriv = [self.metric_sweep[i+1] + self.metric_sweep[i-1] - 2 * self.metric_sweep[i] for i in range(1,size_sweep-1)]
                selected_index = np.argmax(second_deriv) + 1
                self.number_cluster = number_array[selected_index]

        if plot_sweep:

            with plt.style.context(self.style_name):
                plt.figure(figsize = (self.width_plot,self.height_plot), dpi=self.dpi)
                plt.plot(number_array, self.metric_sweep)
                plt.scatter(self.number_cluster,
                            self.metric_sweep[selected_index],
                            marker='x',
                            s = 50,
                            color = 'r',
                            alpha = 0.5,
                            label='Selected number')
                plt.ylabel(f'{metric_sweep}')
                plt.xlabel('Number of cluster')
                plt.tight_layout()
                plt.legend()
                plt.show()

    def clustering_order(self, cluster_iter : int = 10,
                         means_init : Union[None, np.array] = None) -> None:
        """
        Definition of the Gaussian Mixture for the optimal number of clusters.
        The labels are also initialized to follow photon numbers (label shift
        and area of the initial signals).

        Parameters
        ----------
        cluster_iter : int
            Number of initializations to perform for the Gaussian Mixture model.
        means_init : Union[None, np.array]
            If is numpy array, initializes means of every cluster, otherwise
            uses 'k-means++' algorithm to initialize means.

        Returns
        -------
        None

        """

        fit_ = GaussianMixture(n_components = self.number_cluster,
                                tol = self.tol,
                                max_iter = 200,
                                n_init = cluster_iter,
                                init_params = 'k-means++',
                                means_init = means_init,
                                random_state = self.seed).fit(self.X_low)

        # Get prediction
        predict_init = fit_.predict(self.X_low)
        # Get area and labels and applies filter on traces
        b, a = butter(5, self.critical_frequency, 'low')
        X_high_filter = filtfilt(b, a, self.X_high)
        X_Area = simpson(X_high_filter, axis=1).flatten()
        # Get average area of the clusters
        labels = np.arange(self.number_cluster)
        # Order labels based on photon number
        index_sorted = np.argsort(np.array([np.mean(X_Area[predict_init == label_]) for label_ in labels]))
        # Update Gaussian Mixture parameters
        fit_.means_ = fit_.means_[index_sorted]
        fit_.covariances_ = fit_.covariances_[index_sorted]
        fit_.weights_ = fit_.weights_[index_sorted]

        self.cluster_means = fit_.means_
        self.cluster_covariances = fit_.covariances_
        self.cluster_weights = fit_.weights_
        self.predict_ = fit_.predict 

    def remove_clusters(self, number_clusters : list):
        """
        Delete cluster based on index.
        """

        self.cluster_means = np.delete(self.cluster_means, number_clusters, axis=0).reshape(-1,self.dim)
        self.unique_labels = np.delete(self.unique_labels, number_clusters, axis=0)

        self.number_cluster -= len(number_clusters)

        self.clustering_order(cluster_iter = self.cluster_iter,
                              means_init = self.cluster_means)

    def add_clusters(self, number_clusters : list, positions : list):
        """
        Insert cluster between two existing clusters.
        """

        self.cluster_means = np.insert(self.cluster_means, number_clusters, positions, axis=0)
        self.unique_labels = np.insert(self.unique_labels, number_clusters, positions, axis=0)

        self.number_cluster += len(number_clusters)

        self.clustering_order(cluster_iter = self.cluster_iter,
                              means_init = self.cluster_means)

    def predict(self, X_low : np.array) -> np.array:
        """
        Predict the label of samples in `X_low` based on initial latent space separation.

        Parameters
        ----------
        X_low : ndarray
            Array containing all the samples in their low-dimensional representation.

        Returns
        -------
        labels : ndarray
            Array containing the labels associated to the low-dimensional samples in `X_low`.

        """
        X_low = (X_low - self.min) / (self.max - self.min)
        return self.predict_(X_low) + self.label_shift

    def plot_density(self, bw_adjust : float = 1.,
                     plot_scale : str = 'log',
                     plot_gaussians : bool = True,
                     plot_uniform : bool = False,
                     plot_gen_gauss : bool = False,
                     xlim = (-0.05,1.05),
                     ylim = (-0.05,1.05),
                     save_path : Union[None, str] = None,
                     text = None,
                     cluster_number : bool = True) -> None:
        """
        Plot the kernel density estimation of the latent space.

        Parameters
        ----------
        bw_adjust : float
            Bandwidth used in kernel density estimation.
        plot_scale : str
            Scale used in plot (follow matplotlib norm)
        plot_gaussians : bool
            Plot the Gaussians associated with every cluster.
        plot_uniform : bool
            Plot the Uniform associated with every cluster.
        plot_gen_gauss : bool
            Plot the Generalized Gaussians associated with every cluster.
        sides : bool
            Bollean to plot 1D side views of the 2D space.

        Returns
        -------
        None

        """

        def plot_functions_1d(x, func):
            for index, weight in zip(range(self.number_cluster), self.cluster_weights):
                plt.plot(x,
                         weight * func[index],
                         linewidth = 1,
                         label = f"{index + self.label_shift}")

        def plot_functions_2d(x, func):
            for index, weight in zip(range(self.number_cluster), self.cluster_weights):
                plt.contour(x,
                            x,
                            weight * func[index],
                            alpha = 0.5,
                            levels = [func[index].max()/2],
                            linewidths = 0.5,
                            colors = 'k')

        x = np.linspace(0, 1, self.gridsize_density)

        with plt.style.context(self.style_name):

            if self.dim == 1:
                plt.figure(figsize=(self.width_plot, self.height_plot), dpi = self.dpi)

                if plot_gaussians: plot_functions_1d(x, self.multi_gaussian_1d(x))
                if plot_uniform: plot_functions_1d(x, self.multi_uniform_1d(x))
                if plot_gen_gauss: plot_functions_1d(x, self.multi_gen_gaussian_1d(x))

                sns.kdeplot(x = self.X_low.flatten(),
                            cmap = self.cmap,
                            fill = True,
                            bw_adjust = bw_adjust,
                            gridsize = self.gridsize_density)

                y_gauss = self.multi_gaussian_1d(self.cluster_means)

                if cluster_number:
                    for index, mean in enumerate(self.cluster_means[:-1]):
                        plt.text(mean, 
                                self.cluster_weights[index]*y_gauss[index,index]+1, 
                                index, 
                                color = self.text_color, 
                                size=10)
    
                    last = self.cluster_means.shape[0]-1
                    plt.text(self.cluster_means[-1], 
                            self.cluster_weights[index]*y_gauss[last,last]+1, 
                            f'{last}+', 
                            color = self.text_color, 
                            size=10)

                if plot_scale == 'log':
                    ylim = (self.min_log, 1e2)
                else:
                    ylim = (self.min_log, None)

                plt.yscale(plot_scale)
                plt.ylabel(r'Density')
                plt.xlabel(r'$s_1$')
                plt.xlim(xlim)
                plt.ylim(ylim)

            elif self.dim == 2:

                plt.figure(figsize=(self.width_plot, self.height_plot), dpi=self.dpi)

                if plot_gaussians: plot_functions_2d(x, self.multi_gaussian_2d(x, x))

                sns.kdeplot(x = self.X_low[:,0],
                           y = self.X_low[:,1],
                           cmap = self.cmap,
                           cbar=True,
                           fill = True,
                           norm = LogNorm(),
                           bw_adjust = bw_adjust,
                           thresh = self.min_log,
                           levels = self.levels,
                           cbar_kws={"ticks":[0,1,10,20,30,40,50,100,1000],
                                     "label":"Density"},
                           gridsize = self.gridsize_density)

                if cluster_number:
                    for index, mean in enumerate(self.cluster_means[:-1]):
                        # if index % 2 != 0:
                        plt.text(mean[0], mean[1]+0.03, index, color = self.text_color, size=10)
    
                    last = self.cluster_means.shape[0]-1
                    mean = self.cluster_means[-1]
                    plt.text(mean[0], mean[1]+0.03, f'{last}+', color = self.text_color, size=10)

                plt.xlabel(r'$s_1$')
                plt.ylabel(r'$s_2$')
                plt.xlim(xlim)
                plt.ylim(ylim)

            if text is not None:
                file_name = f'{save_path}/{text}.pdf'
                if self.dim == 1:
                    xt, yt = 0.9 * xlim[1], 0.7 * ylim[1]
                elif self.dim == 2:
                    xt, yt = 0.9 * xlim[1], ylim[0] + 0.1

                plt.text(xt, yt, text, ha='right', va='top',)
            else:
                file_name = f'{save_path}/density.pdf'

            if save_path is not None:
                plt.savefig(file_name, bbox_inches='tight', pad_inches = 0)
                plt.show()
            else:
                plt.show()

    def multi_gaussian_1d(self, x : np.array, axis : any = None) -> np.array:
        """

        Create a numpy array of shape (`number_cluster` , `x.size`) containing a discret 1D gaussian for every cluster, 
        considering the Gaussian mixture parameters.

        Parameters
        ----------
        x : ndarray
            Interval of the latent space where the confidence integration is mainly contained.
        axis : int
            Axis to consider in the case of 1D confidence evaluation for a 2D latent space. 

        Returns
        -------
        multi_gaussian : ndarray
            Array of shape (`number_cluster` , `x.size`) containing a discret 1D gaussian for every cluster

        """

        multi_gaussian = np.zeros((self.number_cluster, x.size))

        for index, (mean, covariance) in enumerate(zip(self.cluster_means, self.cluster_covariances)):
            if axis is not None:
                mean = mean[axis]
                covariance = covariance[axis,axis]

            multi_gaussian[index,:] = multivariate_normal(mean = mean, cov = covariance).pdf(x)

        return multi_gaussian

    def multi_uniform_1d(self, x : np.array, axis : any = None) -> np.array:
        """

        Create a numpy array of shape (`number_cluster` , `x.size`) containing a discret 1D uniform for every cluster, 
        considering the Gaussian mixture parameters.

        Parameters
        ----------
        x : ndarray
            Interval of the latent space where the confidence integration is mainly contained.
        axis : int
            Axis to consider in the case of 1D confidence evaluation for a 2D latent space. 

        Returns
        -------
        multi_uniform : ndarray
            Array of shape (`number_cluster` , `x.size`) containing a discret 1D gaussian for every cluster

        """
        multi_uniform = np.zeros((self.number_cluster, x.size))

        for index, (mean, covariance) in enumerate(zip(self.cluster_means, self.cluster_covariances)):
            if axis is not None:
                mean = mean[axis]
                covariance = covariance[axis,axis]

            multi_uniform[index] = uniform(loc = mean - np.sqrt(3 * covariance), scale = 2 * np.sqrt(3 * covariance)).pdf(x)

        return multi_uniform

    def multi_gen_gaussian_1d(self, x : np.array, axis : any = None, beta : float = 5) -> np.array:
        """
        Create a numpy array of shape (`number_cluster` , `x.size`) containing a discret 1D generalized Gaussian for 
        every cluster, considering the Gaussian mixture parameters.

        Parameters
        ----------
        x : ndarray
            Interval of the latent space where the confidence integration is mainly contained.
        axis : int
            Axis to consider in the case of 1D confidence evaluation for a 2D latent space. 

        Returns
        -------
        multi_gen_gauss : ndarray
            Array of shape (`number_cluster` , `x.size`) containing a discret 1D gaussian for every cluster

        """

        multi_gen_gauss = np.zeros((self.number_cluster, x.size))

        for index, (mean, covariance) in enumerate(zip(self.cluster_means, self.cluster_covariances)):
            if axis is not None:
                mean = mean[axis]
                covariance = covariance[axis,axis]

            alpha = np.sqrt(covariance * gamma(1/beta) / gamma(3/beta))
            multi_gen_gauss[index,:] = gennorm(beta = beta,loc = mean, scale = alpha).pdf(x)

        return multi_gen_gauss

    def multi_gaussian_2d(self, x : np.array, 
                                y : np.array) -> np.array:
        """

        Create a numpy array of shape (`number_cluster` , `x.size`, `y.size`) containing a discrete 2D Gaussian for every cluster, 
        considering the Gaussian mixture parameters.

        Parameters
        ----------
        x : ndarray
            Interval in the first dimension of the latent space where the confidence integration is mainly contained.
        y : ndarray
            Interval in the second dimension of the latent space where the confidence integration is mainly contained.

        Returns
        -------
        multi_gaussian : ndarray
            Array of shape (`number_cluster` , `x.size`, `y.size`) containing a discret 2D Gaussian for every cluster

        """
        multi_gaussian = np.zeros((self.number_cluster, x.size, y.size))
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))

        for index, (mean, covariance) in enumerate(zip(self.cluster_means, self.cluster_covariances)):
            multi_gaussian[index,:,:] = multivariate_normal(mean = mean, cov = covariance).pdf(pos)

        return multi_gaussian

    def trapezoid_2d(self, x : np.array, 
                           y : np.array, 
                           Z : np.array) -> np.array:
        """

        Extention of the scipy trapezoid integration. 

        Parameters
        ----------
        x : ndarray
            Interval in the first dimension of the latent space where the confidence integration is mainly contained.
        y : ndarray
            Interval in the second dimension of the latent space where the confidence integration is mainly contained.
        Z : ndarray
            Values of a function considering the grid associated to `x` and `y`.

        Returns
        -------
        integral : ndarray
            Numerical integral of Z.

        """
        return trapezoid(trapezoid(Z, x), y)

    def plot_confidence_1d(self, average_poisson : float = None,
                                 expected_prob : Union[None, np.array] = None,
                                 axis : Union[None, int] = None,
                                 n_points : int = 1000,
                                 size_zone : float = 1000.,
                                 plot_int : bool = False,
                                 function : bool = 'gauss') -> None:
        """

        1D Confidence metric following :

        P. C. Humphreys et al., 'Tomography of photon-number resolving continuous-output detectors', 
        New J. Phys., vol. 17, no. 10, p. 103044, Oct. 2015, doi: 10.1088/1367-2630/17/10/103044.

        Parameters
        ----------
        average_poisson : float
            Average photon number considering an expected Poisson distribution (will be included in the 
            definition of the confidence).
        expected_prob : Union[None, np.array]
            Expected photon number distribution used to compute the Confidence.
        axis : Union[None, int]
            Axis to consider in the case of 1D confidence evaluation for a 2D latent space.
        n_points : int
            Number of points in the latent space discretization.
        size_zone : float
            Size of the latent space used to approximate the integral.
            The value multiplies the variance for every cluster.
        plot_int : bool
            Plot the discrete function in the latent space for every cluster.
            This can be used to make sure the integrated function in contained in the evaluated interval.
        function : str
            Funciton used to compute the Confidence can be :
            'gauss' (Gaussian), 'uni' (Uniform), 'gauss_gen' (Generalized Gaussian).
        Returns
        -------
        None

        """
        self.confidence_1D = np.zeros(self.number_cluster)

        for index, (mean, covariance) in enumerate(zip(self.cluster_means, self.cluster_covariances)):

            if self.dim > 1:
                x = np.linspace(mean[axis] - size_zone*covariance[axis,axis],
                                mean[axis] + size_zone*covariance[axis,axis],
                                n_points).flatten()
            else:
                x = np.linspace(mean - size_zone*covariance,
                                mean + size_zone*covariance,
                                n_points).flatten()

            if function == 'gauss':
                p_sn = self.multi_gaussian_1d(x, axis = axis)
            elif function == 'uni':
                p_sn = self.multi_uniform_1d(x, axis = axis)
            elif function == 'gen_gauss':
                p_sn = self.multi_gen_gaussian_1d(x, axis = axis)

            if average_poisson is not None:
                p_n = poisson(mu = average_poisson).pmf(k = self.unique_labels) 
                p_s = np.sum(p_sn * p_n, axis = 0)
                conf_integral = p_sn[index]**2 * p_n / (p_s + self.eps)
            elif expected_prob is not None:
                p_n = expected_prob[:self.number_cluster].reshape(-1,1)
                p_s = np.sum(p_sn * p_n, axis = 0)
                conf_integral = p_sn[index]**2 * p_n[index] / (p_s + self.eps)
            else:
                p_s = np.sum(p_sn, axis = 0)
                conf_integral = p_sn[index]**2 / (p_s + self.eps)

            if plot_int:
                with plt.style.context(self.style_name):
                    plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
                    plt.plot(x, conf_integral)
                    plt.show()

            self.confidence_1D[index] = trapezoid(x = x, y = conf_integral)

        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,3), dpi=self.dpi)
            plt.plot(self.unique_labels[:-1], self.confidence_1D[:-1])
            plt.title(f"1D confidence over axis {axis}")
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()

    def plot_confidence_2d(self, average_poisson : float = None,
                                 expected_prob : Union[None, np.array] = None,
                                 n_points : int = 1000,
                                 size_zone : float = 1000,
                                 plot_int : bool = False)-> None:
        """

        2D confidence metric following :

        P. C. Humphreys et al., ‘Tomography of photon-number resolving continuous-output detectors’, 
        New J. Phys., vol. 17, no. 10, p. 103044, Oct. 2015, doi: 10.1088/1367-2630/17/10/103044.

        Parameters
        ----------
        average_poisson : float
            Average photon number considering an expected Poisson distribution (will be included in the 
            definition of the confidence).
        expected_prob : Union[None, np.array]
            Expected photon number distribution used to compute the Confidence.
        n_points : int
            Number of points in the latent space discretization.
        size_zone : float
            Size of the latent space used to approximate the integral. 
            The value multiplies the variance for every cluster.
        plot_int : bool
            Plot the discrete function in the latent space for every cluster.
            This can be used to make sure the integrated function in contained in the evaluated interval.

        Returns
        -------
        None

        """
        self.confidence_2D = np.zeros(self.number_cluster)

        for index, (mean, covariance) in enumerate(zip(self.cluster_means, self.cluster_covariances)):

            x = np.linspace(mean[0] - size_zone*covariance[0,0],
                            mean[0] + size_zone*covariance[0,0],
                            n_points)
            y = np.linspace(mean[1] - size_zone*covariance[1,1],
                            mean[1] + size_zone*covariance[1,1],
                            n_points)

            p_sn = self.multi_gaussian_2d(x, y)

            if average_poisson is not None:
                p_n = poisson(mu = average_poisson).pmf(k = self.unique_labels)
                p_s = np.sum(p_sn * p_n, axis = 0)
                conf_integral = p_sn[index]**2 * p_n / (p_s + self.eps)
            elif expected_prob is not None:
                p_n = expected_prob[self.unique_labels].reshape(-1,1,1)
                p_s = np.sum(p_sn * p_n, axis = 0)
                conf_integral = p_sn[index]**2 * p_n[index] / (p_s + self.eps)
            else:
                p_s = np.sum(p_sn, axis = 0)
                conf_integral = p_sn[index] / (p_s + self.eps) * p_sn[index]

            if plot_int:
                with plt.style.context(self.style_name):
                    plt.figure(figsize=(self.width_plot, self.height_plot), dpi = self.dpi)
                    plt.imshow(conf_integral, cmap = self.cmap)
                    plt.colorbar()
                    plt.show()

            self.confidence_2D[index] = self.trapezoid_2d(x, y, conf_integral)


        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,3), dpi=self.dpi)
            plt.plot(self.unique_labels[:-1], self.confidence_2D[:-1])
            plt.title(f"2D confidence")
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()

    
    def plot_trustworthiness(self):
        """

        Trustworthiness using Euclidian and Cosine distances.

        Trustworthiness following :

        J. Venna et S. Kaski, “Neighborhood preservation in nonlinear projection methods: An experimental study,”
        dans Artificial Neural Networks — ICANN 2001, ser. Lecture Notes in Computer Science, G. Dorffner,
        H. Bischof et K. Hornik,  ́edit. Springer, p. 485–491.

        The scikit learn implementation is used :

        ‘sklearn.manifold.trustworthiness’, scikit-learn. Accessed: Mar. 18, 2024. [Online]. 
        Available: https://scikit-learn/stable/modules/generated/sklearn.manifold.trustworthiness.html

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        label = self.unique_labels[-1]

        X_low = self.X_low[self.labels <= label]
        X_high = self.X_high[self.labels <= label]

        self.trustworthiness_eucl[-3] = trustworthiness(X_high, X_low, metric="euclidean", n_neighbors=15)
        self.trustworthiness_eucl[-2] = trustworthiness(X_high, X_low, metric="euclidean", n_neighbors=500)
        self.trustworthiness_eucl[-1] = trustworthiness(X_high, X_low, metric="euclidean", n_neighbors=10_000)

        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
            plt.plot(['15', '500', '10 000'], self.trustworthiness_eucl, label='Euclidean')
            plt.xlabel("Photon number")
            plt.ylabel("Trustworthiness")
            plt.legend()
            plt.show()        


    def plot_traces(self):
        """
        Plot the traces `X_high` and label them by following the order of the low-dimensional representation
        given in the initialization process.  

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        color = iter(cm.Pastel1(np.linspace(0, 1, self.number_cluster)))
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
            for label in self.unique_labels:
                cluster = self.X_high[self.labels == label]
                c = next(color)
                if len(cluster) > 1000: cluster = cluster[:1000]
                plt.plot(cluster.T, alpha=0.05, c=c)

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
        #plt.savefig('traces.svg',format="svg", transparent=True)


    def plot_traces_average(self):
        """
        Plot the traces average and labels them by following the order of the low-dimensional representation
        given in the initialization process.  

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
            for label in self.unique_labels:
                cluster = self.X_high[self.labels == label]
                plt.plot(np.mean(cluster, axis=0), label=label)

            plt.legend(ncol=3)
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('average_traces.svg',format="svg", transparent=True)

    def plot_g2(self, db):

        self.unique_db = []
        self.g2 = []

        for db_ in np.unique(db):
            condition = db == db_
            label_ = self.labels[condition]
            self.g2.append(1 + (np.var(label_, axis=0) - np.mean(label_, axis=0)) / (np.mean(label_, axis=0)**2 + 1e-10))
            self.unique_db.append(db_)

        with plt.style.context(self.style_name):
            plt.scatter(self.unique_db, self.g2, s=10)
            plt.ylabel('g2')
            plt.xlabel('Attenuation [dB]')
            plt.show()