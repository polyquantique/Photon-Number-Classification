import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt 
from matplotlib import cm

from sklearn.mixture import GaussianMixture
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

from scipy.stats import multivariate_normal, poisson
from scipy.integrate import  trapezoid


class gaussian_mixture():
    """

    Gaussian mixture clustering for TES samples inside a 1D or 2D latent space.
    The procedure offers ordered clustering based on the area of the average signal for every cluster.
    This is justified since the area is proportional to the energy and therefore photon number.

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
        Label shift for the predictions. Example for `label_shift` = 1 every label will be shifted by 1.
        This allows labels to be associated to photon numbers for cases when the method ignores 0.
    cluster_iter : int
        Number of initializations to perform for the Gaussian Mixture model.
    tol : float
        Tolerance used in the Gaussian mixture model.
    info_sweep : int
        Values for the different number of clusters evaluated. Example for `info_sweep` = 2 the cluster numbers from 
        [`number_cluster` - `info_sweep` , `number_cluster` + `info_sweep`) are evaluated.
    plot_sweep : bool
        Boolean to plot the number of cluster analysis.
            - Akaike information criterion (aic)
            - Bayesian information criterion (bic)
            - Silhouette score (silhouette)
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
                 cluster_iter : int = 10,
                 means_init : any = None,
                 tol : float = 1e-3,
                 seed : any = 42,
                 info_sweep : int = 5,
                 plot_sweep : bool = False,
                 width_plot : int = 6,
                 height_plot : int = 3,
                 dpi : int = 100) -> None:
        
        # Style
        self.style_name = "seaborn-v0_8"
        self.cmap = 'mako'#'magma' #'Blues'
        self.text_color = 'white'
        self.levels = 30
        self.width_plot = width_plot
        self.height_plot = height_plot
        self.gridsize_density = 20_000
        self.dpi = dpi
        self.seed = seed

        # Dataset
        self.dim = X_low.shape[1]
        self.X_high = np.array(X_high)
        self.X_low = X_low
        self.eps = 1e-10 # To reduce risk of divided by 0 in discrete integral
        self.tol = tol
        self.number_cluster = number_cluster
        for dim in range(self.dim):
            self.X_low[:,dim] = (self.X_low[:,dim] - self.X_low[:,dim].min()) \
                                / (self.X_low[:,dim].max() - self.X_low[:,dim].min())

        assert 0 < self.dim <= 2, \
            'The number of dimensions of X_low must be 1 or 2'
        assert self.X_low.shape[0] == self.X_high.shape[0], \
            'The number of sample in X_low and X_high must be equal'
        assert self.number_cluster < self.X_low.shape[0], \
            'The number of sample must be higher than the number of clusters'
        assert number_cluster - info_sweep > 1, \
            'The number of cluster must be higher than 1'
        
        if info_sweep > 0:
            size_sweep = 2 * info_sweep
        else:
            size_sweep = 1

        # Clustering iteration metrics
        self.aic = np.zeros(size_sweep)
        self.bic = np.zeros(size_sweep)
        self.silhouette = np.zeros(size_sweep)

        # Initial clustering iteration (find optimal number of cluster)
        if means_init is None:
            self.clustering_iter(info_sweep = info_sweep,
                             cluster_iter = cluster_iter,
                             plot_sweep = plot_sweep)
        else: 
            pass

        # Initial clustering parameters
        self.cluster_means = np.zeros((self.number_cluster, self.dim))
        self.cluster_covariances = np.zeros((self.number_cluster, self.dim, self.dim))
        self.cluster_weights = np.zeros(self.number_cluster)
        self.predict_ = None

        # Ordered clustering with initialized means
        self.clustering_order(cluster_iter = cluster_iter,
                              means_init = means_init)
        
        # Labels
        self.label_shift = label_shift
        self.labels = self.predict(self.X_low)
        self.unique_labels = np.arange(self.number_cluster) + label_shift

        # Confidence metrics
        self.confidence_1D = np.zeros(self.number_cluster)
        self.confidence_2D = np.zeros(self.number_cluster)
        self.confidence_poisson = None

        # Trustworthiness
        self.trustworthiness_eucl = np.zeros(self.number_cluster)
        self.trustworthiness_cos = np.zeros(self.number_cluster)


    def clustering_iter(self, info_sweep : int = 10,
                            cluster_iter : int = 10,
                            plot_sweep : int = False) -> None:
        """
        Iteration over interval [`number_cluster` - `info_sweep` , `number_cluster` + `info_sweep`) to find an optimal number of
        clusters for Gaussian Mixture. 

        Parameters
        ----------
        cluster_iter : int
            Number of initializations to perform for the Gaussian Mixture model.
        info_sweep : int
            Values for the different number of clusters evaluated. Example for info_sweep = 2 the cluster numbers from 
            [`number_cluster` - `info_sweep` , `number_cluster` + `info_sweep`) are evaluated.
        plot_sweep : bool
            Boolean to plot the number of cluster analysis.
                - Akaike information criterion (aic)
                - Bayesian information criterion (bic)
                - Silhouette score (silhouette)
        
        Returns
        -------
        None

        """
        if info_sweep > 0:
            number_array = np.arange(self.number_cluster - info_sweep,
                                     self.number_cluster + info_sweep)
        else:
            number_array = np.array([self.number_cluster])

        for index, n_cluster in enumerate(number_array):

            fit_ = GaussianMixture(n_components = n_cluster, 
                                    tol = self.tol, 
                                    max_iter = 200,
                                    n_init = -(cluster_iter//-2),
                                    init_params='k-means++',
                                    random_state = self.seed).fit(self.X_low)
            
            self.aic[index] = fit_.aic(self.X_low)
            self.bic[index] = fit_.bic(self.X_low)
            self.silhouette[index] = silhouette_score(self.X_low, fit_.predict(self.X_low))
        
        self.number_cluster = number_array[np.argmax(self.silhouette)]

        if plot_sweep:

            with plt.style.context("seaborn-v0_8"):
                fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (self.width_plot,self.height_plot))
                axes[0].plot(number_array, self.aic, label='AIC')
                axes[0].plot(number_array, self.bic, label='BIC')
                """
                axes[0].scatter(x = self.number_cluster, 
                                y = self.aic[np.argmax(self.silhouette)], 
                                s = 100,
                                zorder = 10,
                                marker = 'X')
                axes[0].scatter(x = self.number_cluster, 
                                y = self.bic[np.argmax(self.silhouette)], 
                                s = 100, 
                                label = 'Selected number',
                                zorder = 10,
                                marker = 'X')
                """
                axes[0].set_ylabel('AIC / BIC')
                axes[0].set_xlabel('Number of cluster')
                axes[0].set_xticks(number_array[::4])
                axes[0].legend()

                axes[1].plot(number_array, self.silhouette, label = 'Silhouette')
                """
                axes[1].scatter(x = self.number_cluster, 
                                y = self.silhouette[np.argmax(self.silhouette)], 
                                s = 100, 
                                label = 'Selected number',
                                zorder = 10,
                                marker = 'X')
                """
                axes[1].set_ylabel('Silhouette score')
                axes[1].set_xlabel('Number of cluster')
                axes[1].set_xticks(number_array[::4])
                axes[1].legend()

                fig.tight_layout()
                fig.show()

    
    def clustering_order(self, cluster_iter : int = 10,
                               means_init : any = None) -> None:
        """
        Definition of the Gaussian Mixture for the optimal number of clusters.
        The labels are also initialized to follow photon numbers (label shift and area of the initial signals).

        Parameters
        ----------
        cluster_iter : int
            Number of initializations to perform for the Gaussian Mixture model.
        
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
                                
        # Get area and labels
        X_Area = trapezoid(self.X_high, axis=1).flatten()
        predict_init = fit_.predict(self.X_low)
        # Get average area of the clusters
        labels = np.arange(self.number_cluster)
        means_area = [np.mean(X_Area[predict_init == label_]) for label_ in labels]
        # Order clusters
        means_area , labels = zip(*sorted(zip(means_area, labels)))
        means_init = fit_.means_[list(labels)]
        
        fit_ = GaussianMixture(n_components = self.number_cluster, 
                                tol = self.tol,
                                max_iter = 200,
                                n_init = cluster_iter,
                                init_params = 'k-means++',
                                means_init = means_init,
                                random_state = self.seed).fit(self.X_low)
        
        self.cluster_means[:,:] = fit_.means_
        self.cluster_covariances[:,:,:] = fit_.covariances_
        self.cluster_weights[:] = fit_.weights_
        self.predict_ = fit_.predict


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
        return self.predict_(X_low) + self.label_shift
        

    def plot_density(self, bw_adjust : float = 1.,
                           bw_adjust_x : float = 1.,
                           bw_adjust_y : float = 1.,
                           plot_sides : bool = False) -> None:
        """
        Plot the kernel density estimation of the latent space.

        Parameters
        ----------
        bw_adjust : float
            Bandwidth used in the kernel density estimation.
        bw_adjust_x : float
            Bandwidth used in the 2D kernel density estimation (x plot).
        bw_adjust_y : float
            Bandwidth used in the 2D kernel density estimation (y plot).
        sides : bool
            Bollean to plot 1D side views of the 2D space.

        Returns
        -------
        None

        """
        if self.dim == 1:

            with plt.style.context(self.style_name):
                plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
                sns.kdeplot(x = np.array(self.X_low).flatten(), 
                            cmap = self.cmap,  
                            fill = True,
                            antialiased = True,
                            bw_adjust = bw_adjust,
                            gridsize = self.gridsize_density)
                #kde.tick_params(left=False, bottom=False)
                plt.ylabel('Density')
                plt.xlabel('Latent space')
                plt.show()

        elif self.dim == 2:

            with plt.style.context(self.style_name):
                plt.figure(figsize=(self.width_plot,self.width_plot), dpi=self.dpi)
                if plot_sides:
                    g = sns.JointGrid(x = self.X_low[:,0], 
                                    y = self.X_low[:,1],
                                    space = 0)
                    g = g.plot_joint(sns.kdeplot, 
                                    cmap = self.cmap,
                                    bw_adjust = bw_adjust,
                                    fill = True, 
                                    thresh = 0,
                                    levels = self.levels)
                    sns.kdeplot(x = self.X_low[:,0], 
                                cmap = self.cmap,
                                fill = True,
                                bw_adjust = bw_adjust_x, 
                                ax = g.ax_marg_x,
                                gridsize = self.gridsize_density)
                    sns.kdeplot(y = self.X_low[:,1], 
                                cmap = self.cmap,
                                fill = True,
                                bw_adjust = bw_adjust_y, 
                                ax = g.ax_marg_y,
                                gridsize = self.gridsize_density)
                    g.set_axis_labels(xlabel=r'$s_1$', 
                                  ylabel=r'$s_2$')
                else:
                    sns.kdeplot(x = self.X_low[:,0], 
                                y = self.X_low[:,1], 
                                cmap = self.cmap,
                                fill = True,
                                bw_adjust = bw_adjust,
                                thresh = 0,
                                levels = self.levels)
                    for index, mean in enumerate(self.cluster_means):
                        if index == self.cluster_means.shape[0]-1:
                            plt.text(mean[0]+0.03,mean[1], f'{index}+', color = self.text_color)
                        else:
                            plt.text(mean[0]+0.03,mean[1], index, color = self.text_color)
                    plt.xlabel(r'$s_1$')
                    plt.ylabel(r'$s_2$')
                #kde.tick_params(left=False, bottom=False)
                
                plt.show()
      

    def plot_cluster(self, 
                     plot_kde : bool = False, 
                     number_bins : int = 1000,
                     bw_adjust : float = 1) -> None:
        """
        Plot the labelled clusters in the latent space.

        Parameters
        ----------
        plot_kde : bool
            Boolean to plot the kernel density estimation of every cluster instead of the histogram (1D) or the scatter plot (2D).
        number_bins : int
            Number of bins in the histogram (1D).
        bw_adjust : float
            Bandwidth used in the kernel density estimation.

        Returns
        -------
        None

        """
        if self.dim == 1:

            with plt.style.context(self.style_name):
                plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
                for index, (label, weight) in enumerate(zip(self.unique_labels, self.cluster_weights)):
                    X = self.X_low[self.labels == label]
                    if plot_kde:
                        sns.kdeplot(x = np.array(X).flatten(), 
                                    weights = weight,
                                    fill = True,
                                    antialiased = True,
                                    bw_adjust = bw_adjust,
                                    label = f"{index + self.label_shift}",
                                    gridsize = self.gridsize_density)

                    else:
                        plt.hist(X , 
                                bins = np.linspace(0, 1, number_bins), 
                                label = f"{index + self.label_shift}", 
                                fill = True, 
                                histtype = 'step')
                plt.xlabel("Latent Space")
                if plot_kde: plt.ylabel("Density")
                else : plt.ylabel("Counts")
                plt.legend(ncol=3)
                plt.show()

        elif self.dim == 2:

            with plt.style.context(self.style_name):
                fig = plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
                ax = fig.add_subplot()
                if plot_kde:
                    sns.kdeplot(x = self.X_low[:,0], 
                        y = self.X_low[:,1],
                        fill = True,
                        antialiased = True,
                        cmap = self.cmap,
                        thresh = 0,
                        bw_adjust = bw_adjust,
                        levels = self.levels)
                for index, (label, mean) in enumerate(zip(self.unique_labels, self.cluster_means)):
                    X = self.X_low[self.labels == label]
                    ax.scatter(x = X[:,0], 
                            y = X[:,1],
                            s = 1,
                            alpha = 0.02)
                    ax.text(mean[0]-0.01,mean[1]-0.01, index, color = self.text_color)
                plt.xlabel('Dimension 1 of the latent space')
                plt.ylabel('Dimension 2 of the latent space')
                plt.show()


    
    
    def multi_gaussian_1d(self, x : np.array, 
                                axis : any = None) -> np.array:
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
        
        for index, (mean, covariance, weight) in enumerate(zip(self.cluster_means,
                                                               self.cluster_covariances,
                                                               self.cluster_weights)):
            if axis != None:
                mean = mean[axis]
                covariance = covariance[axis,axis]
            else:
                pass

            multi_gaussian[index,:] = weight * multivariate_normal(mean = mean, cov = covariance).pdf(x)

        return multi_gaussian

    
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
            Array of shape (`number_cluster` , `x.size`) containing a discreet 1D Gaussian for every cluster

        """
        multi_gaussian = np.zeros((self.number_cluster, x.size, y.size))
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))

        for index, (mean, covariance, weight) in enumerate(zip(self.cluster_means,
                                                               self.cluster_covariances,
                                                               self.cluster_weights)):
            multi_gaussian[index,:,:] = weight * multivariate_normal(mean = mean, cov = covariance).pdf(pos)

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
                                 expected_prob : any = None,
                                 axis : any = None,
                                 n_points : int = 1000,
                                 size_zone : float = 1000.,
                                 plot_int : bool = False) -> None:
        """

        1D confidence metric following :

        P. C. Humphreys et al., ‘Tomography of photon-number resolving continuous-output detectors’, 
        New J. Phys., vol. 17, no. 10, p. 103044, Oct. 2015, doi: 10.1088/1367-2630/17/10/103044.
        
        Parameters
        ----------
        average_poisson : float
            Average photon number considering an expected Poisson distribution (will be included in the 
            definition of the confidence). 
        axis : any
            Axis to consider in the case of 1D confidence evaluation for a 2D latent space. 
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
        for index , (mean, covariance, weight) in enumerate(zip(self.cluster_means, 
                                                        self.cluster_covariances,
                                                        self.cluster_weights)):
            
            if self.dim > 1:
                x = np.linspace(mean[axis] - size_zone*covariance[axis,axis],
                                mean[axis] + size_zone*covariance[axis,axis],
                                n_points).flatten()
            else:
                x = np.linspace(mean - size_zone*covariance,
                                mean + size_zone*covariance,
                                n_points).flatten()
            
            p_sn = self.multi_gaussian_1d(x, axis = axis)

            if average_poisson is not None:
                p_n = poisson(mu = average_poisson).pmf(k = self.unique_labels) 
                p_s = np.sum(p_sn * p_n, axis = 0)
                conf_integral = p_sn[index]**2 * p_n / (p_s + self.eps) #* p_sn[index]
            elif expected_prob is not None:
                p_n = expected_prob[:self.number_cluster].reshape(-1,1)
                p_s = np.sum(p_sn * p_n, axis = 0)
                conf_integral = p_sn[index]**2 * p_n[index] / (p_s + self.eps) #* p_sn[index]
            else:
                p_s = np.sum(p_sn, axis = 0)
                conf_integral = p_sn[index]**2 / (p_s + self.eps) #* p_sn[index]

            if plot_int:
                with plt.style.context(self.style_name):
                    plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
                    plt.plot(x, conf_integral)
                    plt.show()

            self.confidence_1D[index] = trapezoid(x = x, y = conf_integral) / weight


        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
            plt.plot(self.unique_labels[:-1], self.confidence_1D[:-1])
            plt.title(f"1D confidence over axis {axis}")
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()

    
    def plot_confidence_2d(self, average_poisson : float = None,
                                 expected_prob : any = None,
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
        for index , (mean, covariance, weight) in enumerate(zip(self.cluster_means, 
                                                        self.cluster_covariances,
                                                        self.cluster_weights)):
            
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
                conf_integral = p_sn[index] * p_n / (p_s + self.eps) * p_sn[index]
            elif expected_prob is not None:
                p_n = expected_prob[self.unique_labels].reshape(-1,1,1)
                p_s = np.sum(p_sn * p_n, axis = 0)
                conf_integral = p_sn[index] * p_n[index] / (p_s + self.eps) * p_sn[index]
            else:
                p_s = np.sum(p_sn, axis = 0)
                conf_integral = p_sn[index] / (p_s + self.eps) * p_sn[index]

            if plot_int:
                with plt.style.context(self.style_name):
                    plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
                    plt.imshow(conf_integral, cmap = self.cmap)
                    plt.colorbar()
                    plt.show()

            self.confidence_2D[index] = self.trapezoid_2d(x, y, conf_integral) / weight


        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
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
        for index, label in enumerate(self.unique_labels):
            X_low = self.X_low[self.labels <= label]
            X_high = self.X_high[self.labels <= label]

            self.trustworthiness_eucl[index] = trustworthiness(X_high, X_low, metric="euclidean")
            self.trustworthiness_cos[index] = trustworthiness(X_high, X_low, metric="cosine")
        
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
            plt.plot(self.unique_labels, self.trustworthiness_eucl, label='Euclidean')
            plt.plot(self.unique_labels, self.trustworthiness_eucl, label='Cosine')
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
        color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*self.number_cluster))))
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
            for label in self.unique_labels:
                cluster = self.X_high[self.labels == label]
                c = next(color)
                if len(cluster) > 1000: cluster = cluster[:1000]
                [plt.plot(i, alpha=0.05, c=c) for i in cluster]

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
        color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*self.number_cluster))))
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.width_plot,self.height_plot), dpi=self.dpi)
            for label in self.unique_labels:
                cluster = self.X_high[self.labels == label]
                c = next(color)
                plt.plot(np.mean(cluster, axis=0), c=c)

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('average_traces.svg',format="svg", transparent=True)

    
  
