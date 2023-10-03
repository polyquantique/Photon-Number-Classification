import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from AutoencoderAPI.utils.kernelDensity import kernel_density


import warnings
warnings.filterwarnings("ignore")

class compare():

    def __init__(self, bw = (-5, -2, 20)):
        self.metric_list = [    
                        ('Silhouette'                   , self.silhouette_kernel         , 1), 
                        ('Calinski Harabasz'            , self.calinski_harabasz_kernel  , 1),
                        ('Davies Bouldin'               , self.davies_bouldin_kernel     , 1),
                        ('trustworthiness Euclidian'   , self.trust_euclidian           , 1), 
                        ('trustworthiness Cosine'      , self.trust_cosine              , 1)
                        #('MSE'                          , self.MSE                       , 0)
                        #('Number of\ncluster'           , self.cluster_number            , 1)
                      ]
        self.bw = bw
        self.kd = None

    def MSE(self, X, X_reconst):
        """
        Mean squared error considering the reconstruction of a 
        dimensionality reduction technique. 

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_reconst : numpy.array
            Array containing the reconstructed samples from 
            their low-dimensionality representation.

        Returns
        -------
        MSE : float
            Mean squared error of the reconstruction.

        """
        return mean_squared_error(X, X_reconst)

    
    def trust_euclidian(self, X, X_low_dim, labels):
        """
        Trustworthiness considering Euclidian distance for the 
        original samples and their low-dimensional representation. 

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_low_dim : numpy.array
            Array containing the low-dimensional representation of 
            the original samples.

        Returns
        -------
        Trustworthiness : float
            Trustworthiness of the dimensionality reduction

        """
        return trustworthiness(X, X_low_dim, metric="euclidean")

    
    def trust_cosine(self, X, X_low_dim, labels):
        """
        Trustworthiness considering Cosine distance for the 
        original samples and their low-dimensional representation. 

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_low_dim : numpy.array
            Array containing the low-dimensional representation of 
            the original samples.

        Returns
        -------
        Trustworthiness : float
            Trustworthiness of the dimensionality reduction

        """
        return trustworthiness(X, X_low_dim, metric="cosine")

    
    def silhouette_kernel(self, X, X_low_dim, labels):
        """
        Silhouette score considering the distribution of samples in the 
        feature space. The labels used in the equation are defined using 
        kernel density estimation.

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_low_dim : numpy.array
            Array containing the low-dimensional representation of 
            the original samples.

        Returns
        -------
        Silhouette : float
            Silhouette score of the samples.

        """
        return self.score_kernel_density(X, X_low_dim, silhouette_score, labels)


    def calinski_harabasz_kernel(self, X, X_low_dim, labels):
        """
        Calinski-Harabasz index considering the distribution of samples in the 
        feature space. The labels used in the equation are defined using 
        kernel density estimation.

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_low_dim : numpy.array
            Array containing the low-dimensional representation of 
            the original samples.

        Returns
        -------
        Calinski-Harabasz : float
            Calinski-Harabasz index of the samples.

        """
        return self.score_kernel_density(X, X_low_dim, calinski_harabasz_score, labels)

    
    def davies_bouldin_kernel(self, X, X_low_dim, labels):
        """
        Davies Boudin score considering the distribution of samples in the 
        feature space. The labels used in the equation are defined using 
        kernel density estimation.

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_low_dim : numpy.array
            Array containing the low-dimensional representation of 
            the original samples.

        Returns
        -------
        Davies Boudin : float
            Davies Boudin score of the samples.

        """
        return self.score_kernel_density(X, X_low_dim, davies_bouldin_score, labels)

    
    def cluster_number(self, X, X_low_dim):
        """
        Give the number of clusters when separating the feature space 
        using kernel density estimation.

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_low_dim : numpy.array
            Array containing the low-dimensional representation of 
            the original samples.

        Returns
        -------
        n_cluster : float
            Number of clusters in the feature space.

        """
        return len(self.kd.clusters_low)


    def score_kernel_density(self, X, X_low_dim, score, labels = None):
        """
        Score considering the distribution of samples in the 
        feature space. The labels used in the equation are defined using 
        kernel density estimation.

        Parameters
        ----------
        X : numpy.array
            Array containing the samples.
        X_low_dim : numpy.array
            Array containing the low-dimensional representation of 
            the original samples.
        score : function
            Score function (silhouette_score, calinski_harabasz_score or 
            davies_bouldin_score)

        Returns
        -------
        score : float
            score of the samples.

        """
        if labels.any() == None:
            labels = self.kd.labels

        if len(np.unique(labels)) < 2:
            return 0
  
        return score(X_low_dim, labels)


    def quality_metrics_table(self, X_init, X_reconst, X_low_dim, Title):
        """
        Score the output of a list of dimensionality reduction techniques.
        The scores are defined as either reconstruction-based, manifold-based 
        or cluster-based. 

        - Silhouette score
        - Calinski Harabasz index
        - Davies Bouldin index
        - Trustworthiness (Euclidian distance)
        - Trustworthiness (Cosine distance)
        - Mean squared error (MSE)
        - Number of clusters

        Parameters
        ----------
        X_init : list of numpy.array
            Array containing the samples.
        X_reconst : list of numpy.array
            Reconstruction (or inverse transformation) of the samples for each technique.
        X_low_dim : list of numpy.array
            Low-dimensional representation of the samples for each technique.
        Title : list or str
            Name of the different dimensionality reduction techniques.

        Returns
        -------
        None

        """
        
        metric_list = self.metric_list
        scores = np.zeros((len(X_init) , len(metric_list)))

        for index_samples, X in tqdm(enumerate(X_init), desc='Method', total=len(Title)):
            self.kd = kernel_density(X_low_dim[index_samples], self.bw)
            for index_metric, (name, metric, metric_type) in tqdm(enumerate(metric_list), desc=f'{Title[index_samples]}' , total=len(metric_list)):
            
                if metric_type:
                    scores[index_samples][index_metric] = metric(X, X_low_dim[index_samples])
                else:
                    if X_reconst[index_samples].any() == None:
                        scores[index_samples][index_metric] = None
                    else:
                        scores[index_samples][index_metric] = metric(X, X_reconst[index_samples])
    
        norm_scores = (scores - np.nanmin(scores, axis=0)) / (np.nanmax(scores, axis=0) - np.nanmin(scores, axis=0))

        plt.figure(figsize=(10,5))
        plt.imshow(norm_scores,aspect='auto',cmap="GnBu_r")#, interpolation="bilinear")
        #plt.xlabel('Metric')
        #plt.ylabel('Method')
        plt.xticks(np.arange(len(metric_list)), labels=[i[0] for i in metric_list])
        plt.yticks(np.arange(len(Title)), labels=Title)

        for (j,i),label in np.ndenumerate(scores):
            plt.text(i,j,'{:.2e}'.format(label),ha='center',va='center')
            plt.text(i,j,'{:.2e}'.format(label),ha='center',va='center')

        plt.show()


    
    def quality_metric_plot(self, X_init, X_reconst, X_low_dim, Title, max_number_cluster):

        metric_list = self.metric_list
        scores = np.zeros((len(metric_list), len(X_init), max_number_cluster))


        for index_samples, X_init_it in enumerate(X_init):

            self.kd = kernel_density(X_low_dim[index_samples], self.bw)
            labels = self.kd.labels
            length = len(self.kd.clusters_low)

            for label in range(length):

                if label == 0:
                    continue
                condition = np.in1d(labels, range(label))
                X_init_temp = X_init_it[condition]
                X_low_dim_temp = X_low_dim[index_samples][condition]
                label_temp = labels[condition]
                #try:
                #    X_reconst_temp = X_reconst[index_samples][condition]
                #except:
                #    X_reconst_temp = None
                
                for index_metric, (name, metric, metric_type) in enumerate(metric_list):
                
                    if condition.all():
                        continue
                    
                    if metric_type:
                        scores[index_metric][index_samples][label] = metric(X_init_temp, X_low_dim_temp, label_temp)
                    #else:
                    #    if X_reconst_temp.any() == None:
                    #        scores[index_metric][index_samples][label] = None
                    #    else:
                    #        scores[index_metric][index_samples][label] = metric(X_init_temp, X_reconst_temp)
    
        for index_metric, metric_score in enumerate(scores):

            plt.figure(figsize=(10,5))
            plt.title(f"{self.metric_list[index_metric][0]}")

            for index_method, method in enumerate(X_init):
                sc = scores[index_metric][index_method]
                plt.plot(sc[sc != 0], label=f"{Title[index_method]}")

            plt.legend()
            plt.show()