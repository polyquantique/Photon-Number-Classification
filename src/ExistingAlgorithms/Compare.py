import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error


from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


import warnings
warnings.filterwarnings("ignore")

class compare():

    def __init__(self, config):
        
        self.min_cluster = config['min_cluster']
        self.max_cluster = config['max_cluster']
    

    def MSE(self, X, X_reconst):
        return mean_squared_error(X, X_reconst)
    

    def trust_euclidian(self, X, X_low_dim):
        return trustworthiness(X, X_low_dim, metric="euclidean")
    
    def trust_cosine(self, X, X_low_dim):
        return trustworthiness(X, X_low_dim, metric="cosine")
    

    def score_kmeans(self, X, X_low_dim, score):
        eval_cluster = range(self.min_cluster+1 , self.max_cluster+1)
        temp_score = []

        for cluster_index, cluster_number in enumerate(eval_cluster):
                
                clusters = KMeans(n_clusters=cluster_number, random_state=42, init="k-means++", n_init='auto').fit_predict(X_low_dim[::10])
                temp_score.append(score(X_low_dim[::10], clusters))

        opt_n_clusters = 23 #eval_cluster[np.argmax(temp_score)]
        opt_clusters = KMeans(n_clusters=opt_n_clusters, random_state=42, init="k-means++", n_init='auto').fit_predict(X_low_dim)

        return score(X_low_dim, opt_clusters)
    

    def silhouette_kmeans(self, X, X_low_dim):
         return self.score_kmeans(X, X_low_dim, silhouette_score)

    def calinski_harabasz_kmeans(self, X, X_low_dim):
        return self.score_kmeans(X, X_low_dim, calinski_harabasz_score)
    
    def davies_bouldin_kmeans(self, X, X_low_dim):
        return self.score_kmeans(X, X_low_dim, davies_bouldin_score)
    

    def quality_metrics(self, X_init, X_reconst, X_low_dim, Title):
        """
        0 : metric(X, X_reconst)
        1 : metric(X, X_low_dim)
        
        """
        
        metric_names = []
        metric_list = [
                        
                        
                        ('silhouette'         , self.silhouette_kmeans         , 1), 
                        ('Calinski Harabasz'  , self.calinski_harabasz_kmeans  , 1),
                        ('Davies Bouldin'     , self.davies_bouldin_kmeans     , 1),
                        ('trustworthiness\nEuclidian'   , self.trust_euclidian  , 1), 
                        ('trustworthiness\nCosine'      , self.trust_cosine  , 1), 
                        ('MSE'                      , self.MSE                 , 0)
        ]
        scores = np.zeros((len(X_init) , len(metric_list)))

        for index_samples, X in tqdm(enumerate(X_init), desc='Method', total=len(Title)):
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
        plt.imshow(norm_scores,aspect='auto')#, interpolation="bilinear")
        #plt.xlabel('Metric')
        #plt.ylabel('Method')
        plt.xticks(np.arange(len(metric_list)), labels=[i[0] for i in metric_list])
        plt.yticks(np.arange(len(Title)), labels=Title)
        #plt.colorbar()

        for (j,i),label in np.ndenumerate(scores):
            plt.text(i,j,'{:.2e}'.format(label),ha='center',va='center')
            plt.text(i,j,'{:.2e}'.format(label),ha='center',va='center')

        plt.show()




        
        

