import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm


class silhouette_kmean():

    def __init__(self, feature, min_cluster, max_cluster):

        feature = np.array(feature).reshape(-1,1)
        feature_approx = np.array(feature).reshape(-1,1)[::10]
        scores = []

        # Number of cluster approximation
        for cluster_number in tqdm(range(min_cluster,max_cluster+1) , desc="Clusters") :
            predict = KMeans(n_clusters=cluster_number, random_state=42, algorithm='lloyd', n_init='auto').fit_predict(feature_approx)
            if len(np.unique(predict)) != 1:
                scores.append(silhouette_score(feature_approx, predict))
            else:
                scores.append(0)

        # Clustering using optimal number of cluster
        optimal_cluster = np.argmax(scores) + min_cluster
        km = KMeans(n_clusters=optimal_cluster, random_state=42, algorithm='lloyd', n_init='auto', tol=1e-12, max_iter=600)
        fit = km.fit(feature)

        # Map labels based on their position in latent space
        cluster_centers = fit.cluster_centers_
        unsorted_labels = fit.labels_
        unique_labels = range(len(cluster_centers))
        centroids, temp_labels = zip(*sorted(zip(fit.cluster_centers_, unique_labels)))

        mapping = {}
        for label, key in enumerate(temp_labels):
            mapping[key] = label

        sorted_labels = np.array([mapping[i] for i in unsorted_labels])
        

        # Get cluster based on sorted labels
        clusters = []
        for label in unique_labels:
            clusters.append(feature.flatten()[sorted_labels == label])
        self.clusters = clusters

        # Evaluation of Silhouette score
        try:
            self.optimal_score = silhouette_score(feature_approx, unsorted_labels[::10]) # APPROXIMATION
        except:
            self.optimal_score = 0

        self.feature = feature
        self.scores = scores
        self.optimal_cluster = optimal_cluster
        
        self.fit = fit
        self.sorted_labels = sorted_labels
        self.cluster_centers = cluster_centers
        self.label_mapping = mapping


