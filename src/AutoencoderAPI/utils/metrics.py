import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm


class silhouette_kmean():

    def __init__(self, feature, min_cluster, max_cluster):

        feature = np.array(feature).reshape(-1,1)[::10]
        scores = []

        for cluster_number in tqdm(range(min_cluster,max_cluster+1) , desc="Clusters") :
            predict = KMeans(n_clusters=cluster_number, random_state=42, algorithm='lloyd', n_init='auto').fit_predict(feature)
            if len(np.unique(predict)) != 1:
                scores.append(silhouette_score(feature, predict))
            else:
                scores.append(0)

        optimal_cluster = np.argmax(scores) + min_cluster
        km = KMeans(n_clusters=optimal_cluster, random_state=42, algorithm='lloyd', n_init='auto')
        fit = km.fit(feature)

        self.feature = feature

        self.scores = scores
        self.optimal_cluster = optimal_cluster
        
        self.fit = fit
        self.cluster_centers = fit.cluster_centers_

    def get_labels(self):
        return self.fit.labels_

    def get_centeroids(self):
        return self.fit.cluster_centers_
    
    def get_informations(self):
        labels = self.fit.labels_
        unique_label = np.unique(labels)

        try:
            optimal_score = silhouette_score(self.feature, labels)
        except:
            optimal_score = 0

        clusters = []
        for label in unique_label:
            clusters.append(self.feature[labels == label])

        centroids, clusters = zip(*sorted(zip(self.cluster_centers, clusters)))

        return self.scores, self.optimal_cluster, optimal_score, clusters