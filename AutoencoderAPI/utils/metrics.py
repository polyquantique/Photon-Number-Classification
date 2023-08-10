import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def silhouette_kmean(feature, max_cluster):

    feature = np.array(feature).reshape(-1,1)
    scores = []

    for cluster_number in tqdm(range(2,max_cluster+1) , desc="Clusters") :
        predict = KMeans(n_clusters=cluster_number, random_state=42, algorithm='auto').fit_predict(feature[::10])
        scores.append(silhouette_score(feature[::10], predict))

    optimal_cluster = np.argmax(scores) + 1

    km = KMeans(n_clusters=optimal_cluster, random_state=42)
    fit = km.fit(feature)

    centroids = fit.cluster_centers_
    labels = fit.labels_
    unique_label = np.unique(labels)

    optimal_score = silhouette_score(feature, labels)

    clusters = []
    for label in unique_label:
        clusters.append(feature[labels == label])

    centroids, clusters = zip(*sorted(zip(centroids, clusters)))

    return scores, optimal_cluster, optimal_score, clusters