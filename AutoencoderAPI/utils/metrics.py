import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm


def silhouette_kmean(feature, max_cluster):

    feature = np.array(feature).reshape(-1,1)[::10]
    scores = []

    for cluster_number in tqdm(range(3,max_cluster+1) , desc="Clusters") :
        predict = KMeans(n_clusters=cluster_number, random_state=42, algorithm='lloyd', n_init='auto').fit_predict(feature)
        if len(np.unique(predict)) != 1:
            scores.append(silhouette_score(feature, predict))
        else:
            scores.append(0)

    optimal_cluster = np.argmax(scores) + 3

    km = KMeans(n_clusters=optimal_cluster, random_state=42)
    fit = km.fit(feature)

    centroids = fit.cluster_centers_
    labels = fit.labels_
    unique_label = np.unique(labels)

    try:
        optimal_score = silhouette_score(feature, labels)
    except:
        optimal_score = 0

    clusters = []
    for label in unique_label:
        clusters.append(feature[labels == label])

    centroids, clusters = zip(*sorted(zip(centroids, clusters)))

    return scores, optimal_cluster, optimal_score, clusters