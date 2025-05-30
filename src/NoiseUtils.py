from hdbscan import HDBSCAN, approximate_predict
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt



def assign_labels(X_l, X_h, skip, eps=0.0001,min_cluster_size=50):

   # 1. Fit on subsample
   X_train = X_l[::skip]
   X_full = X_l

   hdb = HDBSCAN(
      min_cluster_size=min_cluster_size,
      cluster_selection_epsilon=eps,
      prediction_data=True
   )
   hdb.fit(X_train)

   # 2. Predict all labels
   labels, strengths = approximate_predict(hdb, X_full)

   # 3. Find unassigned points (labeled -1)
   mask_unassigned = labels == -1
   X_unassigned = X_full[mask_unassigned]

   # 4. Get core points (those with cluster != -1)
   core_mask = hdb.labels_ != -1
   core_points = X_train[core_mask]
   core_labels = hdb.labels_[core_mask]

   # 5. Nearest neighbor search from core points
   nn = NearestNeighbors(n_neighbors=1).fit(core_points)
   _, indices = nn.kneighbors(X_unassigned)

   # 6. Assign label of nearest core point to each unassigned point
   forced_labels = core_labels[indices.flatten()]
   classes = labels.copy()
   classes[mask_unassigned] = forced_labels

   # 7. Sort clusters based on mean area in high-dimensional space
   unique_classes = np.unique(classes)
   mean_areas = [
      X_h[classes == cls].mean(axis=0).sum() for cls in unique_classes
   ]
   sorted_indices = np.argsort(mean_areas)
   sorted_classes = {cls: i for i, cls in enumerate(unique_classes[sorted_indices])}
   sorted_labels = np.vectorize(sorted_classes.get)(classes)

   return sorted_labels

def plot_clusters(X_l, labels, skip):

   # Calculate the mean position of each class
   unique_classes = np.unique(labels)
   class_means = [
      X_l[::skip][labels[::skip] == cls].mean(axis=0)
      for cls in unique_classes
   ]

   # Scatter plot for the classes
   plt.scatter(
      X_l[::skip, 0],
      X_l[::skip, 1],
      c=labels[::skip],
      alpha=0.5,
      s=0.5
   )

   # Annotate the mean position of each class with its class number
   for cls, mean_pos in zip(unique_classes, class_means):
      plt.text(mean_pos[0], mean_pos[1], str(cls), color="red", fontsize=12, ha="center")

   plt.colorbar()
   plt.xlabel('X_l_UMAP1[:,0]')
   plt.ylabel('X_l_UMAP1[:,1]')
   plt.show()
