
from sklearn.cluster import AffinityPropagation
import numpy as np

# Sample data. Initially 2 variables
data = np.array([[1, 1], [1, 4], [1, 0],
                 [4, 1], [4, 4], [5, 0]])

#GOAL: introduce an EDA-based procedure to compute the preferences used by the AP algorithm. 

# Create and fit Affinity Propagation model
af = AffinityPropagation().fit(data)

# Get cluster centers
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Number of clusters:',n_clusters_)
print('Cluster centers:', data[cluster_centers_indices])
print('Labels:', labels)