from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# création du jeu de test
# This part of the code is creating a synthetic dataset using the `make_blobs` function from
# scikit-learn.
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

X = StandardScaler().fit_transform(X)

print("dimension de X", X.shape)
import matplotlib.pyplot as plt

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

print(labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)