import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# generate dataset
dataset = make_blobs(n_samples=200, centers=4, n_features=2, cluster_std=1.6, random_state=50)

points = dataset[0]
# create a Kmeans object
kmeans = KMeans(n_clusters=4)
# fit the KMeans object to the dataset
kmeans.fit(points)
plt.scatter(dataset[0][:, 0], dataset[0][0:, 1])

clusters = kmeans.cluster_centers_
y_km = kmeans.fit_predict(points)
plt.scatter(points[y_km == 0, 0], points[y_km == 0, 1], s=50, color='red')
plt.scatter(points[y_km == 1, 0], points[y_km == 1, 1], s=50, color='blue')
plt.show()
