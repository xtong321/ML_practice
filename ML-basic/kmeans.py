"""
try to implement K-means clustering in python

steps:
1) Decide how many clusters you want, i.e. choose k
2) Randomly assign a centroid to each of the k clusters
3) Calculate the distance of all observations to each of the k centroids
4) Assign observations to the closest centroid
5) Find the new location of the centroid by taking the mean of all the observations in each cluster
6) Repeat steps 3-5 until the centroids do not change position
"""

import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    def fit(self, X):
        # 1. Initialize centroids randomly
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # 2. Assign each data point to the closest centroid
            clusters = [[] for _ in range(self.n_clusters)]
            for i, point in enumerate(X):
                distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
                closest_centroid_idx = np.argmin(distances)
                clusters[closest_centroid_idx].append(i)

            # 3. Update centroids based on the mean of assigned data points
            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx, cluster_points_indices in enumerate(clusters):
                if len(cluster_points_indices) > 0:
                    new_centroids[cluster_idx] = np.mean(X[cluster_points_indices], axis=0)
                else:
                    # Handle empty clusters by re-initializing the centroid
                    new_centroids[cluster_idx] = X[np.random.choice(X.shape[0])]

            # 4. Check for convergence
            if np.array_equal(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        labels = []
        for point in X:
            distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid_idx = np.argmin(distances)
            labels.append(closest_centroid_idx)
        return np.array(labels)

# Example Usage:
if __name__ == "__main__":
    # Generate some sample data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Create and train the KMeans model
    kmeans = KMeans(n_clusters=4, max_iter=100)
    kmeans.fit(X)

    # Predict cluster labels for the data
    labels = kmeans.predict(X)

    # You can visualize the results (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()