"""
Implementing K-Means from scratch with NumPy
"""

import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def kmeans_from_scratch(X, k, max_iterations=100):
    # Initialize centroids randomly from the data points
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        clusters = [[] for _ in range(k)]
        for i, sample in enumerate(X):
            distances = [euclidean_distance(sample, centroid) for centroid in centroids]
            closest_centroid_index = np.argmin(distances)
            clusters[closest_centroid_index].append(i)

        # Update centroids to the mean of the assigned data points
        new_centroids = np.zeros_like(centroids)
        for i, cluster_indices in enumerate(clusters):
            if cluster_indices:  # Avoid division by zero for empty clusters
                new_centroids[i] = np.mean(X[cluster_indices], axis=0)

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Get the final labels
    labels = np.zeros(X.shape[0], dtype=int)
    for i, sample in enumerate(X):
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)

    return labels, centroids

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Run the custom K-Means
labels_scratch, centroids_scratch = kmeans_from_scratch(X, k=2)

print("Cluster Labels (from scratch):", labels_scratch)
print("Cluster Centroids (from scratch):\n", centroids_scratch)