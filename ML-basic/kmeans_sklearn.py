"""
implement K-means clustering with Sk-learn

(base) $ conda install matplotlib numpy pandas seaborn scikit-learn ipython
(base) $ conda install -c conda-forge kneed

#################################################
# Using scikit-learn for K-Means

from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Initialize KMeans with desired number of clusters (n_clusters)
# and other parameters like random_state for reproducibility
kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')

# Fit the model to the data
kmeans.fit(X)

# Get cluster assignments for each sample
labels = kmeans.labels_

# Get the coordinates of the cluster centroids
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Cluster Centroids:\n", centroids)
#################################################
"""

In [1]: import matplotlib.pyplot as plt
   ...: from kneed import KneeLocator
   ...: from sklearn.datasets import make_blobs
   ...: from sklearn.cluster import KMeans
   ...: from sklearn.metrics import silhouette_score
   ...: from sklearn.preprocessing import StandardScaler

In [2]: features, true_labels = make_blobs(
   ...:     n_samples=200,
   ...:     centers=3,
   ...:     cluster_std=2.75,
   ...:     random_state=42
   ...: )

In [3]: features[:5]
Out[3]:
array([[  9.77075874,   3.27621022],
       [ -9.71349666,  11.27451802],
       [ -6.91330582,  -9.34755911],
       [-10.86185913, -10.75063497],
       [ -8.50038027,  -4.54370383]])

In [4]: true_labels[:5]
Out[4]: array([1, 0, 2, 2, 2])

In [5]: scaler = StandardScaler()
   ...: scaled_features = scaler.fit_transform(features)


In [6]: scaled_features[:5]
Out[6]:
array([[ 2.13082109,  0.25604351],
       [-1.52698523,  1.41036744],
       [-1.00130152, -1.56583175],
       [-1.74256891, -1.76832509],
       [-1.29924521, -0.87253446]])

In [7]: kmeans = KMeans(
   ...:     init="random",
   ...:     n_clusters=3,
   ...:     n_init=10,
   ...:     max_iter=300,
   ...:     random_state=42
   ...: )


In [8]: kmeans.fit(scaled_features)
Out[8]:
KMeans(init='random', n_clusters=3, random_state=42)


In [9]: # The lowest SSE value
   ...: kmeans.inertia_
Out[9]: 74.57960106819854

In [10]: # Final locations of the centroid
   ...: kmeans.cluster_centers_
Out[10]:
array([[ 1.19539276,  0.13158148],
       [-0.25813925,  1.05589975],
       [-0.91941183, -1.18551732]])

In [11]: # The number of iterations required to converge
   ...: kmeans.n_iter_
Out[11]: 6

In [12]: kmeans.labels_[:5]
Out[12]: array([0, 1, 2, 2, 2], dtype=int32)


In [13]: kmeans_kwargs = {
   ...:     "init": "random",
   ...:     "n_init": 10,
   ...:     "max_iter": 300,
   ...:     "random_state": 42,
   ...: }
   ...:
   ...: # A list holds the SSE values for each k
   ...: sse = []
   ...: for k in range(1, 11):
   ...:     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
   ...:     kmeans.fit(scaled_features)
   ...:     sse.append(kmeans.inertia_)


   In [14]:  plt.style.use("fivethirtyeight")
   ...:  plt.plot(range(1, 11), sse)
   ...:  plt.xticks(range(1, 11))
   ...:  plt.xlabel("Number of Clusters")
   ...:  plt.ylabel("SSE")
   ...:  plt.show()


In [15]: kl = KneeLocator(
   ...:     range(1, 11), sse, curve="convex", direction="decreasing"
   ...: )

In [16]: kl.elbow
Out[16]: 3


In [17]: # A list holds the silhouette coefficients for each k
   ...: silhouette_coefficients = []
   ...:
   ...: # Notice you start at 2 clusters for silhouette coefficient
   ...: for k in range(2, 11):
   ...:     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
   ...:     kmeans.fit(scaled_features)
   ...:     score = silhouette_score(scaled_features, kmeans.labels_)
   ...:     silhouette_coefficients.append(score)


In [18]: plt.style.use("fivethirtyeight")
   ...: plt.plot(range(2, 11), silhouette_coefficients)
   ...: plt.xticks(range(2, 11))
   ...: plt.xlabel("Number of Clusters")
   ...: plt.ylabel("Silhouette Coefficient")
   ...: plt.show()


In [19]: from sklearn.cluster import DBSCAN
   ...: from sklearn.datasets import make_moons
   ...: from sklearn.metrics import adjusted_rand_score

In [20]: features, true_labels = make_moons(
   ...:     n_samples=250, noise=0.05, random_state=42
   ...: )
   ...: scaled_features = scaler.fit_transform(features)

In [21]: # Instantiate k-means and dbscan algorithms
   ...: kmeans = KMeans(n_clusters=2)
   ...: dbscan = DBSCAN(eps=0.3)
   ...:
   ...: # Fit the algorithms to the features
   ...: kmeans.fit(scaled_features)
   ...: dbscan.fit(scaled_features)
   ...:
   ...: # Compute the silhouette scores for each algorithm
   ...: kmeans_silhouette = silhouette_score(
   ...:     scaled_features, kmeans.labels_
   ...: ).round(2)
   ...: dbscan_silhouette = silhouette_score(
   ...:    scaled_features, dbscan.labels_
   ...: ).round (2)

In [22]: kmeans_silhouette
Out[22]: 0.5

In [23]: dbscan_silhouette
Out[23]: 0.38


In [25]: ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
   ...: ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)

In [26]: round(ari_kmeans, 2)
Out[26]: 0.47

In [27]: round(ari_dbscan, 2)
Out[27]: 1.0

+++++++++++++++++++++++++++++++++++++++++++++++++++
# https://realpython.com/k-means-clustering-python/#how-to-perform-k-means-clustering-in-python
In [1]: import tarfile
   ...: import urllib
   ...:
   ...: import numpy as np
   ...: import matplotlib.pyplot as plt
   ...: import pandas as pd
   ...: import seaborn as sns
   ...:
   ...: from sklearn.cluster import KMeans
   ...: from sklearn.decomposition import PCA
   ...: from sklearn.metrics import silhouette_score, adjusted_rand_score
   ...: from sklearn.pipeline import Pipeline
   ...: from sklearn.preprocessing import LabelEncoder, MinMaxScaler


#Download and extract the TCGA dataset from UCI:
In [2]: uci_tcga_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
   ...: archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

   ...: # Build the url
   ...: full_download_url = urllib.parse.urljoin(uci_tcga_url, archive_name)
   ...:
   ...: # Download the file
   ...: r = urllib.request.urlretrieve (full_download_url, archive_name)

   ...: # Extract the data from the archive
   ...: tar = tarfile.open(archive_name, "r:gz")
   ...: tar.extractall()
   ...: tar.close()


In [3]: datafile = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
   ...: labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"
   ...:
   ...: data = np.genfromtxt(
   ...:     datafile,
   ...:     delimiter=",",
   ...:     usecols=range(1, 20532),
   ...:     skip_header=1
   ...: )
   ...:
   ...: true_label_names = np.genfromtxt(
   ...:     labels_file,
   ...:     delimiter=",",
   ...:     usecols=(1,),
   ...:     skip_header=1,
   ...:     dtype="str"
   ...: )


In [4]: data[:5, :3]
Out[4]:
array([[0.        , 2.01720929, 3.26552691],
       [0.        , 0.59273209, 1.58842082],
       [0.        , 3.51175898, 4.32719872],
       [0.        , 3.66361787, 4.50764878],
       [0.        , 2.65574107, 2.82154696]])


In [5]: true_label_names[:5]
Out[5]: array(['PRAD', 'LUAD', 'PRAD', 'PRAD', 'BRCA'], dtype='<U4')