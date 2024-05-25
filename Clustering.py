import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Laden des Iris-Datensatzes
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Normalisierung der Daten
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering mit K-Means für verschiedene Werte von K
k_values = [2, 3, 4]
kmeans_clusters = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    kmeans_clusters.append(kmeans.labels_)

# Clustering mit DBScan
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_scaled)

# Durchführen der PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Plot der in 3. erstellten Cluster mit PCA
fig = plt.figure(figsize=(15, 6))

# 3D Plot mit PCA für K-Means Cluster
ax1 = fig.add_subplot(121, projection='3d')
for i, target_name in zip(range(3), iris.target_names):
    ax1.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=target_name, marker='o')
for i, k in enumerate(k_values):
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans_clusters[i], cmap='viridis', label=f'K-Means (k={k})')
ax1.set_title('K-Means Clustering with PCA')
ax1.legend()

# 3D Plot mit PCA für DBScan Cluster
ax2 = fig.add_subplot(122, projection='3d')
for i, target_name in zip(range(3), iris.target_names):
    ax2.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=target_name, marker='o')
ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=dbscan_clusters, cmap='viridis', label='DBScan')
ax2.set_title('DBScan Clustering with PCA')
ax2.legend()

plt.tight_layout()
plt.show()

# Scatter-Diagramm für K-Means Cluster
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(3):
    plt.scatter(X_scaled[kmeans_clusters[i] == i, 0], X_scaled[kmeans_clusters[i] == i, 1], label=f'Cluster {i+1}')
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], marker='x', label=target_name)
plt.title('K-Means Clustering')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()

# Scatter-Diagramm für DBScan Cluster
plt.subplot(1, 2, 2)
for i in range(len(np.unique(dbscan_clusters))):
    plt.scatter(X_scaled[dbscan_clusters == i, 0], X_scaled[dbscan_clusters == i, 1], label=f'Cluster {i}')
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], marker='x', label=target_name)
plt.title('DBScan Clustering')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()

plt.tight_layout()
plt.show()
