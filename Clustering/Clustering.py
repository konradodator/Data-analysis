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

# Farben und Marker definieren
colors = ['r', 'g', 'b', 'c', 'm', 'y']
markers = ['o', '^', 's']

# Plot der in 3. erstellten Cluster mit PCA für jedes K in K-Means
fig = plt.figure(figsize=(18, 18))
for i, k in enumerate(k_values):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    for j in range(k):
        ax.scatter(X_pca[kmeans_clusters[i] == j, 0], X_pca[kmeans_clusters[i] == j, 1], X_pca[kmeans_clusters[i] == j, 2], color=colors[j % len(colors)], label=f'Cluster {j + 1}')
    # Plot actual classes
    for class_idx in range(3):
        ax.scatter(X_pca[y == class_idx, 0], X_pca[y == class_idx, 1], X_pca[y == class_idx, 2], marker=markers[class_idx], edgecolor='k', facecolor='none', label=iris.target_names[class_idx])
    ax.set_title(f'K-Means Clustering mit PCA (k={k})')
    ax.legend()

# 3D Plot mit PCA für DBScan Cluster
ax = fig.add_subplot(2, 2, 4, projection='3d')
for cluster in np.unique(dbscan_clusters):
    ax.scatter(X_pca[dbscan_clusters == cluster, 0], X_pca[dbscan_clusters == cluster, 1], X_pca[dbscan_clusters == cluster, 2], color=colors[cluster % len(colors)], label=f'Cluster {cluster}')

for class_idx in range(3):
    ax.scatter(X_pca[y == class_idx, 0], X_pca[y == class_idx, 1], X_pca[y == class_idx, 2], marker=markers[class_idx], edgecolor='k', facecolor='none', label=iris.target_names[class_idx])
ax.set_title('DBScan Clustering mit PCA')
ax.legend()

plt.tight_layout()
plt.show()

# Scatter-Diagramm für K-Means Cluster und Klassen
plt.figure(figsize=(18, 18))



for i, k in enumerate(k_values):
    plt.subplot(2, 2, i + 1)
    for j in range(k):
        plt.scatter(X_scaled[kmeans_clusters[i] == j, 0], X_scaled[kmeans_clusters[i] == j, 1], color=colors[j % len(colors)], label=f'Cluster {j + 1}')
    for j in range(3):
        plt.scatter(X_scaled[y == j, 0], X_scaled[y == j, 1], marker=markers[j], edgecolor='k', facecolor='none', label=iris.target_names[j])
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
plt.tight_layout()
plt.show()

# Scatter-Diagramm für DBScan Cluster und Klassen
plt.figure(figsize=(9, 9))
for cluster in np.unique(dbscan_clusters):
    plt.scatter(X_scaled[dbscan_clusters == cluster, 0], X_scaled[dbscan_clusters == cluster, 1], label=f'Cluster {cluster}')
for j in range(3):
    plt.scatter(X_scaled[y == j, 0], X_scaled[y == j, 1], marker=markers[j], edgecolor='k', facecolor='none', label=iris.target_names[j])
plt.title('DBScan Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()

plt.tight_layout()
plt.show()

#Bewertung des Clustering Algorithmen
#K=2: Gute Trennung von der Setosa-Klasse. Allerdings leichte vermischung von Versicolor und Virginica
#K=3: Gute Trennung aller drei Klassen. Die Cluster entsprechend überwiegend den Klassen.
#k=4: Klassen werden in mehrere Cluster unterteilt. Dies ist eher weniger sinnvoll.
#DBScan: Dichte Clusterbildung, allerdings nicht eindeutig getrennt wie bei K=3
