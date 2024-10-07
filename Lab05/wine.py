import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
wine = datasets.load_wine()
X = wine.data
Y = wine.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def UsingPCA():
    n_clusters = 3
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg_clustering.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_centers = np.array([X_pca[labels == i].mean(axis=0) for i in range(n_clusters)])

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, marker='x', color='black', label='Centers')
    plt.title('Agglomerative Clustering on Wine Dataset using PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

def UsingSparsePCA():
    n_clusters = 3
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg_clustering.fit_predict(X_scaled)

    sparse_pca = SparsePCA(n_components=2, random_state=42)
    X_sparse_pca = sparse_pca.fit_transform(X_scaled)

    cluster_centers = np.array([X_sparse_pca[labels == i].mean(axis=0) for i in range(n_clusters)])

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_sparse_pca[labels == i, 0], X_sparse_pca[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, marker='x', color='black', label='Centers')
    plt.title('Agglomerative Clustering on Wine Dataset using Sparse PCA')
    plt.xlabel('Sparse PCA Component 1')
    plt.ylabel('Sparse PCA Component 2')
    plt.legend()
    plt.show()

def UsingTruncatedSVD():
    n_clusters = 3
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg_clustering.fit_predict(X_scaled)

    truncated_svd = TruncatedSVD(n_components=2, random_state=42)
    X_truncated_svd = truncated_svd.fit_transform(X_scaled)

    cluster_centers = np.array([X_truncated_svd[labels == i].mean(axis=0) for i in range(n_clusters)])

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_truncated_svd[labels == i, 0], X_truncated_svd[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, marker='x', color='black', label='Centers')
    plt.title('Agglomerative Clustering on Wine Dataset using Truncated SVD')
    plt.xlabel('Truncated SVD Component 1')
    plt.ylabel('Truncated SVD Component 2')
    plt.legend()
    plt.show()

def UsingFastICA():
    n_clusters = 3
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg_clustering.fit_predict(X_scaled)

    fast_ica = FastICA(n_components=2, random_state=42)
    X_fast_ica = fast_ica.fit_transform(X_scaled)

    cluster_centers = np.array([X_fast_ica[labels == i].mean(axis=0) for i in range(n_clusters)])

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_fast_ica[labels == i, 0], X_fast_ica[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, marker='x', color='black', label='Centers')
    plt.title('Agglomerative Clustering on Wine Dataset using Fast ICA')
    plt.xlabel('Fast ICA Component 1')
    plt.ylabel('Fast ICA Component 2')
    plt.legend()
    plt.show()

def Dendrogram():
    linked = linkage(X_scaled, 'ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=Y, distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram for Agglomerative Clustering on Wine Dataset')
    plt.show()    

# UsingPCA()
# UsingSparsePCA()
# UsingTruncatedSVD()
UsingFastICA()

Dendrogram()
