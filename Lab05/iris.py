import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, confusion_matrix

iris = datasets.load_iris()
X = iris.data
Y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def UsingPCA():
    n_clusters = 3 
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', method='pam', random_state=42)
    kmedoids.fit(X_scaled)

    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    centers = X_scaled[medoid_indices]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(centers)

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=100, marker='x', color='black', label='Centers')
    plt.title('PAM Clustering on Iris Dataset')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

    print("Medoid Indices:", medoid_indices)
    print("Medoid centers in origianl feature space:\n", X[medoid_indices])

def UsingSparsePCA():
    n_clusters = 3 
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', method='pam', random_state=42)
    kmedoids.fit(X_scaled)

    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    centers = X_scaled[medoid_indices]

    sparse_pca = SparsePCA(n_components=2, random_state=42)
    X_sparse_pca = sparse_pca.fit_transform(X_scaled)
    centers_sparse_pca = sparse_pca.transform(centers)

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_sparse_pca[labels == i, 0], X_sparse_pca[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(centers_sparse_pca[:, 0], centers_sparse_pca[:, 1], s=100, marker='x', color='black', label='Medoids')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('K-Medoids Clustering with SparsePCA')
    plt.legend()
    plt.show()

    print("Medoid Indices:", medoid_indices)
    print("Medoid centers in origianl feature space:\n", X[medoid_indices])

def UsingTruncatedSVD():
    n_clusters = 3 
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', method='pam', random_state=42)
    kmedoids.fit(X_scaled)

    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    centers = X_scaled[medoid_indices]

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(X_scaled)
    centers_svd = svd.transform(centers)

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_svd[labels == i, 0], X_svd[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(centers_svd[:, 0], centers_svd[:, 1], s=100, marker='x', color='black', label='Medoids')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('K-Medoids Clustering with TruncatedSVD')
    plt.legend()
    plt.show()

    print("Medoid Indices:", medoid_indices)
    print("Medoid centers in origianl feature space:\n", X[medoid_indices])

def UsingFastICA():
    n_clusters = 3 
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', method='pam', random_state=42)
    kmedoids.fit(X_scaled)

    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    centers = X_scaled[medoid_indices]

    ica = FastICA(n_components=2, random_state=42)
    X_ica = ica.fit_transform(X_scaled)
    centers_ica = ica.transform(centers)

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(X_ica[labels == i, 0], X_ica[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(centers_ica[:, 0], centers_ica[:, 1], s=100, marker='x', color='black', label='Medoids')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('K-Medoids Clustering with FastICA')
    plt.legend()
    plt.show()

    print("Medoid Indices:", medoid_indices)
    print("Medoid centers in origianl feature space:\n", X[medoid_indices])
    

# UsingPCA()
# UsingSparsePCA()
# UsingTruncatedSVD()
UsingFastICA()
    
    