from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Load dataset
data = load_wine()
X = data.data
y = data.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to plot results
def plot_decomposition(X_transformed, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.colorbar()
    plt.show()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plot_decomposition(X_pca, 'PCA on Wine Dataset')

# SparsePCA
sparse_pca = SparsePCA(n_components=2, random_state=42)
X_sparse_pca = sparse_pca.fit_transform(X_scaled)
plot_decomposition(X_sparse_pca, 'SparsePCA on Wine Dataset')

# TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_scaled)
plot_decomposition(X_svd, 'TruncatedSVD on Wine Dataset')

# FastICA
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X_scaled)
plot_decomposition(X_ica, 'FastICA on Wine Dataset')