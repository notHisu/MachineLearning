import numpy as np  
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Load dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
Y = cancer.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=3, min_samples=20)
labels = dbscan.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

core_samples_mask = labels != -1

anomaly_mask = labels == -1

plt.scatter(X_pca[core_samples_mask, 0], X_pca[core_samples_mask, 1], c=labels[core_samples_mask], cmap='viridis', label='Clustered Points')

plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], c='red', marker='x', label='Anomalies')

plt.title('DBSCAN Clustering with Anomaly Detection on Breast Cancer Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

n_anomalies = np.sum(labels ==-1)
print(f'Number of anomalies detected: {n_anomalies}')