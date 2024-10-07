import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn_extra.cluster import KMedoids

iris = datasets.load_iris()
X = iris.data
Y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 3
pam = KMedoids(n_clusters=n_clusters, metric='euclidean', method='pam', random_state=42)
y_pred = pam.fit_predict(X_scaled)

if len(set(y_pred)) > 1:
    silhouette_avg = silhouette_score(X_scaled, y_pred)
    print(f'Silhouette Score: {silhouette_avg:.3f}')
else:
    print("Cannot calculate silhouette score with fewer than two clusters")

def PurityScore(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

purity = PurityScore(Y, y_pred)
print(f'Purity Score: {purity:.3f}')

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', label='Data Points')
plt.scatter(pam.cluster_centers_[:, 0], pam.cluster_centers_[:, 1], s=200, marker='x', color='red', label='PAM Centers')
plt.title('PAM Clustering on Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()

    