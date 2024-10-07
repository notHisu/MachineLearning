import numpy as np
from sklearn.cluster import KMeans

# Tính ma trận khoảng cách Manhattan
X = np.array([7, 3, 4, 9, 8, 5, 7, 8, 1, 2])
Y = np.array([7, 7, 9, 6, 5, 8, 1, 4, 5, 5])
data = np.vstack((X, Y)).T
distance_matrix = np.sum(np.abs(data[:, None] - data), axis=2)

# Áp dụng thuật toán PAM
num_clusters = 2
pam = KMeans(n_clusters=num_clusters, random_state=0)
labels = pam.fit_predict(data)

# In nhãn cụm
print(labels)