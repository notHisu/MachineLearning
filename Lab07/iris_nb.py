import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = datasets.load_iris()
X = iris.data
y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)

"""
micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
Tạm dịch: Tính toán các chỉ số toàn cục bằng cách đếm tổng số true positives, false negatives và false positives.

macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
Tạm dịch: Tính toán các chỉ số cho từng nhãn và tìm unweighted mean. Không kể imbalance của nhãn.

weighted: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
Tạm dịch: Tính toán các chỉ số cho từng nhãn và tìm average weighted theo support (số lượng mẫu đúng của từng nhãn).
"""

