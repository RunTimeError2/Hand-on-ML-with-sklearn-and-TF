import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


# Fetching the MNIST dataset
mnist = fetch_mldata('MNIST original')
print(mnist)

# Display the shape of each list of element
X, y = mnist['data'], mnist['target']
print('Shape of X is ', X.shape)
print('Shape of y is ', y.shape)
# There are 70,000 images in total and each image is 28*28


# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Shuffle the sets
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Multilabel Classification
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

print('Using multilabel classification')
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

some_digit = X_train[2333]
print(knn_clf.predict([some_digit]))

# print(cross_val_score(knn_clf, X_train, y_multilabel, cv=3, scoring='accuracy'))
#y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
#print(f1_score(y_train, y_train_knn_pred, average='macro'))
