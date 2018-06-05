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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image


def Image_to_matrix(filename):
    im = Image.open(filename)
    width, height = im.size
    print('The size of the picture is ', width, '*', height)
    im = im.convert('L')
    data = im.getdata()
    data = np.matrix(data, dtype='float')
    data = np.ones(shape=data.shape, dtype='float') * 255.0 - data
    plt.imshow(data.reshape(28, 28), cmap = matplotlib.cm.binary,
                        interpolation = 'nearest')
    plt.axis('off')
    plt.show()
    # print(new_data)
    return data


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


filename = 'pic.png'
data = Image_to_matrix(filename)


# Use SGDClassifier
print('Using SGDClassifier')
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
# Evaluate accuracy with cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))
print('result = ', sgd_clf.predict(data.reshape(1, -1)))


# Training a RandomForestClassifier
print('Using RandomForestClassifier ...')
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
# Evaluate accuracy with cross_val_score
print(cross_val_score(forest_clf, X_train, y_train, cv=3, scoring='accuracy'))
print('result = ', forest_clf.predict(data.reshape(1, -1)))
