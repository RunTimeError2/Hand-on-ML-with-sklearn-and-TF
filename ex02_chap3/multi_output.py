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


# Adding noise to images
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


# Pick a image and plot it
Image_index = 36000
some_image = X_train_mod[Image_index]
plt.imshow(some_image.reshape(28, 28), cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis('off')
plt.show()


def plot_digit(digit, width=28, height=28):
    plt.imshow(digit.reshape(width, height), cmap=matplotlib.cm.binary,
               interpolation='nearest')
    plt.axis('off')
    plt.show()


# Using KNN
print('Using KNeiborsClassifier')
Image_index = 2333
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[Image_index]])
plot_digit(X_test_mod[Image_index])
plot_digit(clean_digit)
