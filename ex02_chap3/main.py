# This projects does experiments on MNIST dataset.
# Created by RunTimeError2, June 5th, 2018

from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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


# Fetching the MNIST dataset
mnist = fetch_mldata('MNIST original')
print(mnist)

# Display the shape of each list of element
X, y = mnist['data'], mnist['target']
print('Shape of X is ', X.shape)
print('Shape of y is ', y.shape)
# There are 70,000 images in total and each image is 28*28


# Plot one of the images
Image_index = 36000
some_digit = X[Image_index]
some_digit_image = some_digit.reshape(28, 28)
print('Some_digit = \n', some_digit)
'''
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis('off')
plt.show()
print('The label of image No.', Image_index, ' is ', y[Image_index])
'''


# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Shuffle the sets
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Trainging a binary classifier, which only detects the digit 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# Use SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict(some_digit.reshape(1, -1)))  # The image is 0


# Measuring accuracy using cross_validation
'''
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
'''
# Use cross_val_score
'''
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))
'''

# Using confusion matrix
'''
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
print(confusion_matrix(y_train, y_train_pred))
'''

# If 5-detector is applied, precision_score and recall_score can be calculated
# precision_score(y_train_5, y_pred)
# recall_score(y_train_5, y_train_pred)
# f1_score is also available


# sklearn allows directly using the decision function
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# Plot precision and recall as functions of the threshold value using plt
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])


# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()


# Plot the roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# plot_roc_curve(fpr, tpr)
# plt.show()
# Measure the area under the curve(AUC)
print('ROC AUC score = ', roc_auc_score(y_train_5, y_scores))


# Test and compare the former classifier and RandomForestClassifier
'''
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show()
'''


# Handling multiple classes
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))
# Compare scores for each digit
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)
print(np.argmax(some_digit_scores))
print(sgd_clf.classes_)
print(sgd_clf.classes_[5])


# One Vs One(OvO) Classifier can also be applied, but it is more complex
'''
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)  # Should be 45
'''


# Training a RandomForestClassifier
print('Using RandomForestClassifier ...')
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))
print(forest_clf.predict_proba([some_digit]))


# Using StandardScaler to improve the performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# Calculate the Contusion Matrix
conf_mx = confusion_matrix(y_train, y_train_pred)
print('Confusion Matrix = \n', conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
# Focus only on the errors
np.fill_diagonal(conf_mx, 0)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# Analyzing some types of images, 3 and 5 for example
'''
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8, 8))
plt.subplot(221);  plot_digits(X_aa[:25], images_per_row=5)
...
plt.show()
'''


# Using my own picture
im = Image.open('pic.png')
im.show()
width, height = im.size
print('Size of the picture = ', width, '*', height)
im = im.convert('L')
data = im.getdata()
data = np.matrix(data, dtype='float') / 255.0
new_data = np.reshape(data, (height, width))
print(new_data)
