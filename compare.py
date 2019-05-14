import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(cm, classes, accuracy, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for k, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, k, format(cm[k, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[k, j] > thresh else 'black', fontsize=6, weight='bold')

    plt.ylabel('True label')
    plt.xlabel('Predicted label\n(Accuracy = {:0.4f})'.format(accuracy))
    plt.savefig('image/' + title + '.png')
    plt.show()


df = pd.read_csv('data/data.csv', header=None)

y = df[0]
x = df.drop(0, axis=1)
class_names = np.unique(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=97)

path = 'log'

if not os.path.exists(path):
    os.mkdir(path)

W1 = np.load(os.path.join(path, 'W1.npy'))
b1 = np.load(os.path.join(path, 'b1.npy')).reshape(-1, 1)
W2 = np.load(os.path.join(path, 'W2.npy'))
b2 = np.load(os.path.join(path, 'b2.npy')).reshape(-1, 1)

X_test = np.asarray(x_test).T
y_test = np.asarray(y_test)

Z1 = np.dot(W1.T, X_test) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
predict = np.argmax(Z2, axis=0)
y_predict = np.zeros(y_test.shape[0], dtype='str')

for i in range(len(predict)):
    y_predict[i] = chr(predict[i] + ord('A'))

# print('Accuracy score of MLP: %.2f' % (np.mean(y_predict == y_test)))

clf = load('log/model/model.joblib')

y_predict_sklearn = clf.predict(x_test)

# print('Accuracy score of MLP-sklearn: %.2f' % accuracy_score(y_predict_sklearn, y_test))

cnf_matrix = confusion_matrix(y_test, y_predict)
accuracy = np.mean(y_test == y_predict)

plot_confusion_matrix(cm=cnf_matrix,
                      accuracy=accuracy,
                      normalize=False,
                      classes=class_names,
                      title='Confusion Matrix of MLP')

plot_confusion_matrix(cm=cnf_matrix,
                      accuracy=accuracy,
                      normalize=True,
                      classes=class_names,
                      title='Confusion Matrix Normalized of MLP')

cnf_matrix = confusion_matrix(y_test, y_predict_sklearn)
accuracy = accuracy_score(y_test, y_predict_sklearn)

plot_confusion_matrix(cm=cnf_matrix,
                      accuracy=accuracy,
                      normalize=False,
                      classes=class_names,
                      title='Confusion Matrix of MLP-sklearn')

plot_confusion_matrix(cm=cnf_matrix,
                      accuracy=accuracy,
                      normalize=True,
                      classes=class_names,
                      title='Confusion Matrix Normalized of MLP-sklearn')
