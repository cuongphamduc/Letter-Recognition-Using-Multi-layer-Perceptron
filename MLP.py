import math
import os
import statistics
import sys
from time import sleep

import numpy as np
import pandas as pd
import sklearn.utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def to_one_hot(labels, dimension=26):
    res = np.zeros((len(labels), dimension))

    for index, label in enumerate(labels):
        res[index, ord(label) - ord('A')] = 1.

    return res


def check_lost(arr_loss, delta, number):
    if len(arr_loss) < 11:
        return False

    res = 0
    arr_loss = arr_loss[-11:]

    for k in range(1, len(arr_loss)):
        if math.fabs(arr_loss[k] - arr_loss[k - 1]) <= delta:
            res += 1

    if res == number:
        return True

    return False


def softmax(arr):
    arr_e = np.exp(arr - np.max(arr, axis=0, keepdims=True))
    res = arr_e / arr_e.sum(axis=0)

    return res


def cross_entropy_loss(Y, Y_pre):
    return -np.sum(Y * np.log(Y_pre)) / Y.shape[1]


df = pd.read_csv('data/data.csv', header=None)

y = df[0]  # y = (20000,)
x = df.drop(0, axis=1)  # x = (20000, 16)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=97)
# x_train = (16000, 16), x_test  = (4000, 16), y_train = (16000,), y_test = (4000,)

d0 = 16
d1 = 100
d2 = 26

W1 = 0.01 * np.random.randn(d0, d1)  # W1 = (16, 100)
b1 = np.zeros((d1, 1))  # b1 = (100, 1)
W2 = 0.01 * np.random.randn(d1, d2)  # W2 = (100, 26)
b2 = np.zeros((d2, 1))  # b2 = (26, 1)

loss = []
lr = 1
tol = 0.0001
n_iter_no_change = 10
batch_size = 50

X_train = np.asarray(x_train).T  # X_train = (16, 16000)
Y_train = to_one_hot(y_train).T  # Y_train =(26, 16000)
N = X_train.shape[1]

with tqdm(total=2000, file=sys.stdout, desc='Training') as pbar:
    for i in range(2000):
        tmp = []
        X_train = X_train.T  # X_train = (16000, 16)
        Y_train = Y_train.T  # Y_train =(16000, 16)

        X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train)

        X_train = X_train.T  # X_train = (16, 16000)
        Y_train = Y_train.T  # Y_train =(26, 16000)

        for j in range(0, N, batch_size):
            # Pick a mini-batch
            X_batch = X_train[:, j: j + batch_size]  # X_batch = (16, 50)
            Y_batch = Y_train[:, j: j + batch_size]  # Y_batch =(26, 50)

            # Feedforward
            Z1 = np.dot(W1.T, X_batch) + b1  # Z1 = (100, 16) x (16, 50) + (100, 1) = (100, 50)
            A1 = np.maximum(Z1, 0)  # A1 = (100, 50)
            Z2 = np.dot(W2.T, A1) + b2  # Z2 = (26, 100) x (100, 50) + (26, 1) = (26, 50)
            Y_predict = softmax(Z2)  # Y_predict = (26, 50)

            tmp.append(cross_entropy_loss(Y_batch, Y_predict))

            # Backpropagation
            E2 = (Y_predict - Y_batch) / N  # E2 = (26, 50)
            dW2 = np.dot(A1, E2.T)  # dW2 = (100, 50) x (50, 26) = (100, 26)
            db2 = np.sum(E2, axis=1, keepdims=True)  # db2 = (26, 1)
            E1 = np.dot(W2, E2)  # E1 = (100, 26) x (26, 50) = (100, 50)
            E1[Z1 <= 0] = 0  # E1 = (100, 50)
            dW1 = np.dot(X_batch, E1.T)  # dW1 = (16, 50) x (50, 100) = (16, 100)
            db1 = np.sum(E1, axis=1, keepdims=True)  # db1 = (100, 1)

            # Mini-batch gradient descent update
            W1 += -lr * dW1  # W1 = (16, 100)
            b1 += -lr * db1  # b1 = (100, 1)
            W2 += -lr * dW2  # W2 = (100, 26)
            b2 += -lr * db2  # b2 = (26, 1)

        loss.append(statistics.mean(tmp))
        pbar.write('Iteration %d, loss = %f' % (i + 1, loss[-1]))

        if check_lost(loss, tol, n_iter_no_change):
            pbar.write('Training loss did not improve more than tol=%f'
                       'for %d consecutive epochs. Stopping.'
                       % (tol, n_iter_no_change))
            pbar.close()
            break

        pbar.update(1)
        sleep(1)

X_test = np.asarray(x_test).T  # X_test = (16, 4000)
Z1 = np.dot(W1.T, X_test) + b1  # Z1 = (100, 16) x (16, 4000) + (100, 1) = (100, 4000)
A1 = np.maximum(Z1, 0)  # A1 = (100, 4000)
Z2 = np.dot(W2.T, A1) + b2  # Z2 = (26, 100) x (100, 4000) + (26, 1) = (26, 4000)
predict = np.argmax(Z2, axis=0)  # predict = (4000,)

Y_test = np.asarray(y_test)  # Y_test = (4000,)

for i in range(len(Y_test)):
    Y_test[i] = ord(Y_test[i]) - ord('A')

print('Accuracy score: %.2f' % (np.mean(predict == Y_test)))

path = 'log/'

if not os.path.exists(path):
    os.mkdir(path)

np.save(os.path.join(path, 'W1'), W1)
np.save(os.path.join(path, 'b1'), b1)
np.save(os.path.join(path, 'W2'), W2)
np.save(os.path.join(path, 'b2'), b2)
