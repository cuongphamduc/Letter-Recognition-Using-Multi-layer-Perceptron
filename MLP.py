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


def to_one_hot(vector, dimension=26):
    res = np.zeros((len(vector), dimension))

    for index, value in enumerate(vector):
        res[index, ord(value) - ord('A')] = 1.

    return res


def check_loss(arr_loss, delta, number):
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


def cross_entropy_loss(y, y_pre):
    return -np.sum(y * np.log(y_pre)) / y.shape[1]


class MLP:
    def __init__(self, d0, d1, d2, learning_rate, tol,
                 n_iter_no_change, batch_size, num_epochs):
        self.W1 = 0.01 * np.random.randn(d0, d1)
        # W1 = (16, 100)
        self.b1 = np.zeros((d1, 1))
        # b1 = (100, 1)
        self.W2 = 0.01 * np.random.randn(d1, d2)
        # W2 = (100, 26)
        self.b2 = np.zeros((d2, 1))
        # b2 = (26, 1)
        self.learning_rate = learning_rate
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y):
        N = X.shape[1]
        loss = []

        with tqdm(total=self.num_epochs, file=sys.stdout, desc='Training') as pbar:
            for epoch in range(self.num_epochs):
                tmp = []
                X = X.T
                # X = (16000, 16)
                y = y.T
                # y =(16000, 16)

                X, y = sklearn.utils.shuffle(X, y)

                X = X.T
                # X = (16, 16000)
                y = y.T
                # y =(26, 16000)

                for j in range(0, N, self.batch_size):
                    # Pick a mini-batch
                    X_batch = X[:, j: j + self.batch_size]
                    # X_batch = (16, 50)
                    y_batch = y[:, j: j + self.batch_size]
                    # y_batch =(26, 50)

                    # Feedforward
                    Z1 = np.dot(self.W1.T, X_batch) + self.b1
                    # Z1 = (100, 16) x (16, 50) + (100, 1) = (100, 50)
                    A1 = np.maximum(Z1, 0)
                    # A1 = (100, 50)
                    Z2 = np.dot(self.W2.T, A1) + self.b2
                    # Z2 = (26, 100) x (100, 50) + (26, 1) = (26, 50)
                    y_predict = softmax(Z2)
                    # y_predict = (26, 50)

                    tmp.append(cross_entropy_loss(y_batch, y_predict))

                    # Backpropagation
                    E2 = (y_predict - y_batch) / N
                    # E2 = (26, 50)
                    dW2 = np.dot(A1, E2.T)
                    # dW2 = (100, 50) x (50, 26) = (100, 26)
                    db2 = np.sum(E2, axis=1, keepdims=True)
                    # db2 = (26, 1)
                    E1 = np.dot(self.W2, E2)
                    # E1 = (100, 26) x (26, 50) = (100, 50)
                    E1[Z1 <= 0] = 0
                    # E1 = (100, 50)
                    dW1 = np.dot(X_batch, E1.T)
                    # dW1 = (16, 50) x (50, 100) = (16, 100)
                    db1 = np.sum(E1, axis=1, keepdims=True)
                    # db1 = (100, 1)

                    # Mini-batch gradient descent update
                    self.W1 += -self.learning_rate * dW1
                    # W1 = (16, 100)
                    self.b1 += -self.learning_rate * db1
                    # b1 = (100, 1)
                    self.W2 += -self.learning_rate * dW2
                    # W2 = (100, 26)
                    self.b2 += -self.learning_rate * db2
                    # b2 = (26, 1)

                loss.append(statistics.mean(tmp))
                pbar.write('Iteration %d, loss = %f' % (epoch + 1, loss[-1]))

                if check_loss(loss, self.tol, self.n_iter_no_change):
                    pbar.write('Training loss did not improve more than tol=%f'
                               'for %d consecutive epochs. Stopping.'
                               % (self.tol, self.n_iter_no_change))
                    pbar.close()
                    break

                pbar.update(1)
                sleep(1)

    def predict(self, X):
        Z1 = np.dot(self.W1.T, X) + self.b1
        # Z1 = (100, 16) x (16, 4000) + (100, 1) = (100, 4000)
        A1 = np.maximum(Z1, 0)
        # A1 = (100, 4000)
        Z2 = np.dot(self.W2.T, A1) + self.b2
        # Z2 = (26, 100) x (100, 4000) + (26, 1) = (26, 4000)

        return np.argmax(Z2, axis=0)  # = (4000,)

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

        np.save(os.path.join(path, 'W1'), self.W1)
        np.save(os.path.join(path, 'b1'), self.b1)
        np.save(os.path.join(path, 'W2'), self.W2)
        np.save(os.path.join(path, 'b2'), self.b2)


if __name__ == '__main__':
    df = pd.read_csv('data/data.csv', header=None)

    label = df[0]
    # label = (20000,)
    feature = df.drop(0, axis=1)
    # feature = (20000, 16)

    X_train, X_test, y_train, y_test = \
        train_test_split(feature, label, test_size=0.2, random_state=97)
    # X_train = (16000, 16), X_test  = (4000, 16),
    # y_train = (16000,), y_test = (4000,)

    X_train = np.asarray(X_train).T
    # X = (16, 16000)
    y_train = to_one_hot(y_train).T
    # y =(26, 16000)

    model = MLP(d0=16, d1=100, d2=26, learning_rate=1, tol=0.0001,
                n_iter_no_change=10, batch_size=50, num_epochs=2000)

    model.fit(X_train, y_train)
    model.save('log/weight')

    X_test = np.asarray(X_test).T
    # X_test = (16, 4000)
    predict = model.predict(X_test)
    # predict = (4000,)

    y_test = np.asarray(y_test)
    # y_test = (4000,)

    for ind, val in enumerate(y_test):
        y_test[ind] = ord(val) - ord('A')

    print('Accuracy score: %.2f' % (np.mean(predict == y_test)))
