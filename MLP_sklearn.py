# import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def to_one_hot(labels, dimension=26):
    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):
        results[i, ord(label) - ord('A')] = 1.

    return results


df = pd.read_csv('data/data.csv', header=None)

y = df[0]
x = df.drop(0, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=97)

clf = MLPClassifier(hidden_layer_sizes=(100, 26), max_iter=2000,
                    learning_rate_init=0.001, batch_size=50, solver='sgd',
                    verbose=True, tol=0.0001, random_state=97, n_iter_no_change=10)
clf.fit(x_train, y_train)

dump(clf, 'log/model/model.joblib')
# clf = load('log/model/model.joblib')

print('Accuracy score: %.2f' % accuracy_score(clf.predict(x_test), y_test))

# W1 = np.asarray(clf.coefs_[0])
# W2 = np.asarray(clf.coefs_[1])
# b1 = np.asarray(clf.intercepts_[0])
# b2 = np.asarray(clf.intercepts_[1])
#
# path = 'log/MLP_sklearn'
#
# if not os.path.exists(path):
#     os.mkdir(path)
#
# np.save(os.path.join(path, 'W1'), W1)
# np.save(os.path.join(path, 'b1'), b1)
# np.save(os.path.join(path, 'W2'), W2)
# np.save(os.path.join(path, 'b2'), b2)
