import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/data.csv', header=None)

y = df[0]
x = df.drop(0, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=97)

label_train = np.unique(y_train)
count_train = np.zeros(26, dtype=int)

for i in y_train:
    count_train[ord(i) - ord('A')] += 1

label_test = np.unique(y_test)
count_test = np.zeros(26, dtype=int)

for i in y_test:
    count_test[ord(i) - ord('A')] += 1

ind = np.arange(len(label_train))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

ax.set_xlabel('Labels')
ax.set_ylabel('Number of samples')
ax.set_title('Data distribution')
ax.bar(ind - width / 2, count_train, width, color='SkyBlue', label='Train')
ax.bar(ind + width / 2, count_test, width, color='IndianRed', label='Test')
ax.set_xticks(ind)
ax.set_xticklabels(label_train)
ax.set_yticks(np.arange(0, 800, 100))
ax.legend()

plt.grid()
plt.ylim([0, 800])
plt.savefig('image/distribution.png')
plt.show()
