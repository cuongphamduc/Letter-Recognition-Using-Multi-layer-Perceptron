import matplotlib.pyplot as plt
import numpy as np

loss1 = []

file_path = 'log/log1.txt'
with open(file_path) as f:
    for line in f:
        loss1.append(float(line[-9:]))

loss2 = []

file_path = 'log/log2.txt'
with open(file_path) as f:
    for line in f:
        loss2.append(float(line[-11:-3]))

a = np.arange(0, 2000, 1)
b = np.arange(0, 500, 1)
plt.plot(a, loss1, color='b', label='MLP')
plt.plot(b, loss2, color='r', label='MLP-sklearn')

plt.xlabel('Training epoch')
plt.ylabel('Loss')
plt.title('Loss in Training Process')
plt.xticks(np.arange(0, 2001, 200))

plt.ylim([0.0, 3.5])
plt.yticks(np.arange(0.0, 3.5, 0.3))
plt.grid()
plt.legend()

plt.savefig('image/training.png')
plt.show()
