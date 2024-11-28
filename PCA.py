import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
data = np.random.uniform(1, 10, (30, 2))
data_center = data - data.mean(0)

C = np.cov(data_center.T)
eigenvalues, eigenvectors = np.linalg.eig(C)

idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

k = 1
tmp_v = eigenvectors[:, :k]

new_data = data_center @ tmp_v
new_data = new_data @ tmp_v.T + data.mean(0)

plt.figure(1, (6, 6))
size = 10
plt.xlim([-size, size])
plt.ylim([-size, size])

X = np.array([eigenvectors[:, 0] * -1, eigenvectors[:, 0]]) * size
X = X + data.mean(0)
Y = np.array([eigenvectors[:, 1] * -1, eigenvectors[:, 1]]) * size
Y = Y + data.mean(0)
plt.plot(X[:, 0], X[:, 1], label='new_x_axis', c='purple')
plt.plot(Y[:, 0], Y[:, 1], label='new_y_axis', c='purple')

plt.plot([-size, size], [0, 0], label='x_axis', c='black')
plt.plot([0, 0], [-size, size], label='y_axis', c='black')

plt.scatter(data[:, 0], data[:, 1], label='origin_data')
plt.scatter(new_data[:, 0], new_data[:, 1], label='new_data')

plt.legend(loc='best')
plt.show()
