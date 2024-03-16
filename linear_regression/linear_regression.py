import numpy as np
import matplotlib.pyplot as plt

lr = 0.01
n_iter = 100

x_train = np.array([[1], [2], [3], [4]])
y_train = np.array([[5], [10], [15], [20]])
x_train_b = np.c_[np.ones((len(x_train), 1)), x_train]
theta = np.random.randn(2, 1)

def cal_cost(theta, x, y):
    m = len(y)
    pred = x.dot(theta)
    cost = 1/(2*m)*np.sum(np.square(pred - y))
    return cost

def gradient_descent(x, y, theta, lr=0.01, n_iter=100):
    m = len(y)
    cost_history = np.zeros(n_iter)
    theta_history = np.zeros((n_iter, 2))
    for iter in range(n_iter):
        pred = np.dot(x, theta)
        theta = theta - (1/m)*lr*(x.T.dot(pred-y))
        cost_history[iter] = cal_cost(theta,x, y)
        theta_history[iter, :] = theta.T
    return theta, cost_history, theta_history

theta, cost_history, theta_history = gradient_descent(x_train_b, y_train, theta, lr, n_iter)

print('bias b: {}'.format(theta[0, 0]))
print('weight w: {}'.format(theta[1, 0]))
print('Final cost: {}'.format(cost_history[-1]))

plt.scatter(x_train, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.grid(True)
plt.show()

plt.plot(cost_history, 'k--', label='cost')
plt.plot(theta_history[:, 0], label='$b$')
plt.plot(theta_history[:, 1], label='$w$')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('cost')
plt.grid(True)
plt.show()