import pandas
import math
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def computeCost(X, y, theta, Lambda):
    m = X.shape[0]
    cost = 1 - y * (np.dot(X, theta))
    cost[cost < 0] = 0  # max(0, 1 - y (wx)
    cost = np.sum(cost) / m + Lambda / 2 * np.power(np.linalg.norm(theta), 2)
    return cost


def normalize(X):
    # calculated mean and std of every column and store it in a row vector
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # subtract mean from data and divide it by the standard deviation
    X = np.subtract(X, mu)
    X = np.divide(X, sigma)
    return X


def gradientDescent(X, y, alpha, iterations):
    theta = np.zeros(len(X[0]))
    J = []
    for epoch in range(1, iterations):
        error = 0
        for i, x in enumerate(X):
            # It there is an error
            if (np.dot(y[i], np.dot(X[i], theta))) < 1:
                theta = theta + alpha * (np.dot(X[i], y[i]) + (-2 * (1 / iterations) * theta))
            else:
                theta = theta + alpha * (-2 * (1 / iterations) * theta)
        #J.append(computeCost(X, y, theta, 1 / iterations))
    return theta, J


def predict(X, theta):
    hypothesis = np.dot(X, theta)
    result = [1 if i >= 0 else 0 for i in hypothesis]
    return result


file = 'heart.csv'
data = pandas.read_csv(file)

# np.set_printoptions(threshold=np.inf)

X = data[['trestbps', 'chol','oldpeak']]

X_data = X[:242]
X_test = X[242:]
y = np.array(data['target'])
y = np.array([1 if i == 1 else -1 for i in y])
y_data = y[:242]
y_test = y[242:]
X_data = normalize(X_data)

X_data = np.append(X_data, np.ones((X_data.shape[0], 1)), axis=1)
X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)
w, J = gradientDescent(X_data, y_data, 0.005, 4000)
print(w)
colors = ['black' if i == -1 else 'red' for i in y_data]
plt.scatter(X_data[:, 0], X_data[:, 1], c=colors)

prediction = predict(X_test, w)
print(prediction)
c = 0
for i in range(len(y_test)):
    if y_test[i] == prediction[i]:
        c = c + 1
print("Accuracy of model is:", round(c / len(y_test) * 100), "%")
print(J)

x2 = [w[0] * w[2], w[1], -w[1], w[0]]
x3 = [w[0] * w[2], w[1], w[1], -w[0]]

x2x3 = np.array([x2, x3])
X_data, y_data, U, V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X_data, y_data, U, V, scale=1, color='blue')
plt.show()

