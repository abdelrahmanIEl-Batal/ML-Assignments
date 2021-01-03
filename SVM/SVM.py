import pandas
import math
import numpy as np


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
    theta = np.zeros((X.shape[1], 1))
    Lambda = 1 / iterations
    J = [computeCost(X, y, theta, Lambda)]  # cost history
    N = X.shape[0]
    for i in range(iterations):
        for index, col in enumerate(X):
            hypothesis = y[index] * np.dot(X[index], theta)
            if hypothesis[0] >= 1:  # correct point
                theta = theta - alpha * (2 * Lambda * theta)
            else:
                theta = theta + alpha * (X[index] * y[index] - 2 * Lambda * theta)
        J.append(computeCost(X, y, theta, Lambda))
    return J, theta


def predict(X, theta):
    hypothesis = np.dot(X, theta)
    result = [1 if i >= 0 else 0 for i in hypothesis]
    return result


file = 'heart.csv'
data = pandas.read_csv(file)

# np.set_printoptions(threshold=np.inf)

X = data[['trestbps', 'chol', 'thalach', 'oldpeak']]

X_data = X[:242]
X_test = X[242:]
y = np.array(data[['target']])
y_data = y[:242]
y_test = y[242:]
X_data = normalize(X_data)
X_test = normalize(X_test)
X_data = np.append(X_data, np.ones((X_data.shape[0], 1)), axis=1)

X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

J, newTheta = gradientDescent(X_data, y_data, 0.1, 200)
print(J)
