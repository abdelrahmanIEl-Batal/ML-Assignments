import pandas
import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def computeCost(X, y, theta):
    m = X.shape[0]
    z = np.dot(X, theta)
    firstHalf = y * np.log(sigmoid(z))  # predict 1
    secondHalf = (1 - y) * np.log(1 - sigmoid(z))  # predict 0
    cost = -np.sum(firstHalf + secondHalf) / m
    return cost


def normalize(X):
    # calculated mean and std of every column and store it in a row vector
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # subtract mean from data and divide it by the standard deviation
    X = np.subtract(X, mu)
    X = np.divide(X, sigma)
    return X


def MSE(hypothesis, y):
    diff = np.subtract(hypothesis, y)
    diff = np.power(diff, 2)
    return np.sum(diff) / len(y)


def gradientDescent(X, y, alpha, iterations):
    theta = np.zeros((X.shape[1], 1))
    J = [computeCost(X, y, theta)]  # cost history
    N = X.shape[0]
    Error = []
    for i in range(iterations):
        hypothesis = sigmoid(np.dot(X, theta))
        theta = np.subtract(theta, alpha * (np.dot(np.transpose(X), np.subtract(hypothesis, y)) / N))
        J.append(computeCost(X, y, theta))
        Error.append(MSE(hypothesis, y))
    return J, theta, Error


def predict(X, theta):
    hypothesis = sigmoid(np.dot(X, theta))
    result = [1 if i >= 0.5 else 0 for i in hypothesis]
    return result


file = 'heart.csv'
data = pandas.read_csv(file)

#np.set_printoptions(threshold=np.inf)

X = data[['trestbps', 'chol', 'thalach', 'oldpeak']]

X_data = X[:290]
X_test = X[290:]
y = np.array(data[['target']])
y_data = y[:290]
y_test = y[290:]
X_data = normalize(X_data)
X_test = normalize(X_test)
X_data = np.append(np.ones((X_data.shape[0], 1)), X_data, axis=1)
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)

J, newTheta, Error = gradientDescent(X_data, y_data, 0.2, 400)
print("Error: \n",Error)
print("Prediction using test data: \n", predict(X_test, newTheta))
print(y_test.shape)