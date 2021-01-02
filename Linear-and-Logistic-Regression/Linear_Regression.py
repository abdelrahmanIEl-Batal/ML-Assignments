import pandas
import numpy as np
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    m = len(y)
    hypothesis = np.dot(X, theta)
    diff = np.power(np.subtract(hypothesis, y), 2)
    cost = (1 / (2 * m)) * np.sum(diff)
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
    J = []  # cost history
    m = len(y)
    Error = []
    for i in range(iterations):
        hypothesis = np.dot(X, theta)
        diff = np.subtract(hypothesis, y)
        theta = np.subtract(theta, ((alpha / m) * (np.dot(np.transpose(X), diff))))
        J.append(computeCost(X, y, theta))
        Error.append(MSE(hypothesis, y))
    return J, theta, Error


def predict(X, theta):
    return np.dot(X, theta)


file = 'house_data.csv'
data = pandas.read_csv(file)

# this line to print whole numpy array without truncation
# np.set_printoptions(threshold=np.inf)

# linear regression with one variable
X = data[['sqft_living']]
X_data = X[:20000]
X_test = X[20000:]
X_data = normalize(X_data)
X_data = np.append(np.ones((X_data.shape[0], 1)), X_data, axis=1)
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
y = data[['price']] / 1000
y_data = y[:20000]
y_test = y[20000:]

J, newTheta, Error = gradientDescent(X_data, y_data, 0.1, 200)
print("Prediction using test Data: \n", np.round(predict(X_test, newTheta), 1))

plt.plot(list(range(200)), Error, '-b')
plt.show()
# --------------------------------------------------------------

# Linear Regression with multiple variables
X = data[['grade', 'bathrooms', 'lat', 'sqft_living', 'view']]
X_data = X[:20000]
X_test = X[20000:]
X_data = normalize(X_data)
X_data = np.append(np.ones((X_data.shape[0], 1)), X_data, axis=1)
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)

J, newTheta, Error = gradientDescent(X_data, y_data, 0.1, 200)

print("Error: ", np.round(Error))
print("Prediction using test Data: \n", np.round(predict(X_test, newTheta), 1))

plt.plot(list(range(200)), Error, '-r')
plt.show()
