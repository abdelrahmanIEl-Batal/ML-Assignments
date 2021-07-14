import numpy as np


class SupportVectorMachine:
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.w = np.empty(0)

    def f(self, x_vec):
        return np.dot(x_vec, self.w)

    # x: list of vectors, each vector is a row of values corresponding to a set of features
    # y: list of decision values
    def fit(self, x, y):
        height, width = x.shape
        self.w = np.ones(width)  # (w0, w1, ... wn)
        for _ in range(self.iterations):
            for i in range(height):  # for each row i in the dataset
                decision = self.f(x[i]) * y[i]
                if decision >= 1:
                    self.w -= 2.0 * self.alpha * (1.0 / self.iterations) * self.w
                else:
                    self.w += self.alpha * (y[i] * x[i] - (2.0 / self.iterations) * self.w)

    def score(self, x, y):
        correct = 0
        for i in range(len(y)):
            decision = y[i] * self.f(x[i])
            if decision >= 1:
                correct += 1
        return correct * 100.0 / len(y)
