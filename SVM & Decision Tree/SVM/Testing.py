import SupportVectorMachine
import itertools
import pandas
import numpy as np


def tryFeatures(k):
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    return list(list(itertools.combinations(features, k)))


def Run(length, rate, size, epochs):
    featuresSet = tryFeatures(length)
    iterations = epochs
    learning_rate = rate
    training_size = size
    df = pandas.read_csv("heart.csv")
    df.loc[df['target'] == 0, 'target'] = -1  # replacing all zeroes in 'target' column with -1
    y = df['target'].to_numpy()

    print("Number of iterations are:", iterations)
    print("Learning rate value: ", learning_rate)
    print("Training set size is: ", training_size)

    for i in range(len(featuresSet)):
        currentFeatures = []
        for index in range(len(featuresSet[i])):
            currentFeatures.append(featuresSet[i][index])

        x = df[currentFeatures].to_numpy()
        x = np.insert(x, 0, 1, axis=1)

        x_train = x[:training_size]
        y_train = y[:training_size]

        x_test = x[training_size:]
        y_test = y[training_size:]

        svc = SupportVectorMachine.SupportVectorMachine(learning_rate, iterations)
        svc.fit(x_train, y_train)
        score = svc.score(x_test, y_test)
        print("----------------------------------------")
        print("Current Features \n", currentFeatures)
        print("Model Accuracy:", score, '%')
