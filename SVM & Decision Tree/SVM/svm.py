import SupportVectorMachine
import Testing
import pandas
import numpy as np
import itertools


def initializeLearningRates():
    learningRates = []
    for i in range(1, 10):
        learningRates.append(i / 10)
        learningRates.append(i / 100)
        learningRates.append(i / 1000)
        learningRates.append(i / 10000)
    return learningRates


def main():
    iterations = 2000
    training_size = 200

    df = pandas.read_csv("heart.csv")  # TODO: before running, make sure dataset is shuffled
    df.loc[df['target'] == 0, 'target'] = -1  # replacing all zeroes in 'target' column with -1

    y = df['target'].to_numpy()
    currentFeatures = ['cp', 'oldpeak', 'trestbps', 'chol', 'fbs', 'thalach', 'ca', 'thal']
    learningRates = initializeLearningRates()

    x = df[currentFeatures].to_numpy()
    x = np.insert(x, 0, 1, axis=1)

    x_train = x[:training_size]
    y_train = y[:training_size]

    x_test = x[training_size:]
    y_test = y[training_size:]

    print("Used features are\n", currentFeatures)
    print("Number of Iterations:", iterations)
    print("Training Data Size:", training_size)
    print("Testing Data Size:", len(df) - training_size)
    maxAccuracy = 0
    bestRate = 0
    for i in range(len(learningRates)):
        svc = SupportVectorMachine.SupportVectorMachine(learningRates[i], iterations)
        svc.fit(x_train, y_train)
        score = svc.score(x_test, y_test)
        print("------------------------------------------")
        print("Current Learning Rate:", learningRates[i])
        print("Model Accuracy:", score, '%')
        if score > maxAccuracy:
            maxAccuracy = score
            bestRate = learningRates[i]
    print("------------------------------------------")

    print("A learning rate of: ",bestRate," gives best accuracy of: ", maxAccuracy,"%")

    # optional
    test = input("- Click 'y' to display model testing with different features, press anything else to exit\n")
    if test == "y" or test == "Y":
        length = int(input("Enter lenght of subset\n"))
        Testing.Run(length, bestRate, training_size, iterations)


main()
