import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from decision_tree import column
from neural_network import FCLayer, Activation, Network, sigmoid, sigmoid_prime, binary_crossentropy, \
    binary_crossentropy_prime


def set_up(columnName):
    # Network setup
    data = pd.read_csv('habitable.csv')

    # Columns from decision tree
    x_train = data[columnName].values
    y_train = data['P_HABITABLE'].values

    epochss = len(x_train)

    minX = min(x_train)
    maxX = max(x_train)

    for i in range(epochss):  # Normalization
        x_train[i] = (x_train[i] - minX) / (maxX - minX)

    # removes NaN values
    l = len(x_train)
    i = 0
    while i < l:
        if pd.isnull(x_train[i]):
            y_train = np.delete(y_train, i, axis=0)
            x_train = np.delete(x_train, i, axis=0)
            i -= 1
            l -= 1
        i += 1

    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    x_test = np.array(x_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    epochss = len(x_train)

    neural = Network()
    neural.add(FCLayer(1, 3))
    neural.add(Activation(sigmoid, sigmoid_prime))
    neural.add(FCLayer(3, 5))
    neural.add(Activation(sigmoid, sigmoid_prime))
    neural.add(FCLayer(5, 4))
    neural.add(Activation(sigmoid, sigmoid_prime))
    neural.add(FCLayer(4, 1))
    neural.add(Activation(sigmoid, sigmoid_prime))

    neural.use(binary_crossentropy, binary_crossentropy_prime)
    neural.fit(x_train, y_train, epochs=epochss, lr=0.01)

    return neural, minX, maxX


def runner(input):
    columnName = column()
    neural, minX, maxX = set_up(columnName)
    out = neural.predict_class(input, minX, maxX)
    print(f"Flux {input}: Habitable - {out}")
    return input, out


# input_list = []
# f = float(input(f"Enter input -> "))
# runner(f)
