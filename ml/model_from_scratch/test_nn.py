
import numpy as np

def activation(num):
    return ((num*.1) + .5)

def return_nn_weights(m1, expected, w1=np.array([1, 1, 1])):
    prediction = m1 @ w1
    # print(prediction)

    Ps = np.array(list(map(activation, prediction)))
    print(Ps)

    accuracy = [1 if p > 0.5 else 0 for p in Ps]
    print(accuracy)

    accuracy = sum([x for i, p in enumerate(accuracy) if p == expected[i]])

data_m = np.array([[0, -1, 1],
          [-1, -3, 1],
          [2, -2, 1],
          [3, 1, 1],
          [1, -2, 1],
          [-1, 1, 1],
          [-2, -1, 1],
          [0, 3, 1],
          [-3, 0, 1],
          [-1, 2, 1],
         ])

expected = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

epoch_num = 3

# print("Initial data set:\n", data_m)

return_nn_weights(data_m, expected)
