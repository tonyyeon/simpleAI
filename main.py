import numpy as np
from NormalizingFunc import NormalizingFunc

training_data = np.array([[1,0,0],
                          [0,1,1],
                          [1,1,1],
                          [0,1,0]])
training_output = np.array([[1,0,1,0]]).T
weights = np.random.random((3,1)) * 2 - 1

for x in range(20000):
    input_layer = training_data

    output = NormalizingFunc.sigmoid(np.dot(input_layer, weights))

    error = training_output - output
    adjustments = error * NormalizingFunc.sigmoid_derivative(output)

    weights += np.dot(input_layer.T, adjustments)

print(output)
