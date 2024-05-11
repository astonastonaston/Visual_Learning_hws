from .base_layer import BaseLayer
import numpy as np


class Softmax(BaseLayer):
    """Implement the softmax layer
    Output of the softmax passes to the Cross entropy loss function
    The gradient for the softmax is calculated in the loss_func backward pass
    Thus the backward function here should be an empty pass"""

    def __init__(self):
        pass

    def forward(self, input_x: np.ndarray):
        # Remember input_x is not the input samples, but the output from the last layer just before softmax
        N, C = input_x.shape # 10*3
        # Thus here, N=Number of samples, C = number of classes
        # TODO: Implement the softmax layer forward pass
        # For each of the 10 outputs, the score is given by e_i/sum(e_j) where i is the output from ith class and j sums
        # over the outputs for all the classes. e here is the exponential function

        # scores matrix must be of the dimension NxC, where C is the number of classes
        # softmax trick: maximum subtraction normalization
        scores = input_x - np.max(input_x, axis=-1, keepdims=True)  # avoid numeric instability

        # Calculate softmax outputs e_i/sum(e_j)
        exp_input_x = np.exp(scores)
        exp_input_x_sum = np.sum(exp_input_x, axis=1)[..., None]
        softmax_matrix = exp_input_x / exp_input_x_sum
        # print(input_x)

        assert scores.shape == input_x.shape, "Scores must be NxC"

        return softmax_matrix

    def backward(self, dout):
        # Nothing to do here, pass. The gradient are calculated in the cross entropy loss backward function itself
        return dout
