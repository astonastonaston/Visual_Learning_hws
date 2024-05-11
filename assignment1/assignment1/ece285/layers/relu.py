from .base_layer import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        self.cache = None

    def forward(self, input_x: np.ndarray):
        # TODO: Implement RELU activation function forward pass
        # print(input_x.shape)
        output = np.maximum(input_x, 0)
        # print(output.shape)
        # Store the input in cache, required for backward pass
        self.cache = input_x.copy()
        return output

    def backward(self, dout):
        # Load the input from the cache
        x_temp = self.cache
        # Calculate gradient for RELU
        dx = np.zeros(x_temp.shape) # 0 or 1
        dx[x_temp >= 0] = 1
        return dx * dout
