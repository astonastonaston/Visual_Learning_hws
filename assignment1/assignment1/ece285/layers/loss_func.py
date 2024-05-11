from .base_layer import BaseLayer
import numpy as np


class CrossEntropyLoss(BaseLayer):
    def __init__(self):
        self.cache = None
        pass

    def forward(self, input_x: np.ndarray, target_y: np.ndarray):
        """
        TODO: Implement the forward pass for cross entropy loss function

        """
        N, _ = input_x.shape

        # Calculate the sum of losses for each example, loss for one example -log(e_i/sum(e_j)) where i is the
        # correct class according to the label target_y and j is sum over all classes
        target_prob = input_x[np.arange(N), target_y][:, None] 
        loss = - np.log(target_prob)
        loss = np.sum(loss)

        # # Normalize the loss by dividing by the total number of samples N
        loss /= N

        # Store your loss output and input and targets in cache
        self.cache = [loss.copy(), input_x.copy(), target_y.copy()]
        return loss

    def backward(self):
        # Retrieve data from cache to calculate gradients
        loss_temp, x_temp, y_temp = self.cache
        N, _ = x_temp.shape

        # Create a target probability map with 1s at the correct class indices
        t_map = np.zeros_like(x_temp)
        t_map[np.arange(N), y_temp] = 1

        # Calculate the gradient of the loss with respect to the input scores
        dx = x_temp - t_map

        # Normalize the gradient by dividing by the total number of samples N
        dx /= N

        return dx

    def zero_grad(self):
        pass
