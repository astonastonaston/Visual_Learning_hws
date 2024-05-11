from .base_layer import BaseLayer
import numpy as np


class Sequential(BaseLayer):
    """
    Sequential is used to combine all the modules together and create a model
    For example combining Linear layers, Relu layers, softmax layers , etc. together
    Nothing to implement here, But go through the code and try to understand what is s going on"""

    def __init__(self, modules: list = None):
        self._modules = modules

    def forward(self, input_x):
        inter_x = input_x
        # Go through all the modules, passing the output of one to the input of the next one
        for module in self._modules:
            inter_x = module(inter_x)
        return inter_x

    def backward(self, dx):
        # Go through all the modules in reverse order, passing the gradient from one module to the pervious one
        for module in reversed(self._modules):
            dx = module.backward(dx)

    def zero_grad(self):
        # Initialize all the gradients for all modules to 0
        for module in self._modules:
            module.zero_grad()

    def predict(self, input_x):
        # Predict function used to output the predict class from the model
        output = self.forward(input_x)
        prediction = np.argmax(output, axis=-1)
        return prediction
