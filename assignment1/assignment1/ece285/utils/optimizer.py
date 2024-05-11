import numpy as np
from ece285.layers.sequential import Sequential


class Optimizer:
    def __init__(self, model: Sequential):
        self.model = model

    def step(self):
        pass

    def zero_grad(self):
        pass


class SGD(Optimizer):
    def __init__(self, model: Sequential, lr: np.float32, weight_decay: np.float32):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for module in reversed(self.model._modules):
            for para, grad in zip(module.parameters, module.grads):
                assert grad is not None, "No Gradient"
                # para -= (self.lr * grad + self.weight_decay * para)
                para -= self.lr * (grad + self.weight_decay * para)

    def zero_grad(self):
        self.model.zero_grad()
