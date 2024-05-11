import numpy as np


def get_classification_accuracy(y_prediction, y_ground_truth):
    assert y_ground_truth.shape == y_prediction.shape
    return np.sum(y_prediction == y_ground_truth) / np.prod(y_ground_truth.shape)
