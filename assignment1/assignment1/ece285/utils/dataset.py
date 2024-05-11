import numpy as np


class DataLoader(object):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self._x_train = x_train
        self._x_val = x_val
        self._x_test = x_test

        self._y_train = y_train
        self._y_val = y_val
        self._y_test = y_test

        self._num_train = self._x_train.shape[0]
        self._num_val = self._x_val.shape[0]
        self._num_test = self._x_test.shape[0]

    def random_batch(
        self,
    ):
        pass

    def _iteration(
        self,
        batch_size: int,
        num_samples: int,
        x_samples: np.ndarray,
        y_samples: np.ndarray,
        shuffle: bool = True,
    ):
        indices = np.arange(num_samples)
        if shuffle:
            indices = np.random.permutation(num_samples)

        pos = 0
        while pos < num_samples:
            yield x_samples[indices[pos : pos + batch_size]], y_samples[indices[pos : pos + batch_size]]
            pos += batch_size

    def train_iteration(self, batch_size: int, shuffle: bool = True):
        return self._iteration(batch_size, self._num_train, self._x_train, self._y_train, shuffle)

    def val_iteration(self, batch_size: int, shuffle: bool = True):
        return self._iteration(batch_size, self._num_val, self._x_val, self._y_val, shuffle)

    def test_iteration(self, batch_size: int, shuffle: bool = True):
        return self._iteration(batch_size, self._num_train, self._x_test, self._y_test, shuffle)
