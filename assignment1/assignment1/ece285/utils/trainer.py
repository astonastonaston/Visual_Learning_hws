import numpy as np
from .dataset import DataLoader
from .optimizer import Optimizer
from ece285.layers.sequential import Sequential
from ece285.layers.base_layer import BaseLayer
from ece285.utils.evaluation import get_classification_accuracy


class Trainer(object):
    def __init__(
        self,
        dataset: DataLoader,
        optimizer: Optimizer,
        model: Sequential,
        loss_func: BaseLayer,
        epoch: int,
        batch_size: int,
        evaluate_batch_size: int = None,
        validate_interval: int = 1,
    ):
        self.dataset = dataset
        self.optimizer = optimizer
        self.model = model
        self.loss_func = loss_func
        self.epoch = epoch
        self.batch_size = batch_size
        self.evaluate_batch_size = evaluate_batch_size if not (evaluate_batch_size is None) else batch_size
        self.validate_interval = validate_interval
        self.logs = []

    def validate(self):
        predictions = []
        for batch_x, _ in self.dataset.val_iteration(self.batch_size, shuffle=False):
            predictions.append(self.model.predict(batch_x))
        predictions = np.concatenate(predictions)
        return get_classification_accuracy(predictions, self.dataset._y_val)

    def train(self):
        # self.logs = []
        training_loss = []
        eval_accuracies = []
        for epoch in range(self.epoch):
            epoch_loss = []
            for batch_x, batch_y in self.dataset.train_iteration(self.batch_size):
                output_x = self.model(batch_x)
                loss = self.loss_func.forward(output_x, batch_y)

                self.optimizer.zero_grad()
                self.model.backward(self.loss_func.backward())
                self.optimizer.step()
                # self.logs.append(current_log)
                epoch_loss.append(loss)
            print("Epoch {} Average Loss: {:3f}".format(epoch, np.mean(epoch_loss)))
            training_loss.append(np.mean(epoch_loss))

            if epoch % self.validate_interval == 0:
                eval_accuracy = self.validate()
                eval_accuracies.append(eval_accuracy)
                print("Validate Acc: {:.3f} at epoch {}".format(eval_accuracy, epoch))
        return training_loss, eval_accuracies
