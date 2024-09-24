from time import time
import logging


class callback:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.training_logs = []

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting...")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(
            f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds: Loss: {logs['loss']:.4f},"
            f" Accuracy: {logs['accuracy']:.4f}, Validation Loss: {logs['valLoss']:.4f},"
            f" Validation Accuracy: {logs['valAccuracy']:.4f} "
        )
        logs["epoch_time"] = elapsed_time
        self.training_logs.append(logs)

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            logging.info(
                f"Batch {batch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}"
            )


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def on_epoch_end(self, epoch, logs=None, val=False):
        if val:
            current_loss = logs["valLoss"]
        else:
            current_loss = logs["loss"]
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered.")
                return True
        return False
