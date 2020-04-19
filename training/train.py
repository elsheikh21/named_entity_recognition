import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from training.earlystopping import EarlyStopping


class Trainer:
    def __init__(self, model, loss_function, optimizer, verbose):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, epochs: int = 1, save_to: str = None):
        train_loss = 0.0
        es = EarlyStopping(patience=5)
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            epoch_loss = 0.0
            self.model.train()
            for step, sample in tqdm(enumerate(train_dataset), desc='Training Batches'):
                inputs = sample['inputs']
                labels = sample['outputs']
                self.optimizer.zero_grad()

                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.)  # Gradient Clipping
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            valid_loss = self.evaluate(valid_dataset)

            if self._verbose > 0:
                print(f'Epoch {epoch}: [loss = {avg_epoch_loss:0.4f},  val_loss = {valid_loss:0.4f}]')
            if es.step(valid_loss):
                print(f"Early Stopping callback was activated at epoch num: {epoch}")
                break
        avg_epoch_loss = train_loss / epochs
        if save_to is not None:
            torch.save(self.model, save_to)
        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['inputs']
                labels = sample['outputs']
                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels)
                valid_loss += sample_loss.tolist()
        return valid_loss / len(valid_dataset)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions