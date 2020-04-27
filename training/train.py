import time

import torch
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from callbacks import ProgressBar
from training.earlystopping import EarlyStopping
try:
    from torchcrf import CRF
except ModuleNotFoundError:
    import os
    os.system('pip install pytorch-crf')
    from torchcrf import CRF


class F1_score(object):
    def __init__(self, num_classes=None):
        self.labels = None
        if num_classes:
            self.labels = [i for i in range(num_classes)]

    def __call__(self, best_path, target):
        best_path = torch.argmax(best_path, -1)
        y_pred = best_path.contiguous().view(1, -1).squeeze().cpu()
        y_true = target.contiguous().view(1, -1).squeeze().cpu()
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
        return f1_score(y_true, y_pred, labels=self.labels, average="macro")


class Trainer:
    def __init__(self, model, loss_function, optimizer, batch_num, num_classes, verbose):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose
        self.evaluator = F1_score(num_classes)
        self.progressbar = ProgressBar(n_batch=batch_num, loss_name='loss')

    def train(self, train_dataset, valid_dataset, epochs=1, save_to=None):
        train_loss = 0.0
        es = EarlyStopping(patience=5)
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            epoch_loss = 0.0
            self.model.train()
            for step, sample in enumerate(train_dataset):
                start = time.time()
                inputs = sample['inputs']
                labels = sample['outputs']
                self.optimizer.zero_grad()

                predictions_ = self.model(inputs)
                predictions = predictions_.view(-1, predictions_.shape[-1])
                labels = labels.view(-1)

                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.)  # Gradient Clipping
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
                f1_score_ = self.evaluator(predictions_, labels)
                self.progressbar.step(batch_idx=step,
                                      loss=sample_loss.item(),
                                      f1=f1_score_.item(),
                                      use_time=time.time() - start)

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            valid_loss = self.evaluate(valid_dataset)

            if self._verbose > 0:
                print(
                    f'Epoch {epoch}: [loss = {avg_epoch_loss:0.4f},  val_loss = {valid_loss:0.4f}]')
            if es.step(valid_loss):
                print(
                    f"Early Stopping callback was activated at epoch num: {epoch}")
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


class CRF_Trainer(object):
    def __init__(self, model, loss_function, optimizer, label_vocab, writer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = label_vocab
        self.label_vocab = label_vocab
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer = writer

    def train(self, train_dataset, valid_dataset, epochs=1):
        es = EarlyStopping(patience=10)
        scheduler = ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
        train_loss = 0.0
        epoch, step = 0, 0
        for epoch in tqdm(range(epochs), desc=f'Training Epoch # {epoch + 1} / {epochs}'):
            epoch_loss = 0.0
            self.model.train()
            for step, sample in tqdm(enumerate(train_dataset), desc=f'Train on batch # {step + 1}'):
                inputs, labels = sample['inputs'], sample['outputs']
                mask = (inputs != 0).to(self._device, dtype=torch.uint8)
                self.optimizer.zero_grad()
                # Pass the inputs directly, log_probabilities already calls forward
                sample_loss = -self.model.log_probs(inputs, labels, mask)
                sample_loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.)  # Gradient Clipping
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            valid_loss, valid_acc = self.evaluate(valid_dataset)
            scheduler.step(valid_loss)
            epoch_summary = f'Epoch #: {epoch + 1} [loss: {avg_epoch_loss:0.4f}, val_loss: {valid_loss:0.4f}]'
            print(epoch_summary)

            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', epoch_loss)
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            if es.step(valid_loss):
                print(f"Early Stopping activated on epoch #: {epoch}")
                break

        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        # set dropout to 0!! Needed when we are in inference mode.
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Computing Val Loss'):
                inputs = sample['inputs']
                labels = sample['outputs']
                mask = (inputs != 0).to(self._device, dtype=torch.uint8)
                sample_loss = -self.model.log_probs(inputs, labels, mask).sum()
                valid_loss += sample_loss.tolist()

                # Compute accuracy
                predictions = self.model.predict(inputs)
                # argmax is a list, must convert to tensor
                argmax = labels.new_tensor(predictions)
                acc = (labels == argmax.squeeze()).float().mean()

        return valid_loss / len(valid_dataset), acc / len(valid_dataset)