import os

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from training.earlystopping import EarlyStopping
from utilities import save_checkpoint
from utilities import plot_history

try:
    from livelossplot import PlotLosses
except ModuleNotFoundError:
    os.system('pip install livelossplot')
    from livelossplot import PlotLosses


class Trainer:
    def __init__(self, model: nn.Module, loss_function, optimizer, label_vocab, log_steps: int = 10_000, log_level: int = 2):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.label_vocab = label_vocab
        self.writer = SummaryWriter()
        self.plot_losses_lst = {'loss': [], 'val_loss': []}

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, epochs: int = 1, save_to: str = None):
        best_val_loss = torch.FloatTensor([1e4])
        assert epochs > 1 and isinstance(epochs, int)
        if self.log_level > 0:
            print('Training ...')
        train_loss = 0.0

        es = EarlyStopping(patience=5)
        scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, mode='min', patience=3, min_lr=1e-12)
        liveloss = PlotLosses()

        epoch, epoch_, step = 0, 0, 0
        for epoch in tqdm(range(epochs), desc=f'Training Epoch # {epoch + 1} / {epochs}'):
            logs = {}
            if self.log_level > 0:
                print(f' Epoch {epoch + 1:02d}')

            epoch_loss = 0.0
            self.model.train()
            # (inputs, labels, seq_lengths)
            for step, samples in tqdm(enumerate(train_dataset), desc=f'Training on batches of {step + 1}'):
                self.optimizer.zero_grad()
                inputs, labels = samples['inputs'].squeeze(1), samples['outputs'].squeeze(1)
                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()

                clip_grad_norm_(self.model.parameters(), 1.)  # Gradient Clipping

                self.optimizer.step()

                epoch_loss += sample_loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print(f'\t[Epoch # {epoch:2d} @ step #{step}] Curr avg loss = {epoch_loss / (step + 1):0.4f}')

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            epoch_ += 1

            valid_loss, _ = self.evaluate(valid_dataset)

            self.plot_losses_lst['loss'].append(avg_epoch_loss)
            self.plot_losses_lst['val_loss'].append(avg_epoch_loss)

            logs['loss'] = avg_epoch_loss
            logs['val_loss'] = valid_loss
            # Get bool not ByteTensor
            is_best = bool(torch.FloatTensor([valid_loss]) > best_val_loss)
            # Get greater Tensor to keep track best validation loss
            best_valid_loss = torch.FloatTensor([min(valid_loss, best_val_loss)])
            # Save checkpoint if is a new best
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_val_loss': best_valid_loss
            }, is_best)

            self.writer.add_scalar('Loss', epoch_loss, epoch_)
            self.writer.add_scalar('Loss', valid_loss, epoch_)

            if self.log_level > 0:
                print(f'\tEpoch #: {epoch:2d} [loss: {avg_epoch_loss:0.4f}, val_loss: {valid_loss:0.4f}]')

            scheduler.step(valid_loss)

            if es.step(valid_loss):
                print(f"Early Stopping callback was activated at epoch num: {epoch}")
                break

            liveloss.update(logs)
            liveloss.send()
        if self.log_level > 0:
            print('... Done!')

        avg_epoch_loss = train_loss / epochs
        self.writer.close()
        plot_history(self.plot_losses_lst)

        if save_to is not None:
            torch.save(self.model.state_dict(), save_to)

        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        # set dropout to 0!! Needed when we are in inference mode.
        self.model.eval()
        with torch.no_grad():
            # for step, (inputs, labels, seq_lengths) in enumerate(valid_dataset):
            for step, samples in enumerate(valid_dataset):
                inputs = samples['inputs'].squeeze(1)
                labels = samples['outputs'].squeeze(1)
                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels)
                valid_loss += sample_loss.tolist()

                # Compute accuracy
                _, argmax = torch.max(predictions, 1)
                acc = (labels == argmax.squeeze()).float().mean()

        return valid_loss / len(valid_dataset), acc / len(valid_dataset)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions

