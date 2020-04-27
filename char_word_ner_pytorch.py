import io
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.autograd as autograd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def configure_workspace(SEED=1873337):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


class TSVDatasetParser(Dataset):
    def __init__(self, file_path, verbose=False, max_len=150, is_crf=False):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._file_path = file_path
        self.max_len = max_len
        self.max_char_len = 150  # based on Distribution of Word Lengths (histogram)
        self._verbose = verbose

        self.data_x, self.data_y = self.parse_dataset()
        vocab_dictionaries = self.create_vocabulary(is_crf)
        self.word2idx, self.idx2word, self.char2idx, self.idx2char = vocab_dictionaries

        self.labels2idx = {'<PAD>': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'O': 4}
        self.idx2label = {key: val for key, val in enumerate(self.labels2idx)}
        self.encoded_data = []

        if verbose:
            print(f"Tensors are created on: {self._device}")
            print(f"Tags Size: {len(self.labels2idx)}\nVocab Size: {len(self.word2idx)}")
            print(f"Data_x & Data_y len: {len(self.data_x)}.")

    def parse_dataset(self):
        with open(self._file_path, encoding='utf-8', mode='r') as file_:
            lines = file_.read().splitlines()

        data_x, data_y = [], []
        sentence_tags = []

        for line in tqdm(lines, desc='Parsing Data'):
            if line == '':
                if sentence_tags:
                    data_y.append(sentence_tags[:self.max_len])
                    sentence_tags = []
            elif line[0] == '#':
                sentence = line.replace('# ', '')
                data_x.append([word.lower() for word in (sentence.split()[:self.max_len])])
                if sentence_tags:
                    data_y.append(sentence_tags[:self.max_len])
                    sentence_tags = []
            elif line[0].isdigit():
                sentence_tags.append(line.split('\t')[-1])
        return data_x[:300], data_y[:300]

    def create_vocabulary(self, is_crf):
        all_words = [item for sublist in self.data_x for item in sublist]
        unigrams = list(set(all_words))
        en_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        char_unigrams = set(''.join(unigrams))
        char_unigrams = char_unigrams.union(en_chars)
        char_unigrams = sorted(list(char_unigrams))
        if is_crf:
            char2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
            word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
            start_ = 4
        else:
            char2idx = {'<PAD>': 0, '<UNK>': 1}
            word2idx = {'<PAD>': 0, '<UNK>': 1}
            start_ = 2
        word2idx.update(
            {val: key for key, val in enumerate(unigrams, start=start_)})
        idx2word = {key: val for key, val in enumerate(word2idx)}
        char2idx.update(
            {val: key for key, val in enumerate(char_unigrams, start=start_)})
        idx2char = {key: val for key, val in enumerate(char2idx)}
        return word2idx, idx2word, char2idx, idx2char

    def encode_dataset(self, word2idx, labels2idx):
        data_x_stoi, data_y_stoi = [], []
        for sentence, labels in tqdm(zip(self.data_x, self.data_y),
                                     desc=f'Building & allocating tensors to {self._device}'):
            data_x_stoi.append(torch.LongTensor([word2idx.get(word, 1) for word in sentence]).to(self._device))
            data_y_stoi.append(torch.LongTensor([labels2idx.get(tag) for tag in labels]).to(self._device))

        data_x_stoi = pad_sequence(data_x_stoi, batch_first=True, padding_value=labels2idx.get('<PAD>'))
        data_y_stoi = pad_sequence(data_y_stoi, batch_first=True, padding_value=labels2idx.get('<PAD>'))

        assert data_x_stoi.shape == data_y_stoi.shape, f"pad_x shape: {data_x_stoi.shape} does not match pad_y shape: {data_y_stoi.shape}"

        if self._verbose:
            print(f"Data & Labels shape: {data_x_stoi.shape} => (Samples Num, max_len)")

        for i in tqdm(range(len(data_x_stoi)), desc="Indexing dataset"):
            self.encoded_data.append({'inputs': data_x_stoi[i], 'outputs': data_y_stoi[i]})

    @staticmethod
    def create_word_tensors(word2idx, char2idx, embedding_dim=300):
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        word_tensors = {0: torch.zeros(embedding_dim), 1: torch.zeros(embedding_dim)}
        for word, idx in word2idx.items():
            if not (word == '<PAD>' or word == '<UNK>'):
                word_tensors[idx] = torch.LongTensor([char2idx[char] for char in word]).to(_device)
        return word_tensors

    @staticmethod
    def decode_predictions(logits, idx2label):
        max_indices = torch.argmax(logits, -1).tolist()  # shape = (batch_size, max_len)
        predictions = list()
        for indices in tqdm(max_indices, desc='Decoding Predictions'):
            predictions.append([idx2label.get(i) for i in indices])
        return predictions

    def get_element(self, idx):
        return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Dataset is not indexed yet.\
                                To fetch raw elements, use get_element(idx)")
        return self.encoded_data[idx]

    @property
    def get_device(self):
        return self._device


def load_pretrained_embeddings(fname, word2idx, embeddings_size, is_crf=False):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin, desc=f'Reading data from {fname}'):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=np.float)

    pretrained_embeddings = torch.randn(len(word2idx), embeddings_size)
    initialised = 0
    for idx, word in enumerate(data):
        if word in word2idx:
            initialised += 1
            vector_ = torch.from_numpy(data[word])
            pretrained_embeddings[word2idx.get(word)] = vector_

    pretrained_embeddings[word2idx["<PAD>"]] = torch.zeros(embeddings_size)
    pretrained_embeddings[word2idx["<UNK>"]] = torch.zeros(embeddings_size)
    if is_crf:
        pretrained_embeddings[word2idx["<BOS>"]] = torch.zeros(embeddings_size)
        pretrained_embeddings[word2idx["<EOS>"]] = torch.zeros(embeddings_size)
    print(f'Loaded {initialised} vectors and instantiated random embeddings for {len(word2idx) - initialised}')
    return pretrained_embeddings

# !wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
# !cp '/content/drive/My Drive/HW1_NLP/wiki.en.vec' 'wiki.en.vec'
# pretrained_embeddings_ = load_pretrained_embeddings('wiki.en.vec', train_dataset.word2idx, 300, is_crf=True)


class DualTagger(nn.Module):
    def __init__(self, hparams):
        super(DualTagger, self).__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)

        self.char_embedding = nn.Embedding(hparams.char_vocab_size, hparams.char_embedding_dim)

        self.char_lstm = nn.LSTM(hparams.char_embedding_dim, hparams.char_hidden_dim)

        self.lstm = nn.LSTM(hparams.embedding_dim + hparams.char_hidden_dim,
                            hparams.hidden_dim)

        # self.dropout = nn.Dropout(hparams.dropout)
        self.hidden2tag = nn.Linear(hparams.hidden_dim, hparams.num_classes)

        self.hidden_char = (autograd.Variable(torch.zeros(1, 1, hparams.char_hidden_dim)).to(self._device),
                            autograd.Variable(torch.zeros(1, 1, hparams.char_hidden_dim)).to(self._device))
        self.hidden_words = (autograd.Variable(torch.zeros(1, 1, hparams.hidden_dim)).to(self._device),
                             autograd.Variable(torch.zeros(1, 1, hparams.hidden_dim)).to(self._device))

    def forward(self, sentence, word_tensors_):
        logits_ = None
        for idx, word_idx in enumerate(sentence):
            word_chars_tensor = word_tensors_.get(int(word_idx))
            char_embeds = self.char_embedding(word_chars_tensor)
            lstm_char_out, self.hidden_char = self.char_lstm(char_embeds.view(len(word_chars_tensor), 1, -1), self.hidden_char)
            word_embed = self.word_embedding(word_idx)
            embeds_cat = torch.cat((word_embed.view(1, 1, -1), lstm_char_out[-1].view(1, 1, -1)), dim=2)
            lstm_out, self.hidden_words = self.lstm(embeds_cat, self.hidden_words)
            logits = self.hidden2tag(lstm_out.view(1, -1))
            if idx == 0:
                logits_ = logits
            else:
                logits_ = torch.cat((logits_, logits), dim=0)
        return logits_


class HyperParameters():
    def __init__(self, model_name_, vocab, char_vocab, label_vocab, embeddings_, batch_size_):
        self.model_name = model_name_

        self.char_vocab_size, self.vocab_size = len(char_vocab), len(vocab)
        self.hidden_dim, self.char_hidden_dim = 256, 256
        self.embedding_dim, self.char_embedding_dim = 300, 300

        self.num_classes = len(label_vocab)
        self.bidirectional = False
        self.num_layers = 1
        self.dropout = 0.4
        self.embeddings = embeddings_
        self.batch_size = batch_size_


class Trainer(object):
    def __init__(self, model, loss_function, optimizer, verbose):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, epochs: int = 1, word_tensors=None):
        train_loss = 0.0
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            epoch_loss = 0.0
            self.model.train()
            for step, sample in tqdm(enumerate(train_dataset), desc='Training Batches'):
                inputs = sample['inputs']
                labels = sample['outputs']

                self.optimizer.zero_grad()

                for input_, tags in zip(tqdm(inputs), labels):
                    input_ = input_[input_.nonzero()]
                    predictions = self.model(input_, word_tensors)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    tags = tags[tags.nonzero()].view(-1)

                    sample_loss = self.loss_function(predictions, tags)
                    sample_loss.backward(retain_graph=True)
                    train_loss += sample_loss.item()

                clip_grad_norm_(self.model.parameters(), 5.)  # Gradient Clipping
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            valid_loss = self.evaluate(valid_dataset, word_tensors)

            if self._verbose > 0:
                print(f'Epoch {epoch}: [loss = {avg_epoch_loss:0.4f},  val_loss = {valid_loss:0.4f}]')
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss

    def evaluate(self, valid_dataset, word_tensors):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['inputs']
                labels = sample['outputs']
                for input_, tags in zip(tqdm(inputs), labels):
                    input_ = input_[input_.nonzero()]
                    predictions = self.model(input_, word_tensors)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    tags = tags[tags.nonzero()].view(-1)
                    sample_loss = self.loss_function(predictions, tags)
                    valid_loss += sample_loss.tolist()
        return valid_loss / len(valid_dataset)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions


def compute_precision(model, dev_dataset, is_crf):
    all_predictions = list()
    all_labels = list()
    # for step, (inputs, labels, seq_lengths, perm_idx) in enumerate(self.dev_dataset):
    for step, samples in tqdm(enumerate(dev_dataset), desc="Predicting batches of data"):
        inputs, labels = samples['inputs'], samples['outputs']
        if is_crf:
            _, predictions = model(inputs)
            predictions = torch.LongTensor(predictions).to('cuda').view(-1)
        else:
            predictions = model(inputs)
            predictions = torch.argmax(predictions, -1).view(-1)
        labels = labels.view(-1)
        valid_indices = labels != 0

        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]

        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())
    # global precision. Does take class imbalance into account.
    micro_precision_recall_fscore = precision_recall_fscore_support(all_labels, all_predictions,
                                                                    average="micro")

    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision_recall_fscore = precision_recall_fscore_support(
        all_labels, all_predictions, average="macro")

    per_class_precision = precision_score(all_labels, all_predictions,
                                          average=None)

    return {"macro_precision_recall_fscore": macro_precision_recall_fscore,
            "micro_precision_recall_fscore": micro_precision_recall_fscore,
            "per_class_precision": per_class_precision,
            "confusion_matrix": confusion_matrix(all_labels, all_predictions)}


def pprint_confusion_matrix(conf_matrix):
    df_cm = pd.DataFrame(conf_matrix)
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()


if __name__ == "__main__":
    """LOAD DATASET"""
    configure_workspace()
    file_path = os.path.join(os.getcwd(), 'Data', 'train.tsv')
    train_dataset = TSVDatasetParser(file_path)
    train_dataset.encode_dataset(train_dataset.word2idx, train_dataset.labels2idx)

    dev_file_path = file_path.replace('train.tsv', 'dev.tsv')
    dev_dataset = TSVDatasetParser(dev_file_path)
    dev_dataset.encode_dataset(train_dataset.word2idx, train_dataset.labels2idx)

    test_file_path = file_path.replace('train.tsv', 'test.tsv')
    test_dataset = TSVDatasetParser(test_file_path)
    test_dataset.encode_dataset(train_dataset.word2idx, train_dataset.labels2idx)

    """HyperParameters"""
    hp = HyperParameters('LSTM', train_dataset.word2idx, train_dataset.char2idx, train_dataset.labels2idx, None, 128)

    """Data Loaders"""
    batch_size_ = hp.batch_size
    train_dataset_ = DataLoader(train_dataset, batch_size=batch_size_)
    valid_dataset_ = DataLoader(dev_dataset, batch_size=batch_size_)
    test_dataset_ = DataLoader(test_dataset, batch_size=batch_size_)

    word_tensors = TSVDatasetParser.create_word_tensors(train_dataset.word2idx, train_dataset.char2idx)

    """Model Building"""
    model = DualTagger(hp).to(train_dataset._device)
    print(f"Model Summary:\n{model}")

    """Model Training"""
    trainer = Trainer(model, CrossEntropyLoss(ignore_index=train_dataset.labels2idx.get('<PAD>')),
                      Adam(model.parameters()), verbose=True)

    trainer.train(train_dataset_, valid_dataset_, 1, word_tensors)

    """Model Performance"""
    test_set_loss = trainer.evaluate(test_dataset_)
    print(f"test_loss: {test_set_loss:0.6f}")

    precisions = compute_precision(model, test_dataset_, train_dataset.labels2idx)

    per_class_precision = precisions["per_class_precision"]
    print(f"Micro Precision: {precisions['micro_precision']:0.6f}")
    print(f"Macro Precision: {precisions['macro_precision']:0.6f}")

    print("Per class Precision:")
    for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
        label = train_dataset.idx2label.get(idx_class)
        print(f"{label}: {precision:0.6f}")

    pprint_confusion_matrix(precisions['confusion_matrix'])

    test_x = ['Barack Obama is the first black president of the United States of America']
    print(test_x)
    test_x_stoi = [train_dataset.word2idx.get(word, 1) for word in test_x[0].split()]
    print(test_x_stoi)

    test_tensor_x = torch.LongTensor([test_x_stoi]).to(train_dataset._device)
    print(test_tensor_x)
    print(test_tensor_x.shape)
    # logits = model(test_tensor_x)
    _, predictions = model(test_tensor_x)
    predictions = torch.LongTensor(predictions).to('cuda').view(-1)

    # predictions = torch.argmax(logits, -1)
    preds_ = predictions.tolist()
    indexed_pred = [train_dataset.idx2label.get(pred) for pred in preds_]
    print(predictions)
    print(indexed_pred)

    # print(test_x[0].split())
    # print(indexed_pred)
