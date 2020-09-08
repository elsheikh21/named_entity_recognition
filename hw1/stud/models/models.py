from typing import List

import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    from torchcrf import CRF
except ModuleNotFoundError:
    os.system('pip install pytorch-crf')
    from torchcrf import CRF


class BaselineModel(nn.Module):
    def __init__(self, hparams):
        super(BaselineModel, self).__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = hparams.model_name
        self.word_embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x):
        # [Samples_Num, Seq_Len]
        embeddings = self.word_embedding(x)
        # [Samples_Num, Seq_Len]
        o, _ = self.lstm(embeddings)
        # [Samples_Num, Seq_Len, Tags_Num]
        o = self.dropout(o)
        # [Samples_Num, Seq_Len, Tags_Num]
        logits = self.classifier(o)
        # [Samples_Num, Seq_Len]
        return logits

    def save_checkpoint(self, model_path):
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self._device)
        self.load_state_dict(state_dict)

    def predict_sentences(self, tokens: List[List[str]], words2idx, idx2label):
        self.eval()
        predictions_lst = []
<<<<<<< Updated upstream:models/models.py
        tokens_ = [[_x.lower() for _x in x] for x in tokens]
        for inputs in tqdm(tokens_):
            inputs = torch.LongTensor([words2idx.get(word, 1) for word in inputs]).unsqueeze(0).to(self.device)
=======
        for inputs in tqdm(tokens):
            inputs = torch.LongTensor(
                [words2idx.get(word, 1) for word in inputs]).unsqueeze(0).to(self._device)
>>>>>>> Stashed changes:hw1/stud/models/models.py
            logits = self.predict(inputs)
            predictions = torch.argmax(logits, -1).view(-1)
            valid_indices = predictions != 0
            predictions_ = predictions[valid_indices]
            predictions_lst.append([idx2label.get(tag)
                                    for tag in predictions_[0]])
        return predictions_lst


class CRF_Model(nn.Module):
    def __init__(self, hparams):
        super(CRF_Model, self).__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = hparams.model_name
        self.word_embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0,
                            batch_first=True)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
        self.crf = CRF(hparams.num_classes, batch_first=True)

    def forward(self, x):
        # [Samples_Num, Seq_Len]
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        # [Samples_Num, Seq_Len]
        o, _ = self.lstm(embeddings)
        # [Samples_Num, Seq_Len, Tags_Num]
        o = self.dropout(o)
        # [Samples_Num, Seq_Len, Tags_Num]
        logits = self.classifier(o)
        # [Samples_Num, Seq_Len]
        return logits

    def log_probs(self, x, tags, mask=None):
        emissions = self(x)
        return self.crf(emissions, tags, mask=mask)

    def predict(self, x):
        emissions = self(x)
        return self.crf.decode(emissions)

    def predict_new(self, x, mask=None):
        emissions = self(x)
        return self.crf.decode(emissions, mask=mask)

    def save_checkpoint(self, model_path):
        """
        Saves the model checkpoint
        Args:
            model_path:

        Returns:

        """
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def load_model(self, path):
        """
        Loads the model from a given path, loads it to the available device whether its CUDA or CPU
        Args:
            path:

        Returns:

        """
        state_dict = torch.load(path) if self._device == 'cuda' else torch.load(path,
                                                                                map_location=torch.device(self._device))
        self.load_state_dict(state_dict)

    def encode_tokens(self, tokens, word2idx):
        """
        Helper method during prediction
        Encodes the tokens passed during prediction time, fetches word idx from word2idx
        Args:
            tokens:
            word2idx:

        Returns:

        """
        data = []
        for sentence in tokens:
            paragraph = []
            for i in sentence:
                paragraph.append(word2idx.get(i, 1))
            paragraph = torch.LongTensor(paragraph).to(self._device)
            data.append(paragraph)
        return pad_sequence(data, batch_first=True, padding_value=0)


class BiLSTM_CRF_POS_Model(nn.Module):
    """
    Stacked BiLSM CRF model with POS Embeddings
    """

    def __init__(self, hparams):
        super(BiLSTM_CRF_POS_Model, self).__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = hparams.model_name
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        self.pos_embedding = nn.Embedding(hparams.pos_vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)
        if hparams.pos_embeddings is not None:
            print("initializing pos embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.word_dropout = nn.Dropout(hparams.dropout)
        self.pos_dropout = nn.Dropout(hparams.dropout)
        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0,
                            batch_first=True)

        self.pos_lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                                bidirectional=hparams.bidirectional,
                                num_layers=hparams.num_layers,
                                dropout=hparams.dropout if hparams.num_layers > 1 else 0,
                                batch_first=True)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim * 2, hparams.num_classes)
        self.crf = CRF(hparams.num_classes, batch_first=True)

    def forward(self, x, pos):
        word_embeddings = self.word_embedding(x)
        word_embeddings = self.word_dropout(word_embeddings)
        pos_embeddings = self.pos_embedding(pos)
        pos_embeddings = self.pos_dropout(pos_embeddings)
        word_o, _ = self.lstm(word_embeddings)
        pos_o, _ = self.pos_lstm(pos_embeddings)
        o = torch.cat((word_o, pos_o), dim=-1)
        o = self.dropout(o)
        logits = self.classifier(o)
        return logits

    def log_probs(self, x, tags, mask, pos):
        emissions = self(x, pos)
        return self.crf(emissions, tags, mask=mask)

    def predict(self, x, mask, pos):
        self.eval()
<<<<<<< Updated upstream:models/models.py
        predictions_lst = []
        tokens_ = [[_x.lower() for _x in x] for x in tokens]
        for inputs in tqdm(tokens_):
            inputs = torch.LongTensor([words2idx.get(word, 1) for word in inputs]).unsqueeze(0).to(self.device)
            predictions = self.predict(inputs)
            predictions_lst.append([idx2label.get(tag) for tag in predictions[0]])
        return predictions_lst
=======
        with torch.no_grad():
            emissions = self(x, pos)
            return self.crf.decode(emissions, mask=mask)

    def save_checkpoint(self, dir_path):
        torch.save(self, f"{dir_path}.pt")
        torch.save(self.state_dict(), f"{dir_path}.pth")

    def load_model(self, path):
        state_dict = torch.load(path) if self._device == 'cuda' else torch.load(path, map_location=self._device)
        self.load_state_dict(state_dict)

    def print_summary(self, show_weights=False, show_parameters=False):
        """
        Summarizes torch model by showing trainable parameters and weights.
        """
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = self.print_summary()
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr += ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        print(f'========== {self.name} Model Summary ==========')
        print(tmpstr)
        num_params = sum(p.numel()
                         for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params:,}")
        print('==================================================')
>>>>>>> Stashed changes:hw1/stud/models/models.py
