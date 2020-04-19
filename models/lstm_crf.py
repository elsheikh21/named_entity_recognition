import torch
import torch.nn as nn
from tqdm.auto import tqdm

from models.crf import CRF


class BaselineModel(nn.Module):
    def __init__(self, hyperparams):
        super(BaselineModel, self).__init__()
        self.name = hyperparams.model_name
        self.hidden_dim = hyperparams.hidden_dim
        self.word_embedding = nn.Embedding(hyperparams.vocab_size,
                                           hyperparams.embedding_dim)
        if hyperparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hyperparams.embeddings)

        self.lstm = nn.LSTM(hyperparams.embedding_dim, hyperparams.hidden_dim,
                            bidirectional=hyperparams.bidirectional,
                            num_layers=hyperparams.num_layers,
                            dropout=hyperparams.dropout if hyperparams.num_layers > 1 else 0,
                            batch_first=True)

        lstm_output_dim = hyperparams.hidden_dim if hyperparams.bidirectional is False else hyperparams.hidden_dim * 2
        self.hidden2tag = nn.Linear(lstm_output_dim,
                                    hyperparams.num_classes)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_dim // 2), torch.randn(1, batch_size, self.hidden_dim // 2),)

    def forward(self, batch_of_sentences):
        self.hidden = self.init_hidden(batch_of_sentences.shape[0])
        x = self.word_embedding(batch_of_sentences)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.hidden2tag(x)
        return x


class BiLSTM_CRF(nn.Module):
    def __init__(self, hyperparams):
        super(BiLSTM_CRF, self).__init__()
        self.lstm = BaselineModel(hyperparams)
        self.crf = CRF(hyperparams.num_classes, bos_tag_id=2, eos_tag_id=3, pad_tag_id=0, batch_first=True)
        self.hidden = None

    def forward(self, x, mask=None):
        emissions = self.lstm(x)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path

    def loss(self, x, y, mask=None):
        emissions = self.lstm(x)
        nll = self.crf(emissions, y, mask=mask)
        return nll

    def train_(self, optimizer, epochs, train_dataset):
        train_loss = 0.0
        for _ in tqdm(range(epochs), desc='Training'):
            epoch_loss = 0.
            self.train()

            for _, samples in tqdm(enumerate(train_dataset), desc='Batches of data'):
                inputs, labels = samples['inputs'], samples['outputs']
                optimizer.zero_grad()

                loss = self.loss(inputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.tolist()
            train_loss += epoch_loss / len(train_dataset)
            print(f"Train loss: {epoch_loss / len(train_dataset)}")
        return train_loss

    def predict(self, data_x, idx2label):
        data_x = torch.LongTensor(data_x[:100])
        with torch.no_grad():
            scores, seqs = self(data_x)
            for score, seq in zip(scores, seqs):
                str_seq = "".join([idx2label.get(x) for x in seq if x != 0])
                # print(f'{score.item()}.2f: {str_seq}')
                print(f'{str_seq}')
