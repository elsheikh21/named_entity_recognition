import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, hparams):
        super(BaselineModel, self).__init__()
        self.name = 'BiLSTM'
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
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
