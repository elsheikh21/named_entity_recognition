class HyperParameters:
    def __init__(self, model_name_, vocab, label_vocab, embeddings_, batch_size_):
        self.model_name = model_name_
        self.vocab_size = len(vocab) if type(vocab) is dict else vocab
        self.hidden_dim = 512
        self.embedding_dim = 300
        self.num_classes = len(label_vocab) if type(label_vocab) is dict else label_vocab
        self.bidirectional = True
        self.num_layers = 2
        self.dropout = 0.4
        self.embeddings = embeddings_
        self.batch_size = batch_size_

    def _print_info(self):
        print(
            f'Name: {self.model_name}\nVocab Size: {self.vocab_size}\nTags Size: {self.num_classes}\nEmbeddings Dim: {self.embedding_dim}\nHidden Size: {self.hidden_dim}\nBiLSTM: {self.bidirectional}\nLayers Num: {self.num_layers}\nPretrained_embeddings: {False if self.embeddings is None else True}\nBatch Size: {self.batch_size}')
