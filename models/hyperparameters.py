class HyperParameters():
    def __init__(self, model_name_, vocab, label_vocab, embeddings_, batch_size_):
        self.model_name = model_name_
        self.vocab_size = len(vocab)
        self.hidden_dim = 256
        self.embedding_dim = 300
        self.num_classes = len(label_vocab)
        self.bidirectional = False
        self.num_layers = 1
        self.dropout = 0.4
        self.embeddings = embeddings_
        self.batch_size = batch_size_
