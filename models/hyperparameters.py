class HyperParameters:
    def __init__(self, batch_size, vocab_size, tags_num):
        self.hidden_dim = 256
        self.embedding_dim = 300
        self.bidirectional = True
        self.num_layers = 2
        self.dropout = 0.2
        self.embeddings = None
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_classes = tags_num
