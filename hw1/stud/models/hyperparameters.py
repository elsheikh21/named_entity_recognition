class HyperParameters():
    def __init__(self, model_name_, vocab, label_vocab, embeddings_, batch_size_):
        """Defines the model hyperparams"""
        self.model_name = model_name_
        self.vocab_size = len(vocab)
        self.hidden_dim = 512
        self.embedding_dim = 300
        self.num_classes = len(label_vocab)
        self.bidirectional = True
        self.num_layers = 2
        self.dropout = 0.4
        self.embeddings = embeddings_
        self.batch_size = batch_size_

    def _print_info(self):
        """
        prints summary of model's hyperparameters
        """
        print("========== Hyperparameters ==========",
              f"Name: {self.model_name.replace('_', ' ')}",
              f"Vocab Size: {self.vocab_size}",
              f"Tags Size: {self.num_classes}",
              f"Embeddings Dim: {self.embedding_dim}",
              f"Hidden Size: {self.hidden_dim}",
              f"BiLSTM: {self.bidirectional}",
              f"Layers Num: {self.num_layers}",
              f"Dropout: {self.dropout}",
              f"Pretrained_embeddings: {False if self.embeddings is None else True}",
              f"PoS Pretrained_embeddings: {False if self.pos_embeddings is None else True}",
              f"Batch Size: {self.batch_size}", sep='\n')