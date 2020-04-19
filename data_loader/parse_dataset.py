import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TSVDatasetParser(Dataset):
    def __init__(self, file_path, verbose, max_len=80, is_crf=False):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._file_path = file_path
        self.max_len = max_len
        self._verbose = verbose

        self.data_x, self.data_y = self.parse_dataset()
        self.word2idx, self.idx2word = self.create_vocabulary(is_crf)

        self.labels2idx = {'<PAD>': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'O': 4}
        self.idx2label = {key: val for key, val in enumerate(self.labels2idx)}
        self.encoded_data = []

        if verbose:
            print(f"Tensors are created on: {self._device}")
            print(f"Tagset length: {len(self.labels2idx)}\nVocab Size: {len(self.word2idx)}")
            print(f"Data_x & Data_y len: {len(self.data_x)}.")

    def parse_dataset(self):
        with open(self._file_path, encoding='utf-8', mode='r') as file_:
            lines = file_.read().splitlines()

        data_x, data_y = [], []
        sentence_tags = []

        for line in tqdm(lines, desc='Parsing Data'):
            sentence_x = []
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

        return data_x, data_y

    def create_vocabulary(self, is_crf):
        all_words = [item for sublist in self.data_x for item in sublist]
        unigrams = list(set(all_words))
        if is_crf:
            word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
            start_ = 4
        else:
            word2idx = {'<PAD>': 0, '<UNK>': 1}
            start_ = 2
        word2idx.update(
            {val: key for key, val in enumerate(unigrams, start=start_)})
        idx2word = {key: val for key, val in enumerate(word2idx)}
        return word2idx, idx2word

    def encode_dataset(self, word2idx, labels2idx):
        data_x_stoi = []
        for sentence in tqdm(self.data_x,
                             desc='Converting data_x to tensors and allocating them'):
            data_x_stoi.append(torch.LongTensor(
                [word2idx.get(word, 1) for word in sentence]).to(self._device))

        data_y_stoi = []
        for labels in tqdm(self.data_y,
                           desc='Converting data_x to tensors and allocating them'):
            data_y_stoi.append(torch.LongTensor(
                [labels2idx.get(tag) for tag in labels]).to(self._device))

        pad_x = pad_sequence(data_x_stoi, batch_first=True,
                             padding_value=labels2idx.get('<PAD>'))
        pad_y = pad_sequence(data_y_stoi, batch_first=True,
                             padding_value=labels2idx.get('<PAD>'))
        assert pad_x.shape == pad_y.shape, f"pad_x shape: {pad_x.shape} does not match pad_y shape: {pad_y.shape}"

        if self._verbose:
            print(f"Data & Labels shape: {pad_x.shape} => (Samples Num, max_len)")

        self.encoded_list = []
        for i in tqdm(range(len(pad_x)), desc="Indexing dataset"):
            self.encoded_data.append({'inputs': pad_x[i], 'outputs': pad_y[i]})

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
