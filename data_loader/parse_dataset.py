import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TSVDatasetParser(Dataset):
    def __init__(self, file_path, verbose=False, max_len=80, is_crf=False):
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
        return data_x, data_y

    def create_vocabulary(self, is_crf):
        all_words = [item for sublist in self.data_x for item in sublist]
        unigrams = list(set(all_words))
        char_unigrams = sorted(list(set((''.join(unigrams)))))
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

    def encode_dataset(self, word2idx, labels2idx, char2idx):
        data_x_stoi, word_x_stoi, data_y_stoi = [], [], []
        for sentence, labels in tqdm(zip(self.data_x, self.data_y), desc=f'Building & allocating tensors to {self._device}'):
            # Words 2 index to create the word_inputs for model
            data_x_stoi.append(torch.LongTensor([word2idx.get(word, 1) for word in sentence]).to(self._device))
            # Characters 2 index to create the char_inputs for model
            word_x_stoi.append(torch.LongTensor([char2idx.get(char, 1) for char in ''.join(sentence)[:self.max_char_len]]).to(self._device))
            # Tags 2 index to create the tag_inputs for model
            data_y_stoi.append(torch.LongTensor([labels2idx.get(tag) for tag in labels]).to(self._device))

        pad_x = pad_sequence(data_x_stoi, batch_first=True,
                             padding_value=labels2idx.get('<PAD>'))
        char_pad_x = pad_sequence(word_x_stoi, batch_first=True, padding_value=labels2idx.get('<PAD>'))
        pad_y = pad_sequence(data_y_stoi, batch_first=True,
                             padding_value=labels2idx.get('<PAD>'))
        assert pad_x.shape == pad_y.shape, f"pad_x shape: {pad_x.shape} does not match pad_y shape: {pad_y.shape}"

        if self._verbose:
            print(f"Data & Labels shape: {pad_x.shape} => (Samples Num, max_len)")

        for i in tqdm(range(len(pad_x)), desc="Indexing dataset"):
            self.encoded_data.append({'inputs': pad_x[i], 'char_inputs': char_pad_x[i], 'outputs': pad_y[i]})

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
