import os
import logging

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm.auto import tqdm
from utilities import configure_workspace


class DatasetParser(Dataset):
    def __init__(self, file_path: str, device: str):
        self.device = device
        self.train_x, self.train_y = None, None
        self.encoded_data = None

        logging.info(f'Parsing the dataset from "{file_path}".')
        self.data_x, self.data_y = self.parse_dataset(file_path)
        logging.info('Creating Tags Vocabulary')
        self.tags2idx = {'<PAD>': 0, 'LOC': 1, 'ORG': 2, 'PER': 3, 'O': 4}
        self.idx2tags = {0: '<PAD>', 1: 'LOC', 2: 'ORG', 3: 'PER', 4: 'O'}
        logging.info(f'Number of tags {len(self.tags2idx) - 1}')
        logging.info('Creating Vocabulary')
        self.word2idx, self.idx2word = self.create_vocab()
        logging.info(f'Vocabulary created with {len(self.word2idx)} unique tokens')

    @staticmethod
    def parse_dataset(file_path):
        with open(file_path, encoding='utf-8', mode='r') as file_:
            lines = file_.read().splitlines()
        data_x, data_y = [], []
        sentence_tags = []
        for line in tqdm(lines):
            sentence_x = []
            if line == '':
                if sentence_tags:
                    data_y.append(sentence_tags)
                    sentence_tags = []
                continue
            elif line[0] == '#':
                sentence = line.replace('# ', '')
                sentence_x.append(sentence)
                data_x.append(sentence_x)
                if sentence_tags:
                    data_y.append(sentence_tags)
                    sentence_tags = []
            elif line[0].isdigit():
                sentence_tags.append(line.split('\t')[-1])
        return data_x, data_y

    def create_vocab(self):
        sentences = sum(self.data_x, [])
        all_words = list(set([word for sentence in sentences for word in sentence.split()]))
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        word2idx.update({val: key for (key, val) in enumerate(all_words, start=2)})
        idx2word = {val: key for (key, val) in word2idx.items()}
        return word2idx, idx2word

    def index_data_x(self):
        sentences = []
        for sentence in self.data_x:
            sentence_ = []
            for word in sentence[0].split():
                sentence_.append(self.word2idx.get(word, "<UNK>"))
            sentences.append(sentence_)
        return sentences

    def index_data_y(self):
        labels = []
        for label_lst in self.data_y:
            label_ = []
            for label in label_lst:
                label_.append(self.tags2idx.get(label, "<UNK>"))
            labels.append(label_)
        return labels

    def vectorize_data(self):
        # data_x.shape = [samples_num, max_chars_sentence]
        self.train_x = self.index_data_x()
        self.train_y = self.index_data_y()

        assert len(self.train_x) == len(self.train_y)
        self.encoded_data = list()
        for i in range(len(self.train_x)):
            train_x = torch.LongTensor(self.train_x[i]).to(self.device)
            train_y = torch.LongTensor(self.train_y[i]).to(self.device)
            self.encoded_data.append({"inputs": train_x, "outputs": train_y})

    @staticmethod
    def decode_logits(logits: torch.Tensor, idx2label):
        max_indices = torch.argmax(logits, -1).tolist()  # shape = (batch_size, max_len)
        predictions = list()
        for indices in max_indices:
            predictions.append([idx2label[i] for i in indices])
        return predictions

    @staticmethod
    def decode_data(data: torch.Tensor, idx2label):
        data_ = data.tolist()
        return [idx2label.get(idx, None) for idx in data_]

    @staticmethod
    def pad_collate(batch):
        data_x, data_y = [], []
        for item in batch:
            data_x.append(item.get('inputs'))
            data_y.append(item.get('outputs'))
            seq_tensor, seq_lengths = batch.pad_per_batch(data_x)
            lbl_tensor, _ = batch.pad_per_batch(data_y)
            return seq_tensor, lbl_tensor, seq_lengths

    @staticmethod
    def pad_per_batch(data_):
        seq_lengths = torch.LongTensor(list(map(len, data_)))
        seq_tensor = Variable(torch.zeros((len(data_), seq_lengths.max()))).long()
        for idx, (seq, seq_len) in enumerate(zip(data_, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        return seq_tensor.to('cuda'), seq_lengths

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Trying to retrieve elements, but dataset is not vectorized yet")
        return self.encoded_data[idx]


if __name__ == '__main__':
    configure_workspace()
    file_path_ = os.path.join(os.getcwd(), 'Data', 'train.tsv')
    is_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = DatasetParser(file_path_, device=is_cuda)
    dataset.vectorize_data()
    print(dataset.train_x[:10])
