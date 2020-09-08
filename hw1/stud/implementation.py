import torch
import os
from typing import List, Any
from model import Model
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import nltk
from stud.data_loader import TSVDatasetParser
from stud.models import HyperParameters, CRF_Model
from stud.utilities import load_pickle, configure_workspace


class TSVTestDataParser(Dataset):
    def __init__(self, tokens):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokens = [[_x.lower() for _x in x] for x in tokens]
        self.label2idx = {'<PAD>': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'O': 4}
        self.idx2label = {key: val for key, val in enumerate(self.label2idx)}
        self.encoded_data = []

    def get_element(self, idx):
        return self._tokens[idx]

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Dataset is not indexed yet.\
                                To fetch raw elements, use get_element(idx)")
        return self.encoded_data[idx]

    def encode_data(self, word2idx):
        for sentence in self._tokens:
            self.encoded_data.append({
                "inputs": torch.LongTensor([word2idx.get(word, 1) for word in sentence])
            })

    @staticmethod
    def pad_batch(batch):
        return {"inputs": pad_sequence([sample["inputs"] for sample in batch], batch_first=True)}

    @staticmethod
    def decode_predictions(labels, idx2label):
        return list(map(lambda x: [idx2label.get(w) for w in x], labels))


def build_model(device: str) -> Model:
    configure_workspace(seed=1873337)
    return StudentModel(device)


class StudentModel(Model):
    def __init__(self, device):
        self.device = device
        self.word2idx = load_pickle(os.path.join(os.getcwd(), 'model', 'Stacked_BiLSTM_CRF_Fasttext_2315_word2idx.pkl'))
        self.idx2label = load_pickle(os.path.join(os.getcwd(), 'model', 'Stacked_BiLSTM_CRF_Fasttext_2315_idx2label.pkl'))
        self._build_model()

    def _build_model(self):
        hp = HyperParameters(model_name_='BiLSTM_CRF', vocab=self.word2idx,
                             label_vocab=self.idx2label, embeddings_=None,
                             batch_size_=128)
        model_path = os.path.join(os.getcwd(), 'model',
                                  'Stacked_BiLSTM_CRF_Fasttext_2315.pth')

        self.model = CRF_Model(hp).to(self.device)
        self.model.load_model(model_path)
        self.model.eval()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        data_set = TSVTestDataParser(tokens)
        data_set.encode_data(self.word2idx)
        data_set_loader = DataLoader(dataset=data_set, batch_size=128,
                                     collate_fn=TSVTestDataParser.pad_batch)
        with torch.no_grad():
            for sample in data_set_loader:
                inputs = sample["inputs"].to(self.device)
                attention_mask = (inputs != 0).to(self.device, dtype=torch.uint8)
                labels = self.model.predict_new(inputs, attention_mask)
                return TSVTestDataParser.decode_predictions(labels, self.idx2label)
