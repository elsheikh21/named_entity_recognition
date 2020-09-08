import os

import torch
from torch.utils.data import DataLoader

from evaluator import Evaluator
from main_crf import prepare_data
from models import HyperParameters, BaselineModel, CRF_Model
from utilities import configure_workspace, load_pretrained_embeddings, load_pickle

if __name__ == "__main__":
    CRF_MODEL = True
    PRETRAINED = False
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
    batch_size = 128
    name_ = 'CRF_Model' if CRF_MODEL else 'Model'
    model_path = os.path.join(RESOURCES_PATH, 'Stacked_BiLSTM_CRF_Fasttext_2315.pth')

    configure_workspace(seed=1873337)

    train_dataset, dev_dataset, test_dataset = prepare_data(CRF_MODEL)
    train_dataset_ = DataLoader(dataset=train_dataset, batch_size=batch_size)
    dev_dataset_ = DataLoader(dataset=dev_dataset, batch_size=batch_size)
    test_dataset_ = DataLoader(dataset=test_dataset, batch_size=batch_size)

    embeddings_path = os.path.join(RESOURCES_PATH, 'wiki.en.vec')
    pretrained_embeddings = load_pretrained_embeddings(embeddings_path, train_dataset.word2idx, 300,
                                                       is_crf=CRF_MODEL) if PRETRAINED else None

    idx2label = load_pickle(os.path.join(RESOURCES_PATH, 'Stacked_BiLSTM_CRF_Fasttext_2315_idx2label.pkl'))
    word2idx = load_pickle(os.path.join(RESOURCES_PATH, 'Stacked_BiLSTM_CRF_Fasttext_2315_word2idx.pkl'))
    hp = HyperParameters(name_, word2idx, train_dataset.idx2label,
                         pretrained_embeddings, batch_size)

    model = CRF_Model(hp).to(train_dataset.get_device) if CRF_MODEL else BaselineModel(hp).to(train_dataset.get_device)
    model.load_model(model_path)

    evaluator = Evaluator(model, test_dataset_, CRF_MODEL)
    evaluator.check_performance(idx2label)
    tokens = test_dataset.data_x
    preds_lst = model.predict_sentences(tokens, word2idx, idx2label)
    with open('preds.txt', encoding='utf-8', mode='w+') as f:
        for lst in preds_lst:
            f.write(f"{str(lst)}\n")
