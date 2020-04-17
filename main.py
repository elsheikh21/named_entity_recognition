import os
import torch
from data_loader import DatasetParser
from torch.utils.data import DataLoader
from utilities import configure_workspace
from models import HyperParameters, BaselineModel


def prepare_data():
    is_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

    file_path_ = os.path.join(os.getcwd(), 'Data', 'train.tsv')
    train_dataset = DatasetParser(file_path_, device=is_cuda)
    train_dataset.vectorize_data()

    file_path_dev = os.path.join(os.getcwd(), 'Data', 'dev.tsv')
    dev_dataset = DatasetParser(file_path_dev, device=is_cuda)
    dev_dataset.word2idx = train_dataset.word2idx
    dev_dataset.idx2word = train_dataset.idx2word
    dev_dataset.tags2idx = train_dataset.tags2idx
    dev_dataset.idx2tags = train_dataset.idx2tags
    dev_dataset.vectorize_data()

    file_path_test = os.path.join(os.getcwd(), 'Data', 'test.tsv')
    test_dataset = DatasetParser(file_path_test, device=is_cuda)
    test_dataset.word2idx = train_dataset.word2idx
    test_dataset.idx2word = train_dataset.idx2word
    test_dataset.tags2idx = train_dataset.tags2idx
    test_dataset.idx2tags = train_dataset.idx2tags
    test_dataset.vectorize_data()

    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    configure_workspace(seed=1873337)
    train_dataset, dev_dataset, test_dataset = prepare_data()

    batch_size = 128
    hp = HyperParameters(batch_size, train_dataset.vocab_size, train_dataset.tags_num)

    baseline_model = BaselineModel(hp)
    print('\n========== Model Summary ==========')
    print(baseline_model)

    # train_dataset_ = DataLoader(train_dataset, batch=batch_size,
    #                             shuffle=True, collate_fn=DatasetParser.pad_collate)
    # dev_dataset_ = DataLoader(dev_dataset, batch_size=batch_size,
    #                           collate_fn=DatasetParser.pad_collate)
    # test_dataset_ = DataLoader(test_dataset, batch_size=batch_size,
    #                           collate_fn=DatasetParser.pad_collate)
    #
