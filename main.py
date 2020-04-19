import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_loader import TSVDatasetParser
from evaluator import Evaluator
from models import HyperParameters, BaselineModel
from training import Trainer
from utilities import configure_workspace


def prepare_data():
    file_path_ = os.path.join(os.getcwd(), 'Data', 'train.tsv')
    train_dataset = TSVDatasetParser(file_path_, max_len=173, verbose=False, is_crf=True)
    train_dataset.encode_dataset(train_dataset.word2idx, train_dataset.labels2idx)

    dev_file_path = os.path.join(os.getcwd(), 'Data', 'test.tsv')
    dev_dataset = TSVDatasetParser(dev_file_path, max_len=173, verbose=False, is_crf=True)
    dev_dataset.encode_dataset(train_dataset.word2idx, train_dataset.labels2idx)

    test_file_path = os.path.join(os.getcwd(), 'Data', 'test.tsv')
    test_dataset = TSVDatasetParser(test_file_path, max_len=173, verbose=False, is_crf=True)
    test_dataset.encode_dataset(train_dataset.word2idx, train_dataset.labels2idx)

    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
    configure_workspace(seed=1873337)
    train_dataset, dev_dataset, test_dataset = prepare_data()

    batch_size = 128
    hp = HyperParameters('LSTM', train_dataset.word2idx, train_dataset.labels2idx, None, batch_size)
    baseline_model = BaselineModel(hp).to(train_dataset._device)
    print('\n========== Model Summary ==========')
    print(baseline_model)

    # , collate_fn=DatasetParser.pad_collate
    train_dataset_ = DataLoader(dataset=train_dataset, batch_size=batch_size)
    dev_dataset_ = DataLoader(dataset=dev_dataset, batch_size=batch_size)
    test_dataset_ = DataLoader(dataset=test_dataset, batch_size=batch_size)

    trainer = Trainer(
        model=baseline_model,
        loss_function=nn.CrossEntropyLoss(ignore_index=train_dataset.labels2idx['<PAD>']),
        optimizer=Adam(baseline_model.parameters()),  # weight_decay=1e-5, momentum=0.95, nesterov=True),
        verbose=True
    )

    trainer.train(train_dataset_, dev_dataset_, epochs=1, save_to=save_model_path)

    evaluator = Evaluator(baseline_model, dev_dataset_, train_dataset.idx2label)
    evaluator.check_performance()
