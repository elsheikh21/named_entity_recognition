import os
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from data_loader import DatasetParser
from utilities import configure_workspace
from models import HyperParameters, BaselineModel
from training import Trainer
from evaluator import Evaluator


def prepare_data(is_cuda):
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
    is_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
    configure_workspace(seed=1873337)
    train_dataset, dev_dataset, test_dataset = prepare_data(is_cuda)

    batch_size = 128
    hp = HyperParameters(batch_size, train_dataset.vocab_size, train_dataset.tags_num)

    baseline_model = BaselineModel(hp).to(is_cuda)
    print('\n========== Model Summary ==========')
    print(baseline_model)

    train_dataset_ = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_dataset, batch_size=batch_size)
    test_dataset_ = DataLoader(dataset=test_dataset, batch_size=batch_size)

    trainer = Trainer(
        model=baseline_model,
        loss_function=nn.CrossEntropyLoss(ignore_index=train_dataset.tags2idx['<PAD>']),
        optimizer=SGD(baseline_model.parameters(), lr=1e-6, weight_decay=1e-5, momentum=0.95, nesterov=True),
        label_vocab=train_dataset.tags2idx
    )

    save_model_path = os.path.join(RESOURCES_PATH, f'{baseline_model.name}_model.pt')
    trainer.train(train_dataset_, dev_dataset_, epochs=2, save_to=save_model_path)

    evaluator = Evaluator(baseline_model, dev_dataset_, train_dataset.idx2tags, train_dataset.tags_num)
