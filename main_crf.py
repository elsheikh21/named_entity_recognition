from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from utilities import configure_workspace, load_pretrained_embeddings
from data_loader import TSVDatasetParser
from models import HyperParameters, BaselineModel, CRF_Model
from evaluator import Evaluator
from training import Trainer, CRF_Trainer
from os.path import join
from os import getcwd
from torch.utils.data import DataLoader


def prepare_data(crf_model):
    DATA_PATH = join(getcwd(), 'Data')

    print("==========Training Dataset==========")
    file_path_ = join(DATA_PATH, 'train.tsv')
    training_set = TSVDatasetParser(file_path_, max_len=80, is_crf=crf_model)
    training_set.encode_dataset(training_set.word2idx, training_set.labels2idx)

    print("==========Validation Dataset==========")
    dev_file_path = join(DATA_PATH, 'dev.tsv')
    validation_set = TSVDatasetParser(dev_file_path, max_len=80, is_crf=crf_model)
    validation_set.encode_dataset(training_set.word2idx, training_set.labels2idx)

    print("==========Testing Dataset==========")
    test_file_path = join(DATA_PATH, 'test.tsv')
    testing_set = TSVDatasetParser(test_file_path, max_len=80, is_crf=crf_model)
    testing_set.encode_dataset(training_set.word2idx, training_set.labels2idx)

    return training_set, validation_set, testing_set


if __name__ == '__main__':
    RESOURCES_PATH = join(getcwd(), 'resources')
    configure_workspace(seed=1873337)
    crf_model = True
    train_dataset, dev_dataset, test_dataset = prepare_data(crf_model)

    batch_size = 64
    pretrained_embeddings = None

    embeddings_path = join(RESOURCES_PATH, 'wiki.en.vec')
    # pretrained_embeddings = load_pretrained_embeddings(embeddings_path,
    #                                                    train_dataset.word2idx,
    #                                                    300, is_crf=crf_model)

    name_ = 'LSTM_CRF' if crf_model else 'LSTM'
    hp = HyperParameters(name_, train_dataset.word2idx,
                         train_dataset.labels2idx,
                         pretrained_embeddings,
                         batch_size)

    # , collate_fn=DatasetParser.pad_collate
    train_dataset_ = DataLoader(dataset=train_dataset, batch_size=batch_size)
    dev_dataset_ = DataLoader(dataset=dev_dataset, batch_size=batch_size)
    test_dataset_ = DataLoader(dataset=test_dataset, batch_size=batch_size)

    if not crf_model:
        model = BaselineModel(hp).to(train_dataset.get_device)
        print(f'\n========== Model Summary ==========\n{model}')

        trainer = Trainer(
            model=model,
            loss_function=CrossEntropyLoss(ignore_index=train_dataset.labels2idx['<PAD>']),
            optimizer=Adam(model.parameters()),
            batch_num=hp.batch_size,
            num_classes=hp.num_classes,
            verbose=True
        )
        save_to_ = join(RESOURCES_PATH, f"{model.name}_model.pt")
        trainer.train(train_dataset_, dev_dataset_, epochs=1, save_to=save_to_)
    else:
        model = CRF_Model(hp).to(train_dataset.get_device)
        print(f'========== CRF Model Summary ==========\n{model}')

        trainer = CRF_Trainer(
            model=model,
            loss_function=CrossEntropyLoss(ignore_index=train_dataset.labels2idx['<PAD>']),
            optimizer=SGD(model.parameters(), lr=1e-4, nesterov=True, momentum=0.9),
            label_vocab=train_dataset.labels2idx
        )
        trainer.train(train_dataset_, dev_dataset_, epochs=1)
        model.save_(join(RESOURCES_PATH, f"{model.name}_model.pt"))

    evaluator = Evaluator(model, test_dataset_, train_dataset.idx2label)
    evaluator.check_performance()
