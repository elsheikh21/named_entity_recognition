import os
from data_loader import DatasetParser
from utilities import configure_workspace


if __name__ == '__main__':
    configure_workspace()
    file_path_ = os.path.join(os.getcwd(), 'Data', 'train.tsv')
    train_dataset = DatasetParser(file_path_)

    file_path_dev = os.path.join(os.getcwd(), 'Data', 'dev.tsv')
    dev_dataset = DatasetParser(file_path_)

    file_path_test = os.path.join(os.getcwd(), 'Data', 'test.tsv')
    test_dataset = DatasetParser(file_path_)
