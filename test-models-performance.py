import os

from torch.utils.data import DataLoader

from evaluator import Evaluator
from main_crf import prepare_data
from models import HyperParameters, BaselineModel, CRF_Model
from utilities import configure_workspace, load_pretrained_embeddings

if __name__ == "__main__":
    CRF_MODEL = True
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
    PRETRAINED = True
    batch_size = 16
    name_ = 'CRF_Model' if CRF_MODEL else 'Model'
    model_path = os.path.join(RESOURCES_PATH, 'bilstm-crf-new (2).pt')

    configure_workspace(seed=1873337)

    train_dataset, dev_dataset, test_dataset = prepare_data(CRF_MODEL)
    train_dataset_ = DataLoader(dataset=train_dataset, batch_size=batch_size)
    dev_dataset_ = DataLoader(dataset=dev_dataset, batch_size=batch_size)
    test_dataset_ = DataLoader(dataset=test_dataset, batch_size=batch_size)

    embeddings_path = os.path.join(RESOURCES_PATH, 'wiki.en.vec')
    pretrained_embeddings = load_pretrained_embeddings(embeddings_path, train_dataset.word2idx, 300,
                                                       is_crf=CRF_MODEL) if not PRETRAINED else None

    hp = HyperParameters(name_, train_dataset.word2idx,
                         train_dataset.labels2idx,
                         pretrained_embeddings,
                         batch_size)

    model = CRF_Model(hp).to(train_dataset.get_device) if CRF_MODEL else BaselineModel(hp).to(train_dataset.get_device)
    model.load_model(model_path)
    evaluator = Evaluator(model, test_dataset_, train_dataset.idx2label, CRF_MODEL)
    evaluator.check_performance()
