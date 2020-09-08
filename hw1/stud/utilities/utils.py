import logging
import os
import pickle
import random
import warnings

import numpy as np
import torch
import nltk


def configure_workspace(seed):
    """
    Fixes the SEED to maintain reproducibility, configure logging configuration
    :param seed: int
    :return: None
    """
    nltk.download('averaged_perceptron_tagger', quiet=True)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


def save_pickle(save_to, save_what):
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f)


def load_pickle(load_from):
    with open(load_from, 'rb') as f:
        return pickle.load(f)


def ensure_dir(path):
    """
    Ensures if a directory exists, else creates it
    """
    if not os.path.exists(path):
        os.makedirs(path)
