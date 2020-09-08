import io
import os
import torch
import multiprocessing
from gensim.models import Word2Vec
from torch.nn.modules.module import _addindent
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from stud.models import BaselineModel


def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """
    Save best model
    Args
        state:
        is_best:
        filename:

    Returns:

    """
    if is_best:
        print("Saving a new best model")
        torch.save(state, filename)  # save checkpoint


def load_checkpoint(resume_weights_path, hyperparams):
    """
    Loads best checkpoint to retrain the model
    Args:
        resume_weights_path:
        hyperparams:

    Returns:

    """
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(resume_weights_path)

    start_epoch = checkpoint['epoch']
    best_validation_loss = checkpoint['best_val_loss']
    model = BaselineModel(hyperparams)
    model.load_state_dict(checkpoint['state_dict'])
    print(
        f"loaded checkpoint '{checkpoint}' (trained for {start_epoch} epochs, val loss: {best_validation_loss})")
    return model


def load_pretrained_embeddings(file_name, word2idx, embeddings_size, is_crf=False, save_to=None):
    """
    Loads pretrained embeddings for Fasttext files, creates tensors full of zeros [vocab size, Embedding size], and
    initialize those tensors with data from fasttext file, and keeps count of how many was init.

    Args:
        file_name:
        word2idx:
        embeddings_size:
        is_crf:

    Returns:
        pretrained_embeddings

    """
    if os.path.exists(save_to):
        pretrained_embeddings = np.load(save_to)
    else:
        fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        for line in tqdm(fin, desc=f'Reading data from {file_name}'):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float)

        pretrained_embeddings = torch.randn(len(word2idx), embeddings_size)
        initialised = 0
        for idx, word in enumerate(data):
            if word in word2idx:
                initialised += 1
                vector_ = torch.from_numpy(data[word])
                pretrained_embeddings[word2idx.get(word)] = vector_

        pretrained_embeddings[word2idx["<PAD>"]] = torch.zeros(embeddings_size)
        pretrained_embeddings[word2idx["<UNK>"]] = torch.zeros(embeddings_size)
        if is_crf:
            pretrained_embeddings[word2idx["<BOS>"]] = torch.zeros(embeddings_size)
            pretrained_embeddings[word2idx["<EOS>"]] = torch.zeros(embeddings_size)
        print(f'Loaded {initialised} vectors and instantiated random embeddings for {len(word2idx) - initialised}')

        np.save(save_to, pretrained_embeddings) # save the file as "outfile_name.npy"
    return pretrained_embeddings


def plot_history(history):
    """
    Utility function to plot training history [NOT USED]
    Args:
        history:

    Returns:

    """
    loss_list = [s for s in history['loss']]
    val_loss_list = [s for s in history['val_loss']]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = [i for i in range(1, len(history['loss']) + 1)]

    # Loss
    plt.figure(1)
    plt.plot(epochs, loss_list, label="loss")
    plt.plot(epochs, val_loss_list, label="val_loss")

    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'resources', 'loss_plot.png'))
    plt.show()


def torch_summarize(model, show_weights=True, show_parameters=True):
    """
    Code Source: https://stackoverflow.com/a/45528544/8279593
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr



def train_pos2vec(training_data, window_=25, embed_size=300, lr=1e-3,
                  epochs=50, verbose=False):
    cpu_cores = multiprocessing.cpu_count() - 1
    pos2vec_model = Word2Vec(min_count=5, window=window_, size=embed_size,
                             sample=1e-3, alpha=lr, min_alpha=0.0007,
                             negative=20, workers=cpu_cores, seed=1873337)
    pos2vec_model.build_vocab(training_data, progress_per=10000)
    if verbose: print(pos2vec_model)
    pos2vec_model.train(training_data, total_examples=pos2vec_model.corpus_count,
                        epochs=epochs, report_delay=1)
    return pos2vec_model


def save_pos_embeddings(pos2vec_model, path_to_save):
    pos_embeds = [token for token in pos2vec_model.wv.vocab]
    with open(path_to_save, "w+") as f:
        for key in pos_embeds:
            f.write(f"{key} ")
            for el in pos2vec_model[key]:
                f.write(f"{el} ")
            f.write("\n")
