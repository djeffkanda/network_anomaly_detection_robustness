import os
import random
import tempfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from typing import Type, Callable


def predict_proba(scores):
    """
    Predicts probability from the score

    Parameters
    ----------
    scores: the score values from the model

    Returns
    -------

    """
    prob = F.softmax(scores, dim=1)
    return prob


def check_dir(path):
    """
    This function ensure that a path exists or create it
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_file_exists(path):
    """
    This function ensure that a path exists
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_X_from_loader(loader):
    """
    This function returns the data set X from the provided pytorch @loader
    """
    X = []
    y = []
    for i, X_i in enumerate(loader, 0):
        X.append(X_i[0])
        y.append(X_i[1])
    X = torch.cat(X, axis=0)
    y = torch.cat(y, axis=0)
    return X.numpy(), y.numpy()


def average_results(results: dict):
    """
        Calculate Means and Stds of metrics in @results
    """

    final_results = defaultdict()
    for k, v in results.items():
        final_results[f'{k}'] = f"{np.mean(v):.4f}({np.std(v):.4f})"
        # final_results[f'{k}_std'] = np.std(v)
    return final_results


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> Callable[
    [torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Parameters
    ----------
    optimizer_class: optimizer used to train the model
    hyperparameters: hyperparameters for the model

    Returns
    -------

    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f


def random_split_to_two(table, ratio=.2):
    n1 = int(len(table) * (1 - ratio))
    shuffle_idx = torch.randperm(len(table)).long()

    t1 = table[shuffle_idx[:n1]]
    t2 = table[shuffle_idx[n1:]]

    return t1, t2


def seed_everything(seed=1234):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Code from https://github.com/boschresearch/LatentOE-AD
class EarlyStopper:

    def stop(self, epoch, val_loss, val_auc=None, test_loss=None, test_auc=None, test_ap=None, test_f1=None,
             test_score=None, train_loss=None):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return self.train_loss, self.val_loss, self.val_auc, self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score, self.best_epoch


class Patience(EarlyStopper):
    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=10, use_train_loss=True, model=None):
        self.local_val_optimum = float("inf")
        self.use_train_loss = use_train_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.model = model
        self.temp_dir = tempfile.TemporaryDirectory()
        self.best_model_path = f"{self.temp_dir.name}/model.pk"
        self.train_loss = None
        self.val_loss, self.val_auc, = None, None
        self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score = None, None, None, None, None

    def stop(self, epoch, val_loss, val_auc=None, test_loss=None, test_auc=None, test_ap=None, test_f1=None,
             test_score=None, train_loss=None):
        if self.use_train_loss:
            if train_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = train_loss
                self.best_epoch = epoch
                self.train_loss = train_loss
                self.val_loss, self.val_auc = val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score \
                    = test_loss, test_auc, test_ap, test_f1, test_score

                self.model.save(self.best_model_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss = train_loss
                self.val_loss, self.val_auc = val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score \
                    = test_loss, test_auc, test_ap, test_f1, test_score

                self.model.save(self.best_model_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience

    def get_best_vl_metrics(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        return self.model, self.train_loss, self.val_loss, self.val_auc, self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score, self.best_epoch

