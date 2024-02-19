import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Union
from sklearn import metrics as sk_metrics
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import trange

from utils.metrics import _estimate_threshold_metrics

from utils.utils import Patience


class BaseTrainer(ABC):

    def __init__(self, model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 **kwargs):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.optimizer = self.set_optimizer(self.weight_decay)

        patience = kwargs.get('patience', 5)
        self.early_stopper = Patience(patience=patience, use_train_loss=False, model=self.model)
        self.kwargs = kwargs

        # self.contamination_rate =

    @abstractmethod
    def train_iter(self, sample: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    def after_training(self):
        """
        Perform any action after training is done
        """
        pass

    def before_training(self, dataset: DataLoader):
        """
        Optionally perform pre-training or other operations.
        """
        pass

    def set_optimizer(self, weight_decay):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, test_loader: DataLoader = None, **kwargs):
        self.model.train()
        self.before_training(train_loader)

        print("Started training")
        should_stop = False
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            # if not val_loader is None:
            #     loss = self.eval(train_loader)
            #     print(f'validation score: loss:{loss:.3f}, score:{np.mean(self.test(val_loader)[1]):.3f}')
            len_trainloader = len(train_loader)
            counter = 1

            with trange(len_trainloader) as t:
                for sample in train_loader:
                    X = sample[0]
                    kwargs['label'] = sample[1]
                    kwargs['index'] = sample[2]
                    kwargs['epoch'] = epoch
                    X = X.to(self.device).float()
                    # TODO handle this just for trainer DBESM
                    # if len(X) < self.batch_size:
                    #     t.update()
                    #     break

                    # Reset gradient
                    self.optimizer.zero_grad()

                    loss = self.train_iter(X, **kwargs)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.3f}'.format(epoch_loss / counter),
                        epoch=epoch + 1
                    )
                    t.update()
                    counter += 1
            if val_loader is not None:
                val_loss = self.eval(val_loader, **kwargs).item()
                results = self._eval(test_loader)

                if kwargs['early_stopping']:
                    should_stop = self.early_stopper.stop(epoch=epoch,
                                                          val_loss=val_loss,
                                                          train_loss=epoch_loss,
                                                          val_auc=results["proc1p"],
                                                          test_f1=results["f_score"])

                    print(f'Val loss :{val_loss} | Train loss: {self.eval(train_loader, **kwargs).item()} '
                          f'| early_stop? {should_stop} | patience:{self.early_stopper.counter}  ')
                print(results)

                if should_stop:
                    break
        if kwargs['early_stopping']:
            self.early_stopper.get_best_vl_metrics()

        self.after_training()

    def eval(self, dataset: DataLoader, **kwargs):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                X = row[0]
                kwargs['label'] = row[1]
                kwargs['index'] = row[2]
                X = X.to(self.device).float()
                loss += self.train_iter(X, **kwargs)
        self.model.train()

        return loss

    def _eval(self, dataset: DataLoader):
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # if len(X) < self.batch_size:
                #     break
                score = self.score(X)
                y_true.append(y.cpu().numpy())
                scores.append(score.cpu().numpy())
        self.model.train()

        y_true, scores = np.concatenate(y_true, axis=0), np.concatenate(scores, axis=0)
        # _estimate_threshold_metrics

        accuracy, precision, recall, f_score, roc, avgpr = _estimate_threshold_metrics(scores, y_true,
                                                                                       optimal=False)

        return {k: round(v, 3) for k, v in
                dict(accuracy=accuracy,
                     precision=precision, recall=recall, f_score=f_score, avgpr=avgpr, proc1p=roc).items()}

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # if len(X) < self.batch_size:
                #     break
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores)

    def test_return_all(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores, xs = [], [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # if len(X) < self.batch_size:
                #     break
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
                xs.extend(X.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores), np.array(xs)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.lr,
            "epochs": self.n_epochs,
            "batch_size": self.batch_size,
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)


class BaseShallowTrainer(ABC):

    def __init__(self, model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda"):
        """
        Parameters are mostly ignored but kept for better code consistency

        Parameters
        ----------
        model
        batch_size
        lr
        n_epochs
        n_jobs_dataloader
        device
        """
        self.device = None
        self.model = model
        self.batch_size = None
        self.n_jobs_dataloader = None
        self.n_epochs = None
        self.lr = None

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)

    def score(self, sample: torch.Tensor):
        return self.model.predict(sample.numpy())

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        y_true, scores = [], []
        for row in dataset:
            X, y = row[0], row[1]
            score = self.score(X)
            y_true.extend(y.cpu().tolist())
            scores.extend(score)

        return np.array(y_true), np.array(scores)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)
