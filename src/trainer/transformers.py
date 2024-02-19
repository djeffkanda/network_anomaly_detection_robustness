import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from .base import BaseTrainer


class NeuTraLADTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(NeuTraLADTrainer, self).__init__(**kwargs)
        self.metric_hist = []

        mask_params = list()
        for mask in self.model.masks:
            mask_params += list(mask.parameters())
        self.optimizer = optim.Adam(list(self.model.enc.parameters()) + mask_params, lr=self.lr,
                                    weight_decay=self.weight_decay)

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.9)
        self.criterion = nn.MSELoss()

    def score(self, sample: torch.Tensor):
        loss_n, loss_a = self.model(sample)
        return loss_n

    def train_iter(self, X, **kwargs):
        alpha = kwargs['contamination_rate']
        y = kwargs['label']
        loss_n, loss_a = self.model(X)
        loss = loss_n.mean()
        if self.kwargs['rob'] and self.kwargs["warmup"] < 1 + kwargs["epoch"]:
            if self.kwargs["rob_method"] == 'loe':
                score = loss_n - loss_a
                _, idx_n = torch.topk(score, int(score.shape[0] * (1 - alpha)), largest=False,
                                      sorted=False)
                _, idx_a = torch.topk(score, int(score.shape[0] * alpha), largest=True,
                                      sorted=False)
                loss = torch.cat([loss_n[idx_n], 0.5 * loss_n[idx_a] + 0.5 * loss_a[idx_a]], 0)
                loss = loss.mean()
            elif self.kwargs["rob_method"] == 'sup':
                idx_n = torch.nonzero(y == 0, as_tuple=True)
                idx_a = torch.nonzero(y == 1, as_tuple=True)
                loss = torch.cat([loss_n[idx_n], loss_a[idx_a]], dim=0)
                loss = loss.mean()
            else:
                loss = loss_n.mean()

        return loss
