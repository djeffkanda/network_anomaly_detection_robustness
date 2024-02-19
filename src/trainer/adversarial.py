import torch
import torch.nn as nn
from tqdm import trange
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .base import BaseTrainer

torch.autograd.set_detect_anomaly(True)


class ALADTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim_ge, self.optim_d = None, None
        super(ALADTrainer, self).__init__(**kwargs)
        # self.set_optimizer()

    def train_iter(self, sample: torch.Tensor, **kwargs):
        pass

    def score(self, sample: torch.Tensor):
        _, feature_real = self.model.D_xx(sample, sample)
        _, feature_gen = self.model.D_xx(sample, self.model.G(self.model.E(sample)))
        return torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

    def set_optimizer(self, weight_decay):
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) + list(
                self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter_dis(self, X):

        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Compute loss
        # Discriminators Losses
        loss_dxz = self.criterion(out_truexz, y_true) + self.criterion(out_fakexz, y_fake)
        loss_dzz = self.criterion(out_truezz, y_true) + self.criterion(out_fakezz, y_fake)
        loss_dxx = self.criterion(out_truexx, y_true) + self.criterion(out_fakexx, y_fake)
        loss_d = loss_dxz + loss_dzz + loss_dxx

        return loss_d

    def train_iter_gen(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Generator losses
        loss_gexz = self.criterion(out_fakexz, y_true) + self.criterion(out_truexz, y_fake)
        loss_gezz = self.criterion(out_fakezz, y_true) + self.criterion(out_truezz, y_fake)
        loss_gexx = self.criterion(out_fakexx, y_true) + self.criterion(out_truexx, y_fake)
        cycle_consistency = loss_gexx + loss_gezz
        loss_ge = loss_gexz + cycle_consistency

        return loss_ge

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, test_loader: DataLoader = None, **kwargs):
        self.model.train()

        should_stop = False
        for epoch in range(self.n_epochs):
            ge_losses, d_losses = 0, 0
            with trange(len(train_loader)) as t:
                for sample in train_loader:
                    X = sample[0]
                    X_dis, X_gen = X.to(self.device).float(), X.clone().to(self.device).float()
                    # Forward pass

                    # Cleaning gradients
                    self.optim_d.zero_grad()
                    loss_d = self.train_iter_dis(X_dis)
                    # Backward pass
                    loss_d.backward()
                    self.optim_d.step()

                    # Cleaning gradients
                    self.optim_ge.zero_grad()
                    loss_ge = self.train_iter_gen(X_gen)
                    # Backward pass
                    loss_ge.backward()
                    self.optim_ge.step()

                    # Journaling
                    d_losses += loss_d.item()
                    ge_losses += loss_ge.item()
                    t.set_postfix(
                        ep=epoch + 1,
                        loss_d='{:05.4f}'.format(loss_d),
                        loss_ge='{:05.4f}'.format(loss_ge),
                    )
                    t.update()
            if val_loader is not None:
                val_loss = self.eval(val_loader, **kwargs)
                results = self._eval(test_loader)

                if kwargs['early_stopping']:
                    should_stop = self.early_stopper.stop(epoch=epoch,
                                                          val_loss=val_loss,
                                                          train_loss=d_losses + ge_losses,
                                                          val_auc=results["proc1p"],
                                                          test_f1=results["f_score"])

                    print(f'\nVal loss :{val_loss} | Train loss: {d_losses + ge_losses}'
                          f'| early_stop? {should_stop} | patience:{self.early_stopper.counter}  ')
                print(results)

                if should_stop:
                    break
        if kwargs['early_stopping']:
            self.early_stopper.get_best_vl_metrics()

    def eval(self, dataset: DataLoader, **kwargs):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                X = row[0]
                kwargs['label'] = row[1]
                kwargs['index'] = row[2]
                X = X.to(self.device).float()
                loss_d = self.train_iter_dis(X)
                loss_ge = self.train_iter_gen(X)
                loss += loss_d.item() + loss_ge.item()
        self.model.train()
        return loss
