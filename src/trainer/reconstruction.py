from collections import defaultdict

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearnex import patch_sklearn

# The names match scikit-learn estimators
patch_sklearn()

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler, Normalizer
from torch.utils.data.dataloader import DataLoader
from .base import BaseTrainer
from loss.EntropyLoss import EntropyLoss
from torch import nn
import torch.nn.functional as F

from sklearn import metrics as sk_metrics


class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 **kwargs):

        super(AutoEncoderTrainer, self).__init__(model, batch_size, lr, n_epochs, n_jobs_dataloader, device, **kwargs)
        self.tracker = defaultdict(list)

    def score(self, sample: torch.Tensor):
        _, X_prime = self.model(sample)
        return ((sample - X_prime) ** 2).sum(axis=1)

    def re_evaluation_ocsvm(self, X, p, num_clusters=20):

        clf = LocalOutlierFactor(n_neighbors=30)
        clf.fit(X.cpu().numpy())  # OneClassSVM(gamma='auto').fit(X)
        pred_label = clf.negative_outlier_factor_  # .score_samples(X)
        # gmm = GaussianMixture(n_components=num_clusters, max_iter=800)
        # gmm.fit(X)
        # pred_label = gmm.score_samples(X)
        scaler = MinMaxScaler(feature_range=(-2, 2))
        pred_label = pred_label.reshape(-1, 1)
        scaler.fit(pred_label)
        pred_label = scaler.transform(pred_label.reshape(-1, 1))
        tau = 1.0
        eps = 0.0
        pred_label = 1 / (1 + np.exp(-(eps + tau * pred_label)))

        indices_selection = 1 - np.ravel(pred_label)

        return indices_selection

    def re_evaluation_proba_(self, X, p, num_clusters=20):

        gmm = BayesianGaussianMixture(n_components=num_clusters, max_iter=800)
        gmm.fit(X)
        pred_label = gmm.predict_proba(X)
        pred_label = np.dot(pred_label, gmm.weights_)
        scaler = MinMaxScaler(feature_range=(-1, 2))
        pred_label = pred_label.reshape(-1, 1)
        scaler.fit(pred_label)
        pred_label = scaler.transform(pred_label.reshape(-1, 1))
        tau = 1.0
        eps = 0.0
        pred_label = 1 / (1 + np.exp(-(eps + tau * pred_label)))

        indices_selection = np.ravel(pred_label)

        return indices_selection

    def re_evaluation_proba(self, X, p, num_clusters=20):

        gmm = GaussianMixture(n_components=num_clusters, max_iter=800)
        gmm.fit(X)
        pred_label = gmm.score_samples(X)
        scaler = MinMaxScaler(feature_range=(-1, 2))
        pred_label = pred_label.reshape(-1, 1)
        scaler.fit(pred_label)
        pred_label = scaler.transform(pred_label.reshape(-1, 1))
        tau = 1.0
        eps = 0.0
        pred_label = 1 / (1 + np.exp(-(eps + tau * pred_label)))

        indices_selection = np.ravel(pred_label)
        return indices_selection

    def re_evaluation(self, X, p, num_clusters=20):
        # uv = np.unique(X, axis=0)
        gmm = GaussianMixture(n_components=num_clusters, max_iter=800)
        gmm.fit(X)
        pred_label = gmm.predict(X)
        X_means = torch.from_numpy(gmm.means_)

        clusters_vars = []
        for i in range(num_clusters):
            n_samples = (pred_label == i).sum()
            var_ci = torch.sum((X[pred_label == i] - X_means[i].unsqueeze(dim=0)) ** 2)
            var_ci /= n_samples
            clusters_vars.append(var_ci)

        clusters_vars = torch.stack(clusters_vars)
        clusters_vars = torch.nan_to_num(clusters_vars, nan=-10000)

        qp = 100 - p
        # q_ = np.percentile(clusters_vars, qp)
        q = torch.quantile(clusters_vars[clusters_vars > 0], qp / 100)

        selected_clusters = (clusters_vars <= q).nonzero().squeeze()

        df = pd.DataFrame(dict(var=clusters_vars.cpu().numpy(), weights=gmm.weights_))
        # pred_label_ = torch.from_numpy(pred_label).unsqueeze(dim=1)
        selected_clusters = list(selected_clusters.cpu().numpy())

        selection_mask = [pred not in selected_clusters for pred in pred_label]
        indices_selection = torch.from_numpy(
            np.array(selection_mask))  # .nonzero().squeeze()

        return indices_selection

    def train_iter(self, X, **kwargs):
        code, X_prime = self.model(X)

        reg_n = self.kwargs.get('reg_n', 0)  # 1e-3 # 1e-3
        reg_a = self.kwargs.get('reg_a', 0)  # 1e-1 * 0
        alpha = kwargs['contamination_rate']
        alpha_off_set = self.kwargs.get('alpha_off_set', 0.0)
        num_clusters = self.kwargs.get("num_clusters", 3)

        alpha -= alpha * alpha_off_set
        y = kwargs['label']

        dataset = {}

        if self.kwargs.get('rob', False) and self.kwargs.get('warmup', 0) < 1 + kwargs["epoch"]:
            rob_method = self.kwargs.get("rob_method", 'ours')
            if rob_method == 'loe':
                loss_n = ((X - X_prime) ** 2).sum(axis=-1)
                loss_a = 1 / (((X - X_prime) ** 2).sum(axis=-1))

                score = loss_n - loss_a
                _, idx_n = torch.topk(score, int(score.shape[0] * (1 - alpha)), largest=False,
                                      sorted=False)
                _, idx_a = torch.topk(score, int(score.shape[0] * alpha), largest=True,
                                      sorted=False)
                loss = torch.cat([loss_n[idx_n], 0.5 * loss_n[idx_a] + 0.5 * loss_a[idx_a]], 0)
                loss = loss.mean()
            elif rob_method == 'sup':
                loss_n = ((X - X_prime) ** 2).sum(axis=-1)
                loss_a = 1 / (((X - X_prime) ** 2).sum(axis=-1))

                idx_n = torch.nonzero(y == 0, as_tuple=True)
                idx_a = torch.nonzero(y == 1, as_tuple=True)
                loss = torch.cat([loss_n[idx_n], loss_a[idx_a]], dim=0)
                loss = loss.mean()
            else:

                l2_z = (code ** 2).sum(axis=-1)  # code.norm(2, dim=1)
                loss_n = ((X - X_prime) ** 2).sum(axis=-1) + reg_n * l2_z
                loss_a = reg_a * 1 / (((X - X_prime) ** 2).sum(axis=-1)) + reg_a * 1 / l2_z

                data = torch.cat([code, loss_n.unsqueeze(-1)], dim=-1).cpu().detach()
                # data = torch.cat([code, torch.log(loss_n).unsqueeze(-1)], dim=-1).cpu().detach()

                with torch.no_grad():
                    selection_mask = self.re_evaluation_proba(data, alpha * 100, num_clusters=num_clusters)  # 10
                    # df = pd.DataFrame(dict(gt=y.cpu().numpy(), mask=selection_mask.cpu().numpy().astype(int)))

                    # df = pd.DataFrame(dict(gt=y.cpu().numpy(), mask=selection_mask,
                    #                        cos_sim=data[:, -1], loss_n=loss_n.cpu().detach().numpy(),
                    #                        loss_a=loss_a.cpu().detach().numpy()))
                    # df['loss_n_s'] = df['loss_n'] * selection_mask
                    # df['loss_a_s'] = df['loss_a'] * (1 - selection_mask)
                selection_mask = torch.from_numpy(selection_mask).to(loss_n.device)
                loss = torch.cat([selection_mask * loss_n, (1 - selection_mask) * loss_a], dim=0)

                # loss = torch.cat([loss_n, (1 - selection_mask) * loss_a], dim=0)
                loss = loss.mean()
        else:
            loss = ((X - X_prime) ** 2).sum(axis=-1).mean() # + reg_n * (code ** 2).sum(axis=-1).mean()
            type_of_center = kwargs.get('type_center', 'zero')
            if type_of_center == 'zero':
                loss += reg_n * (code ** 2).sum(dim=-1).mean()
            elif type_of_center == 'learnable':
                loss += reg_n * ((code - self.model.latent_center) ** 2).sum(axis=-1).mean()
            elif type_of_center == 'mean':
                loss += reg_n * ((code - code.mean(dim=0)) ** 2).sum(axis=-1).mean()
        return loss


class DAGMMTrainer(BaseTrainer):
    def __init__(self, lamb_1: float = 0.1, lamb_2: float = 0.005, **kwargs) -> None:
        super(DAGMMTrainer, self).__init__(**kwargs)
        self.lamb_1 = lamb_1
        self.lamb_2 = lamb_2
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.covs = None

    def train_iter(self, sample: torch.Tensor, **kwargs):
        z_c, x_prime, _, z_r, gamma_hat = self.model(sample)
        phi, mu, cov_mat = self.compute_params(z_r, gamma_hat)
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z_r, phi, mu, cov_mat
        )
        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        return self.loss(sample, x_prime, energy_result, pen_cov_mat)

    def loss(self, x, x_prime, energy, pen_cov_mat):
        rec_err = ((x - x_prime) ** 2).mean()
        return rec_err + self.lamb_1 * energy + self.lamb_2 * pen_cov_mat

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()

        with torch.no_grad():
            scores, y_true = [], []
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # forward pass
                code, x_prime, cosim, z, gamma = self.model(X)
                sample_energy, pen_cov_mat = self.estimate_sample_energy(
                    z, self.phi, self.mu, self.cov_mat, average_energy=False
                )
                y_true.extend(y)
                scores.extend(sample_energy.cpu().numpy())

        return np.array(y_true), np.array(scores)

    def weighted_log_sum_exp(self, x, weights, dim):
        """
        Inspired by https://discuss.pytorch.org/t/moving-to-numerically-stable-log-sum-exp-leads-to-extremely-large-loss-values/61938

        Parameters
        ----------
        x
        weights
        dim

        Returns
        -------

        """
        m, idx = torch.max(x, dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m) * (weights.unsqueeze(2)), dim=dim))

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`

        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`

        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]
        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # phi = torch.mean(gamma, dim=0)

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True, eps=1e-12):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.cholesky_inverse(torch.cholesky(cov_mat))
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        if exp_term.ndim == 1:
            exp_term = exp_term.unsqueeze(0)
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def score(self, sample: torch.Tensor):
        _, _, _, z, _ = self.model(sample)
        return self.estimate_sample_energy(z)


class MemAETrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super(MemAETrainer, self).__init__(**kwargs)
        self.alpha = 2e-4
        self.recon_loss_fn = nn.MSELoss().to(self.device)
        self.entropy_loss_fn = EntropyLoss().to(self.device)

    def train_iter(self, sample: torch.Tensor, **kwargs):
        x_hat, w_hat = self.model(sample)
        R = self.recon_loss_fn(sample, x_hat)
        E = self.entropy_loss_fn(w_hat)
        return R + (self.alpha * E)

    def score(self, sample: torch.Tensor):
        x_hat, _ = self.model(sample)
        return torch.sum((sample - x_hat) ** 2, axis=1)


class SOMDAGMMTrainer(BaseTrainer):

    def before_training(self, dataset: DataLoader):
        self.train_som(dataset.dataset.dataset.X)

    def train_som(self, X):
        self.model.train_som(X)

    def train_iter(self, X, **kwargs):
        # SOM-generated low-dimensional representation
        code, X_prime, cosim, Z, gamma = self.model(X)

        phi, mu, Sigma = self.model.compute_params(Z, gamma)
        energy, penalty_term = self.model.estimate_sample_energy(Z, phi, mu, Sigma)

        return self.model.compute_loss(X, X_prime, energy, penalty_term)

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()

        with torch.no_grad():
            scores, y_true = [], []
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()

                sample_energy, _ = self.score(X)

                y_true.extend(y)
                scores.extend(sample_energy.cpu().numpy())

            return np.array(y_true), np.array(scores)

    def score(self, sample: torch.Tensor):
        code, x_prime, cosim, z, gamma = self.model(sample)
        phi, mu, cov_mat = self.model.compute_params(z, gamma)
        sample_energy, pen_cov_mat = self.model.estimate_sample_energy(
            z, phi, mu, cov_mat, average_energy=False
        )
        return sample_energy, pen_cov_mat
