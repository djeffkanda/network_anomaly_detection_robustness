import numpy as np
import pandas as pd
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple

from utils.utils import random_split_to_two


class AbstractDataset(Dataset):
    def __init__(self, path: str, pct: float = 1.0, **kwargs):
        self.name = self.__class__.__name__

        self.labels = None
        X = self._load_data(path)

        self.X, self.y, self.anomaly_ratio = self.select_data_subset(pct, X, **kwargs)

        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

        self.normal_train_idx = []
        self.test_idx = []
        self.validation_idx = []

    def select_data_subset(self, pct, X, **kwargs):
        anomaly_label = kwargs.get('anomaly_label', 1)
        normal_label = kwargs.get('normal_label', 0)
        if pct < 1.0:
            # Keeps `pct` percent of the original data while preserving
            # the normal/anomaly ratio
            anomaly_idx = np.where(X[:, -1] == anomaly_label)[0]
            normal_idx = np.where(X[:, -1] == normal_label)[0]
            np.random.shuffle(anomaly_idx)
            np.random.shuffle(normal_idx)

            X = np.concatenate(
                (X[anomaly_idx[:int(len(anomaly_idx) * pct)]],
                 X[normal_idx[:int(len(normal_idx) * pct)]])
            )

            if not self.labels is None:
                self.labels = np.concatenate(
                    (self.labels[anomaly_idx[:int(len(anomaly_idx) * pct)]],
                     self.labels[normal_idx[:int(len(normal_idx) * pct)]])
                )

        return X[:, :-1], X[:, -1].astype(int), (X[:, -1] == anomaly_label).sum() / len(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], index

    def _load_data(self, path: str):
        if path.endswith(".npz"):
            return np.load(path)[self.npz_key()]
        elif path.endswith(".mat"):
            data = scipy.io.loadmat(path)
            X = np.concatenate((data['X'], data['y']), axis=1)
            return X
        else:
            raise RuntimeError(f"Could not open {path}. Dataset can only read .npz and .mat files.")

    def D(self):
        return self.X.shape[1]

    def shape(self):
        return self.X.shape

    def get_data_index_by_label(self, label):
        return np.where(self.y == label)[0]

    def scale_values(self):
        pass

    def loaders(self,
                test_pct: float = 0.5,
                label: int = 0,
                holdout: float = 0.0,
                contamination_rate: float = 0.0,
                validation_ratio: float = .2,
                batch_size: int = 128,
                num_workers: int = 0,
                seed: int = None,
                drop_last_batch: bool = False, **kwargs) -> (DataLoader, DataLoader, DataLoader):

        train_set, test_set, neg_val_set, val_set = self.split_train_test(test_pct=test_pct,
                                                                          label=label,
                                                                          holdout=holdout,
                                                                          contamination_rate=contamination_rate,
                                                                          validation_ratio=validation_ratio,
                                                                          seed=seed,
                                                                          )

        train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers,
                               drop_last=drop_last_batch)

        neg_val_ldr = DataLoader(dataset=neg_val_set, batch_size=kwargs.get('batch_size_test', None) or batch_size,
                                 num_workers=num_workers,
                                 drop_last=drop_last_batch,
                                 shuffle=False)

        test_ldr = DataLoader(dataset=test_set, batch_size=kwargs.get('batch_size_test', None) or batch_size,
                              num_workers=num_workers,
                              drop_last=drop_last_batch, shuffle=False)

        val_ldr = DataLoader(dataset=val_set, batch_size=kwargs.get('batch_size_test', None) or batch_size,
                             num_workers=num_workers,
                             drop_last=drop_last_batch,
                             shuffle=False)

        return train_ldr, test_ldr, neg_val_ldr, val_ldr

    def split_train_test(self, test_pct: float = .5,
                         label: int = 0,
                         holdout=0.05,
                         contamination_rate=0.01,
                         validation_ratio: float = .2,
                         seed=None, debug=True, corruption_label=None, **kwargs) -> Tuple[
        Subset, Subset, Subset, Subset]:
        assert (label == 0 or label == 1)
        assert 1 > holdout  # >=
        assert 0 <= contamination_rate <= 1

        # if seed:
        #     torch.manual_seed(seed)

        # Fetch and shuffle indices of a single class
        normal_data_idx = np.where(self.y == label)[0]
        shuffled_norm_idx = torch.randperm(len(normal_data_idx)).long()

        # Generate training set indices
        num_norm_train_sample = int(len(normal_data_idx) * (1. - test_pct))
        normal_train_idx = normal_data_idx[shuffled_norm_idx[:num_norm_train_sample]]

        #
        abnormal_data_idx = np.where(self.y == int(not label))[0]
        abnorm_test_idx = abnormal_data_idx

        if debug:
            print(f"Dataset size\nPositive class :{len(abnormal_data_idx)}"
                  f"\nNegative class :{len(normal_data_idx)}\n")

        if holdout > 0:
            # Generate test set by holding out a percentage [holdout] of abnormal data
            # sample for a possible contamination
            shuffled_abnorm_idx = torch.randperm(len(abnormal_data_idx)).long()
            num_abnorm_test_sample = int(len(abnormal_data_idx) * (1 - holdout))
            abnorm_test_idx = abnormal_data_idx[shuffled_abnorm_idx[:num_abnorm_test_sample]]

            if contamination_rate > 0:
                # num_abnorm_to_inject = int(len(shuffled_abnorm_idx[
                #                                num_abnorm_test_sample:]) * contamination_rate)
                holdout_ano_idx = abnormal_data_idx[shuffled_abnorm_idx[num_abnorm_test_sample:]]

                # Injection of only specified type of attacks
                if corruption_label:
                    all_labels = np.char.lower(self.labels[holdout_ano_idx].astype('str'))
                    corruption_label = corruption_label.lower()
                    corruption_by_lbl_idx = np.char.startswith(all_labels,
                                                               corruption_label)
                    holdout_ano_idx = holdout_ano_idx[corruption_by_lbl_idx]

                # Calculate the number of abnormal samples to inject
                # according to the contamination rate
                num_abnorm_to_inject = int(normal_train_idx.shape[0] * contamination_rate / (1 - contamination_rate))

                print(f"\nMax contamination ratio:{len(holdout_ano_idx)/(len(holdout_ano_idx)+normal_train_idx.shape[0])}"
                      f"\n#toinject : {num_abnorm_to_inject}, "
                      f"holdout_size:{len(holdout_ano_idx)}, \n"
                      )
                assert num_abnorm_to_inject <= len(holdout_ano_idx)

                normal_train_idx = np.concatenate([
                    holdout_ano_idx[:num_abnorm_to_inject],
                    normal_train_idx
                ])

        # Generate training set with contamination when applicable
        # Split the training set to train and validation
        normal_train_idx, normal_val_idx = random_split_to_two(normal_train_idx, ratio=validation_ratio)

        train_set = Subset(self, normal_train_idx)
        neg_val_set = Subset(self, normal_val_idx)
        if debug:
            print(
                f'Training set\n'
                f'Contamination rate: '
                f'{len(np.where(self.y[normal_train_idx] == int(not label))[0]) / len(normal_train_idx)}\n')

        # Generate test set based on the remaining data and the previously filtered out labels
        remaining_idx = np.concatenate([
            normal_data_idx[shuffled_norm_idx[num_norm_train_sample:]],
            abnorm_test_idx
        ])

        self.normal_train_idx = normal_train_idx
        self.test_idx = remaining_idx
        self.validation_idx = normal_val_idx

        remaining_idx_test, remaining_idx_val = random_split_to_two(remaining_idx, ratio=validation_ratio / 2)
        val_set = Subset(self, remaining_idx_val)
        test_set = Subset(self, remaining_idx_test)

        print(len(self.y), (len(test_set) + len(train_set) + len(neg_val_set)))
        self.scale_values()

        return train_set, test_set, neg_val_set, val_set


class BaseDataset(AbstractDataset):
    def __init__(self, X, y):
        self.name = self.__class__.__name__

        self.labels = None
        self.X, self.y, = X, y

        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]


def generate_data(mean, cov, num_samples):
    return np.random.multivariate_normal(mean, cov, num_samples)


class ToyDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super(ToyDataset, self).__init__(**kwargs)
        self.normal_train_idx = []
        self.test_idx = []
        self.validation_idx = []
        self.name = "Toy"

    def _load_data(self, path: str = None,
                   mean_normal=np.array([1, 1]),
                   cov_normal=0.07 * np.identity(2),
                   num_normal_samples=200,
                   mean_abnormal1=np.array([-0.25, 2.5]),
                   mean_abnormal2=np.array([-1.0, 0.5]),
                   cov_abnormal=0.03 * np.identity(2),
                   num_abnormal_samples=100
                   ):
        # Generate normal and abnormal data
        normal_data = generate_data(mean_normal, cov_normal, num_normal_samples)
        abnormal_data1 = generate_data(mean_abnormal1, cov_abnormal, num_abnormal_samples // 2)
        abnormal_data2 = generate_data(mean_abnormal2, cov_abnormal, num_abnormal_samples // 2)
        abnormal_data = np.vstack([abnormal_data1, abnormal_data2])

        # Combine normal and abnormal data
        all_data = np.vstack([normal_data, abnormal_data])
        labels = np.concatenate([np.zeros(num_normal_samples), np.ones(num_abnormal_samples)])

        X = np.concatenate([all_data, labels.reshape(-1, 1)], axis=1)
        self.labels = labels
        return X

    def npz_key(self):
        return "toy"


class ArrhythmiaDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Arrhythmia"

    def npz_key(self):
        return "arrhythmia"


class IDS2018Dataset(AbstractDataset):

    def __init__(self, **kwargs):
        super(IDS2018Dataset, self).__init__(**kwargs)
        self.normal_train_idx = []
        self.test_idx = []
        self.validation_idx = []
        self.name = "IDS2018"

    def _load_data(self, path: str):
        data = np.load(path, allow_pickle=True)
        X = data[self.npz_key()]
        self.labels = data['label']
        return X

    def npz_key(self):
        return "ids2018"

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], index, self.labels[index]


class IOT2023Dataset(IDS2018Dataset):
    def __init__(self, **kwargs):
        super(IOT2023Dataset, self).__init__(**kwargs)
        self.name = "IOT2023"
        self.X_copy = np.copy(self.X)
        # self.scale_values()

    def _load_data(self, path: str):
        df = pd.read_parquet(path)
        colums = list(df.columns)
        colums.remove('Label_cat')
        X = df[colums].to_numpy()
        self.labels = df['Label_cat'].to_numpy()
        return X

    def scale_values(self):
        scaler = MinMaxScaler()
        # self.X = scaler.fit_transform(self.X_copy)
        self.X[self.normal_train_idx] = scaler.fit_transform(self.X_copy[self.normal_train_idx])
        self.X[self.test_idx] = scaler.transform(self.X_copy[self.test_idx])

        try:
            self.X[self.validation_idx] = scaler.transform(self.X_copy[self.validation_idx])
        except Exception as ex:
            pass


        # self.X[self.normal_train_idx] = scaler.fit_transform(self.X[self.normal_train_idx])
        # self.X[self.test_idx] = scaler.transform(self.X[self.test_idx])

    def npz_key(self):
        return "iot2023"


class KitsuneDataset(IOT2023Dataset):
    def __init__(self, **kwargs):
        super(KitsuneDataset, self).__init__(**kwargs)
        self.name = "Kitsune"

    def _load_data(self, path: str):
        df = pd.read_parquet(path)
        colums = list(df.columns)
        colums.remove('Label_cat')
        X = df[colums].to_numpy()
        self.labels = df['Label_cat'].to_numpy()
        # The first column is the index
        return X[:, 1:]

    def npz_key(self):
        return "kitsune"


class CIC18ImpDataset(KitsuneDataset):
    def __init__(self, **kwargs):
        super(CIC18ImpDataset, self).__init__(**kwargs)
        self.name = "CIC18Imp"

    def _load_data(self, path: str, desired_normal_ratio=.80):
        df = pd.read_parquet(path)
        colums = list(df.columns)
        colums.remove('Src Port')
        colums.remove('Label_cat')
        colums.remove('Label')
        bin_labels = df['Label'].to_numpy()
        X = np.concatenate([df[colums].to_numpy(), bin_labels.reshape(-1, 1)], axis=1)
        anomaly_count = np.sum(bin_labels == 1)

        normal_count_required = anomaly_count * desired_normal_ratio / (1 - desired_normal_ratio)
        X_normal = X[bin_labels == 0]
        X_abnormal = X[bin_labels == 1]
        np.random.shuffle(X_normal)
        X_normal = X_normal[:int(normal_count_required)]
        X = np.concatenate([X_abnormal, X_normal])

        self.labels = df['Label_cat'].to_numpy()
        return X  # [:, 1:]

    def npz_key(self):
        return "CIC18Imp"


class KDD10Dataset(AbstractDataset):
    """
    This class is used to load KDD Cup 10% dataset as a pytorch Dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "KDD10"

    def npz_key(self):
        return "kdd"


class NSLKDDDataset(AbstractDataset):
    """
    This class is used to load NSL-KDD Cup dataset as a pytorch Dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NSLKDD"

    def npz_key(self):
        return "kdd"


class ThyroidDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Thyroid"

    def npz_key(self):
        return "thyroid"


class USBIDSDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "USBIDS"

    def npz_key(self):
        return "usbids"


class MalMem2022Dataset(AbstractDataset):

    def __init__(self, **kwargs):
        super(MalMem2022Dataset, self).__init__(**kwargs)
        self.name = "MalMem2022"

    def npz_key(self):
        return "malmem2022"
