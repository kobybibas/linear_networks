import logging
import os
import os.path as osp

import numpy as np
import numpy.linalg as npl
import torch
import torch.linalg as tln
from pmlb import fetch_data
from sklearn.datasets import make_low_rank_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

pmlb_set_names_all = ['1028_SWD', '1201_BNG_breastTumor', '192_vineyard', '201_pol',
                      '210_cloud', '215_2dplanes', '1029_LEV', '218_house_8L', '225_puma8NH',
                      '228_elusage', '229_pwLinear', '230_machine_cpu',
                      '294_satellite_image', '344_mv', '4544_GeographicalOriginalofMusic',
                      '1030_ERA', '519_vinnie', '522_pm10', '523_analcatdata_neavote',
                      '527_analcatdata_election2000', '529_pollen', '537_houses',
                      '542_pollution', '1096_FacultySalaries', '1191_BNG_pbc',
                      '1193_BNG_lowbwt', '1196_BNG_pharynx', '1199_BNG_echoMonths']

pmlb_set_names = ["1028_SWD", "1030_ERA", "1196_BNG_pharynx",
                  "1199_BNG_echoMonths", "1201_BNG_breastTumor", "215_2dplanes",
                  "218_house_8L", "225_puma8NH", "229_pwLinear",
                  "344_mv", "522_pm10", "537_houses", "542_pollution"]


def calc_effective_rank(s):
    s_norm = tln.norm(s, ord=1)
    p = s / s_norm
    effective_rank = torch.exp(
        torch.sum(torch.tensor([-p_i * torch.log(p_i) for p_i in p if p_i > 0]))
    )
    return effective_rank


def add_test_to_train(x_train, y_train, x_test, y_test, model_degree, insert_loc: int = 0):
    x_all = np.insert(x_train, insert_loc, x_test, axis=0)
    y_all = np.insert(y_train, insert_loc, y_test, axis=0)
    trainset = PolynomialDataset(x_all, y_all, model_degree)
    return trainset


def add_test_sample_to_dataset(dataset, x_test, y_test, insert_loc: int = 0, is_replace_sample: bool = False):
    phi_test = dataset.convert_point_to_features(x_test)

    if is_replace_sample is True:
        # Replace exiting sample
        dataset.x[insert_loc] = x_test
        dataset.phi_train[insert_loc] = phi_test
        dataset.y[insert_loc] = y_test
    else:
        # Create new entry
        dataset.x = np.insert(dataset.x, insert_loc, x_test, axis=0)
        dataset.y = np.insert(dataset.y, insert_loc, y_test, axis=0)
        dataset.phi_train = np.insert(dataset.phi_train, insert_loc, phi_test, axis=0)


class PolynomialDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x_train: list, y_train: list, model_degree: int):
        # Model degree, proportional to the learn-able parameters
        self.model_degree = model_degree

        # Generate the training data.
        self.x = np.array(x_train)
        self.y = np.array(y_train).astype(np.float32)

        # Matrix of training feature [phi0;phi1;phi2...]. phi is the features phi(x0)
        self.phi_train = self.create_train_features()
        self.theta_erm = fit_least_squares_estimator(self.phi_train, self.y)
        logger.info('self.phi_train.shape: {}'.format(self.phi_train.shape))

    def __len__(self):
        return len(self.y)

    def convert_point_to_features(self, x: float, pol_degree: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param pol_degree: the assumed polynomial degree.
        :return: phi = [x0^0,x0^1,x0^2,...], row vector
        """
        ns = np.arange(0, pol_degree, 1)
        phi = np.power(x, ns)
        phi = np.expand_dims(phi, 1)
        return phi

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        phi_train = self.phi_train[idx]
        y = self.y[idx]

        return phi_train, y

    def create_train_features(self) -> np.ndarray:
        """
        Convert data points to feature matrix: phi=[x0^0,x0^1,x0^2...;x1^0,x1^1,x1^2...;x2^0,x2^1,x2^2...]
        Each row corresponds to feature vector.
        :return: phi: training set feature matrix.
        """
        phi_train = []
        for x_i in self.x:
            phi_train_i = self.convert_point_to_features(x_i, self.model_degree)

            # Convert column vector to raw and append
            phi_train.append(np.squeeze(phi_train_i.T))
        phi_train = np.asarray(phi_train)
        return phi_train.astype(np.float32)


def get_pmlb_data(dataset_name, data_dir: str) -> (np.ndarray, np.ndarray):
    pmlb_data_dir = osp.join(data_dir, 'pmlb_datasets')
    os.makedirs(pmlb_data_dir, exist_ok=True)
    x_arr, y_vec = fetch_data(dataset_name, return_X_y=True, local_cache_dir=pmlb_data_dir)
    return x_arr, y_vec


def split_set_with_preprocess(x_all, y_all, train_set_size: int, val_set_size: int, test_set_size: int):
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,
                                                        test_size=test_set_size + val_set_size,
                                                        train_size=train_set_size)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_set_size)
    x_train, x_val, x_test = convert_to_row(x_train), convert_to_row(x_val), convert_to_row(x_test)

    # Preprocess
    x_train, x_val, x_test = standardize_features(x_train, x_val, x_test)
    y_train, y_val, y_test = standardize_features(y_train, y_val, y_test)
    y_train, y_val, y_test = y_train.squeeze(), y_val.squeeze(), y_test.squeeze()
    x_train, x_val, x_test = add_bias_term(x_train), add_bias_term(x_val), add_bias_term(x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def convert_to_row(x):
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    return x


def convert_to_column(x):
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    return x


def add_bias_term(x_arr):
    x_arr_with_bias = np.hstack((x_arr, np.ones((x_arr.shape[0], 1))))
    return x_arr_with_bias


def normalize_set(x_arr):
    x_arr_norm = x_arr / npl.norm(x_arr, axis=1, keepdims=True)
    return x_arr_norm


def standardize_features(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> (
        np.ndarray, np.ndarray, np.ndarray):
    x_train, x_val, x_test = convert_to_column(x_train), convert_to_column(x_val), convert_to_column(x_test)

    # Learn training statistics
    scaler_h = StandardScaler().fit(x_train)

    # Apply on sets
    x_train_stand = scaler_h.transform(x_train)
    x_val_stand = scaler_h.transform(x_val)
    x_test_stand = scaler_h.transform(x_test)
    return x_train_stand, x_val_stand, x_test_stand


def create_synthetic_dataset(n_samples, n_features, effective_rank, seed):
    vecs = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=effective_rank,
                                tail_strength=0.01, random_state=seed)
    x_arr = torch.from_numpy(vecs)
    x_arr = torch.cat((x_arr, torch.ones(n_samples, 1)), dim=1)  # add bias
    theta_gt = torch.randn(n_features + 1, 1).double()
    y_vec = x_arr @ theta_gt

    n_set = int(round(n_samples / 2))
    x_train, y_train = x_arr[:n_set], y_vec[:n_set]
    x_test, y_test = x_arr[n_set:], y_vec[n_set:]
    return x_train, y_train, x_test, y_test, theta_gt
