import logging

import numpy as np
import numpy.linalg as npl
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def fit_least_squares_estimator(x_arr: np.ndarray, y_vec: np.ndarray, lamb: float = 0.0) -> np.ndarray:
    """
    Fit least squares estimator
    :param x_arr: The training set features matrix. Each row represents an example.
    :param y_vec: the labels vector.
    :param lamb: regularization term.
    :return: the fitted parameters. A column vector
    """
    n, m = x_arr.shape
    phi_t_phi_plus_lamb = x_arr.T @ x_arr + lamb * np.eye(m)

    # If invertible, regular least squares
    if npl.cond(phi_t_phi_plus_lamb) < 1 / np.finfo('float').eps:
        inv = npl.inv(phi_t_phi_plus_lamb)
        theta = inv @ x_arr.T @ y_vec
    else:  # minimum norm
        # inv = npl.pinv(x_arr @ x_arr.T)
        # theta = x_arr.T @ inv @ y_vec
        reg = LinearRegression(fit_intercept=False).fit(x_arr, y_vec)  # using scipy is more stable
        theta = reg.coef_

    theta = np.expand_dims(theta, 1)
    return theta


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
