import logging

import numpy as np
import numpy.linalg as npl
import torch
import torch.linalg as tln
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class LinearNet(torch.nn.Module):
    def __init__(self, input_size, output_size, intermediate_sizes: list = None):
        super(LinearNet, self).__init__()
        self.input_size, self.output_size = input_size, output_size
        self.intermediate_sizes = intermediate_sizes
        if intermediate_sizes is None or len(intermediate_sizes) == 0:
            layer_sizes = [input_size, output_size]
        else:
            layer_sizes = [input_size] + list(intermediate_sizes) + [output_size]

        layers = []
        for i in range(1, len(layer_sizes)):
            input_size_i, output_size_i = layer_sizes[i - 1], layer_sizes[i]
            bias = False if i == 1 else False  # no bias for the first layer, we have it in the features
            linear = torch.nn.Linear(input_size_i, output_size_i, bias=bias)
            layers.append(linear)
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        y_hat = self.layers(x)
        return y_hat

    def get_features(self, x) -> list:
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x.detach().cpu())
        return features


def train_model(net, x_train, y_train, epochs, criterion, optimizer, scheduler,
                is_print: bool = True, print_interval: int = 1000):
    y_hat = net(x_train)
    loss_init, loss = criterion(y_hat.squeeze(), y_train.squeeze()), 0

    net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_hat = net(x_train)
        loss = criterion(y_hat.squeeze(), y_train.squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()

        if is_print and epoch % print_interval == 0:
            log = "[{}/{}] loss={}".format(epoch, epochs - 1, loss.item())
            logger.info(log)
    log = "\nloss [init last]=[{} {}]".format(loss_init.item(), loss.item())
    logger.info(log)
    net.eval()


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


def eval_performance(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor,
                     intermediate_layer_ratio: list, lr: float, milestones: list, epochs: int,
                     is_debug: bool = False, print_interval: int = 1000):
    n_train, n_features = x_train.shape

    criterion = torch.nn.MSELoss(reduction="mean")
    intermediate_sizes = [int(round(n_features * ratio_i)) for ratio_i in intermediate_layer_ratio]
    input_size, output_size = n_features, 1

    # Iterate on different training set size
    net = LinearNet(input_size=n_features, output_size=output_size, intermediate_sizes=intermediate_sizes)
    net = net.double()
    for layer in net.layers:
        layer.weight.data.uniform_(-1e-4, 1e-4)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    train_model(net, x_train, y_train, epochs, criterion, optimizer, scheduler,
                is_print=is_debug, print_interval=print_interval)

    # Test net
    with torch.no_grad():
        y_hat_test_net = net(x_test)
        mse_test_net = criterion(y_hat_test_net.squeeze(), y_test.squeeze()).item()
        y_hat_train_net = net(x_train)
        mse_train_net = criterion(y_hat_train_net.squeeze(), y_train.squeeze()).item()

    # Train ERM
    pn = tln.pinv(x_train.T @ x_train) @ x_train.T
    theta_mn = pn @ y_train

    # Performance
    y_hat_test_mn = x_test @ theta_mn
    mse_test_mn = criterion(y_hat_test_mn, y_test).item()
    y_hat_train_mn = x_train @ theta_mn
    mse_train_mn = criterion(y_hat_train_mn.squeeze(), y_train.squeeze()).item()

    return mse_train_mn, mse_test_mn, mse_train_net, mse_test_net, net
