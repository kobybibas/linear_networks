import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import PolynomialDataset
from lighting_utils import LitLinearNet
from pnml_utils import PnmlAnalytical

logger = logging.getLogger(__name__)


@hydra.main(config_name="defaults", config_path='../configs')
def execute_linear_network(cfg):
    logger.info(cfg.pretty())

    # Training sets
    trainset = PolynomialDataset(cfg.x_train, cfg.y_train, cfg.model_degree)

    # Testing set
    x_test = np.arange(cfg.x_test_min, cfg.x_test_max, cfg.dx_test)
    y_test = np.empty(len(x_test))
    testset = PolynomialDataset(x_test, y_test, cfg.model_degree)

    # Loaders
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=cfg.batch_size, num_workers=cfg.num_workers)
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    logger.info('Created dataset. len [trainset testset]=[{} {}]'.format(len(trainset), len(testset)))

    # ERM
    linear_net = LitLinearNet.load_from_checkpoint(cfg.ptretrained_path)

    # Get trainset features
    x0_train = []
    x1_train = []
    with torch.no_grad():
        for x, _ in trainloader:
            features_list = linear_net.model.get_features(x)
            x0_train.append(features_list[0].cpu().detach())
            x1_train.append(features_list[1].cpu().detach())

    # First layer
    x0_train = torch.cat(x0_train, dim=0)  # Each raw is a sample
    x1_train = torch.cat(x1_train, dim=0)
    logger.info('Shapes: [x0_train x1_train]=[{} {}]'.format(x0_train.shape, x1_train.shape))

    pnml_h = PnmlAnalytical(x0_train, x1_train)

    y_hat_list, nfs, intermediate_dicts = [], [], []
    with torch.no_grad():
        for x, _ in testloader:
            features_list = linear_net.model.get_features(x)
            x0s, x1s, y_hats = features_list[0], features_list[1], features_list[2]

            for i, (x0, x1, y_hat) in enumerate(zip(x0s, x1s, y_hats)):
                nf, intermediate_dict = pnml_h.calc_nf(x0, x1)

                nfs.append(nf)
                intermediate_dicts.append(intermediate_dict)
                y_hat_list.append(y_hat.item())
    y_hats = y_hat_list
    regrets = np.log(nfs)
    regrets0 = np.log([d['k0'] for d in intermediate_dicts])
    keys = list(intermediate_dicts[0].keys())

    # SGD
    fig, axs = plt.subplots(2, 1, sharex=True)

    ax = axs[0]
    ax.plot(testloader.dataset.x, y_hats, label='SGD')

    # Analytical
    trainset, testset = trainloader.dataset, testloader.dataset
    y_hats_analytical = testset.phi_train @ trainset.theta_erm
    ax.plot(testset.x, y_hats_analytical, label='Analytical 1 layer')
    ax.plot(cfg.x_train, cfg.y_train, 'r*', label='Training')
    ax.set_ylabel('y')
    ax.set_title('Model degree = {}'.format(cfg.model_degree))
    ax.grid()
    ax.legend()
    ax.set_ylim(-2,2)

    ax = axs[1]
    ax.plot(testloader.dataset.x, regrets, label='2 layer')
    ax.plot(testloader.dataset.x, regrets0, label='1 layer')
    ax.plot(cfg.x_train, [0]*len(cfg.x_train), 'r*', label='Training')
    ax.grid()
    ax.set_ylabel('Regret')

    # Save
    plt.tight_layout()
    plt.savefig('prediction.jpg')
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, sharex=True)
    ax = axs
    for key in keys:
        ax.plot(testloader.dataset.x, [d[key] for d in intermediate_dicts], label=key)
    ax.plot(cfg.x_train, [0]*len(cfg.x_train), 'r*', label='Training')
    ax.grid()
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('additional')
    ax.set_yscale('symlog')
    plt.tight_layout()
    plt.savefig('Additional.jpg')
    plt.close(fig)


if __name__ == '__main__':
    execute_linear_network()
