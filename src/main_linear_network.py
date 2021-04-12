import logging

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from data_utils import PolynomialDataset, add_test_sample_to_dataset
from lighting_utils import LitLinearNet

logger = logging.getLogger(__name__)


@hydra.main(config_name="linear_network", config_path='../configs')
def execute_linear_network(cfg):
    logger.info(cfg.pretty())

    # Trainset
    trainset = PolynomialDataset(cfg.x_train, cfg.y_train, cfg.model_degree)

    # Testset
    x_test = np.arange(cfg.x_test_min, cfg.x_test_max, cfg.dx_test)
    y_test = np.empty(len(x_test))
    testset = PolynomialDataset(x_test, y_test, cfg.model_degree)

    # Loaders
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=cfg.batch_size, num_workers=cfg.num_workers)
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    logger.info('Created dataset. len [trainset testset]=[{} {}]'.format(len(trainset), len(testset)))

    # ERM
    linear_net = LitLinearNet(cfg)
    trainer = pl.Trainer(callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=cfg.epochs, min_epochs=cfg.epochs - 1,
                         fast_dev_run=cfg.fast_dev_run)
    trainer.fit(linear_net, trainloader)
    trainer.save_checkpoint("model.ckpt")

    # pNML
    is_first = True
    y_vec = np.linspace(cfg.y_min, cfg.y_max, cfg.y_num)
    for i, x_test_i in enumerate(x_test):
        # predict erm
        phi_test_i = trainset.convert_point_to_features(x_test_i, cfg.model_degree)
        phi_test_i = torch.from_numpy(phi_test_i).unsqueeze(0)
        y_hat_i = linear_net(phi_test_i).cpu().numpy().squeeze()
        y_vec_i = y_hat_i + y_vec

        # translate vector
        for j, y_test_ij in enumerate(y_vec_i):
            add_test_sample_to_dataset(trainset, x_test, y_test_ij, cfg.insert_loc, is_replace_sample=i == 0 and j == 0)

            # ERM
            linear_net_ij = LitLinearNet(cfg)
            trainer = pl.Trainer(callbacks=[LearningRateMonitor(logging_interval='epoch')],
                                 max_epochs=cfg.epochs, min_epochs=cfg.epochs - 1,
                                 fast_dev_run=cfg.fast_dev_run)
            trainer.fit(linear_net_ij, trainloader)
            linear_net_ij.predict_testset(trainloader, testloader)


if __name__ == '__main__':
    execute_linear_network()
