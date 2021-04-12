import logging
import os

import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data_utils import PolynomialDataset
from lighting_utils import LitLinearNet

logger = logging.getLogger(__name__)


@hydra.main(config_name="defaults", config_path='../configs')
def train_erm_model(cfg):
    logger.info(cfg.pretty())
    logger.info(os.getcwd())
    tb_logger = TensorBoardLogger(save_dir=os.getcwd())
    pl.seed_everything(cfg.seed)

    # Trainset
    trainset = PolynomialDataset(cfg.x_train, cfg.y_train, cfg.model_degree)

    # Testset
    x_test = np.arange(cfg.x_test_min, cfg.x_test_max, cfg.dx_test)
    y_test = np.empty(len(x_test))
    testset = PolynomialDataset(x_test, y_test, cfg.model_degree)

    # Loaders
    trainloader = DataLoader(trainset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
    testloader = DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.num_workers)
    logger.info('Created dataset. len [trainset testset]=[{} {}]'.format(len(trainset), len(testset)))

    # ERM
    linear_net = LitLinearNet(cfg)
    trainer = pl.Trainer(callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=cfg.epochs, min_epochs=cfg.epochs - 1,
                         fast_dev_run=cfg.fast_dev_run, num_sanity_val_steps=0,
                         logger=tb_logger, default_root_dir=os.getcwd())
    trainer.fit(linear_net, trainloader, testloader)
    trainer.save_checkpoint("model.ckpt")
    logger.info(f'Finish out_dir={os.getcwd()}')


if __name__ == '__main__':
    train_erm_model()
