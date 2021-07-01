import logging
import os
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

from data_utils import get_pmlb_data, pmlb_set_names
from real_data_utils import RealDataExperiment

logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name="real_data_set")
def execute_real_data_set(cfg):
    t0 = time.time()
    logger.info(OmegaConf.to_yaml(cfg))
    out_path = os.getcwd()
    np.random.seed(seed=1234)

    # Move from output directory to src
    os.chdir(hydra.utils.get_original_cwd())
    logger.info('[cwd out_dir]=[{} {}]'.format(os.getcwd(), out_path))

    # Get available datasets
    dataset_names = pmlb_set_names if cfg.set_name == '' else [cfg.set_name]
    logger.info(f'dataset_names={dataset_names}')

    # Evaluate dataset
    for i, dataset_name in enumerate(dataset_names):
        logger.info(f'[{i}/{len(dataset_names) - 1}] {dataset_name}')
        x_all, y_all = get_pmlb_data(dataset_name, cfg.data_dir)
        experiment_h = RealDataExperiment(x_all, y_all, dataset_name=dataset_name, num_splits=cfg.num_splits,
                                          net_params=cfg.net_params,
                                          m_over_n_cfg=cfg.m_over_n_cfg,
                                          max_set_size=cfg.max_set_size, max_test_set_size=cfg.max_test_set_size,
                                          val_train_ratio=cfg.val_train_ratio, out_dir=out_path)
        experiment_h.execute_cross_validation()
        experiment_h.aggregate_results()

    # Save
    logger.info('Finish! in {:.2f} sec. {}'.format(time.time() - t0, out_path))


if __name__ == "__main__":
    execute_real_data_set()
