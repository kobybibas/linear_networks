import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from data_utils import create_synthetic_dataset
from model_utils import eval_performance

logger = logging.getLogger(__name__)


@hydra.main(config_name="high_dim_synthetic_set", config_path='../configs')
def execute_synthetic_data_regression(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(os.getcwd())
    torch.manual_seed(cfg.seed)

    # Create set
    x_train, y_train, x_test, y_test, theta_star = create_synthetic_dataset(cfg.n_samples, cfg.n_features,
                                                                            cfg.effective_rank,
                                                                            cfg.seed)
    torch.save((x_train, y_train, x_test, y_test), 'dataset.pth')
    torch.save(theta_star, 'theta_star.pth')
    logger.info("shape x [train test]=[{} {}] y [train test]. rank={}".format(x_train.shape, y_train.shape,
                                                                              x_test.shape, y_test.shape,
                                                                              torch.matrix_rank(x_train).item()))

    # Iterate on different training set size
    res_dict_list = []
    n_trains = eval(cfg.n_trains)
    n_trains = np.round(n_trains).astype(int)
    logger.info(f'n_trains={n_trains}')
    for i, n_train in enumerate(n_trains):
        # Create training set
        x_train_n, y_train_n = x_train[:n_train], y_train[:n_train]

        # Execute
        (mse_train_mn, mse_test_mn,
         mse_train_net, mse_test_net,
         net) = eval_performance(x_train_n, y_train_n,
                                 x_test, y_test,
                                 cfg.intermediate_layer_ratio,
                                 cfg.lr, cfg.milestones, cfg.epochs,
                                 is_debug=True, print_interval=cfg.print_interval)

        # Save result
        res_dict = {'n_train': n_train,
                    'mse_test_net': mse_test_net, 'mse_test_mn': mse_test_mn,
                    'mse_train_net': mse_train_net, 'mse_train_mn': mse_train_mn,
                    'input_size': net.input_size,
                    'output_size': net.output_size,
                    'intermediate_sizes': net.intermediate_sizes}
        res_dict_list.append(res_dict)

        # Print
        logger.info('[{}/{}] training set size = {}'.format(i, len(n_trains) - 1, x_train_n.shape))
        logger.info("MSE train: [MN net]=[{} {}]".format(mse_train_mn, mse_train_net))
        logger.info("MSE test:  [MN net]=[{} {}]".format(mse_test_mn, mse_test_net))
        torch.save(net.state_dict(), f'net_n_train={n_train}.pth')
        logger.info('')
    df = pd.DataFrame(res_dict_list)
    df.to_csv('result.csv')
    logger.info(os.getcwd())


if __name__ == "__main__":
    execute_synthetic_data_regression()
