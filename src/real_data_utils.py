import logging
import os.path as osp

import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats
import scipy.stats as st
import torch
from tqdm import tqdm

from data_utils import split_set_with_preprocess
from model_utils import eval_performance

logger = logging.getLogger(__name__)


class RealDataExperiment:

    def __init__(self, x_all, y_all, dataset_name: str, num_splits: int, m_over_n_cfg,
                 net_params,
                 max_set_size: int,
                 max_test_set_size: int, val_train_ratio: float, out_dir: str):

        # Result for M/N experiment
        self.res_dicts = []

        # Experiment parameters
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.num_splits = num_splits
        self.max_test_set_size = max_test_set_size
        self.net_params = net_params

        # Load data
        self.x_all = x_all if len(x_all) < max_set_size else x_all[:max_set_size]
        self.y_all = y_all if len(y_all) < max_set_size else y_all[:max_set_size]
        self.N, self.M = self.x_all.shape
        logger.info(f'{self.dataset_name}: [N M]=[{self.N} {self.M}]')

        # Set the training set size to evaluate
        self.train_sizes, self.val_sizes, self.test_sizes = self.calc_set_sizes(m_over_n_cfg, val_train_ratio,
                                                                                self.M, self.N)
        self.m_over_n_list = self.M / self.train_sizes
        logger.info('train_sizes={}'.format(self.train_sizes))
        logger.info('val_sizes={}'.format(self.val_sizes))
        logger.info('test_sizes={}'.format(self.test_sizes))
        logger.info('M/N={}'.format(self.m_over_n_list.round(2)))

    def calc_set_sizes(self, m_over_n_cfg, val_train_ratio, M, N):
        # Set the training set size to evaluate
        m_over_n_list = np.linspace(m_over_n_cfg.min_m_over_n,
                                    m_over_n_cfg.max_m_over_n,
                                    # M / m_over_n_cfg.min_n_in_m_over_n,
                                    m_over_n_cfg.num_m_over_n)
        train_sizes = M / m_over_n_list
        train_sizes = np.append(train_sizes, M) if m_over_n_cfg.is_add_m_equal_n is True else train_sizes
        train_sizes = np.unique(np.round(train_sizes).astype(int))[::-1]

        val_sizes = np.maximum(np.array([n * val_train_ratio for n in train_sizes]).round().astype(int), 2)
        test_sizes = N - (train_sizes + val_sizes)
        is_valid = np.where((test_sizes > 1) & (train_sizes > 1))[0]
        return train_sizes[is_valid], val_sizes[is_valid], test_sizes[is_valid]

    def execute_split(self, train_set_size: int, val_set_size: int, test_set_size: int) -> dict:
        x_train, y_train, x_val, y_val, x_test, y_test = split_set_with_preprocess(self.x_all, self.y_all,
                                                                                   train_set_size, val_set_size,
                                                                                   test_set_size)

        # Evaluate split: calc mse for specific lambdas
        x_train, y_train, = torch.from_numpy(x_train), torch.from_numpy(y_train)
        x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
        erm_train, erm_test, net_train, net_test, net = eval_performance(x_train, y_train,
                                                                         x_test, y_test,
                                                                         self.net_params.intermediate_layer_ratio,
                                                                         self.net_params.lr,
                                                                         self.net_params.milestones,
                                                                         self.net_params.epochs,
                                                                         self.net_params.is_debug,
                                                                         self.net_params.print_interval)

        split_res_dict = {'erm_train_mse': erm_train, 'net_train_mse': net_train,
                          'erm_mse': erm_test, 'net_mse': net_test}
        return split_res_dict

    def execute_cross_validation(self):

        for train_size, val_size, test_size in tqdm(zip(self.train_sizes, self.val_sizes, self.test_sizes),
                                                    total=len(self.m_over_n_list)):
            # Evaluate train-test split
            result_list = [self.execute_split(train_size, val_size, test_size) for _ in tqdm(range(self.num_splits))]

            # Get statistics
            res_dict = {}
            keys = list(result_list[0].keys())
            for key in keys:
                values = np.array([split_res_dict[key] for split_res_dict in result_list])
                res_dict[key] = values.mean(axis=0)
                res_dict[key + '_median'] = np.median(values, axis=0)
                res_dict[key + '_sem'] = scipy.stats.sem(values, axis=0)

                # Compute confidence interval at 95%
                lower, upper = st.t.interval(alpha=0.95, df=test_size * len(values) - 1, loc=res_dict[key],
                                             scale=res_dict[key + '_sem'])
                res_dict[key + '_ci_lower'] = lower
                res_dict[key + '_ci_upper'] = upper

            res_dict['test_size'] = test_size * self.num_splits
            res_dict['val_size'] = val_size * self.num_splits
            self.res_dicts.append(res_dict)

    def get_res_values(self, key):
        return np.array([res_dict[key] for res_dict in self.res_dicts])

    def get_res_dict_without_arr(self):
        keys = [key for key, val in self.res_dicts[0].items() if not isinstance(val, np.ndarray)]
        return {key: self.get_res_values(key) for key in keys}

    def aggregate_results(self):
        df_dict = self.get_res_dict_without_arr()
        df_dict['m_over_n'] = self.m_over_n_list
        df_dict['M'] = self.M
        df_dict['N'] = self.N

        df = pd.DataFrame(df_dict)
        df_out_path = osp.join(self.out_dir, self.dataset_name + '.csv')
        df.to_csv(df_out_path)
