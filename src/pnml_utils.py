import logging

import torch

logger = logging.getLogger(__name__)


class PnmlAnalytical:
    def __init__(self, x0_train, x1_train):
        self.x0_train, self.x1_train = x0_train, x1_train

        corr0 = x0_train.T @ x0_train
        corr1 = x1_train.T @ x1_train

        u0, h0, _ = torch.svd(corr0)
        u1, h1, _ = torch.svd(corr1)
        logger.info(f'h0={h0}')
        logger.info(f'h1={h1}')

        self.u0, self.h0 = u0, h0
        self.u1, self.h1 = u1, h1

        mu = torch.mean(x1_train, dim=0, keepdim=True).T  # make column vector
        Sigma1 = corr1 - mu @ mu.T

        self.c = (torch.diag(u1.T @ Sigma1 * u1) / h1).sum().item()

    def calc_nf(self, x0, x1):

        # Make column vector
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(1)
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(1)  # make column vector

        m0, m1 = x0.shape[0], x1.shape[0]

        P0_proj = (torch.square(self.u0.T @ x0).squeeze() / self.h0).sum()
        P1_proj = (torch.square(self.u1.T @ x1).squeeze() / self.h1).sum()

        k0 = 1 + P0_proj
        k1 = 1 + P1_proj

        # calc c
        x1_all = self.x1_train
        # x1_all = torch.vstack((self.x1_train, x1.T)) # todo
        mu = torch.mean(x1_all, dim=0, keepdim=True).T  # make column vector
        Sigma1 = x1_all.T @ x1_all - mu @ mu.T

        import numpy as np
        Sigma1_np = np.cov(x1_all.numpy().T)
        # print(Sigma1_np,Sigma1)
        Sigma1 = Sigma1_np

        diags = torch.diag(self.u1.T @ Sigma1 * self.u1)
        diags[diags < 0] = 0.0
        self.c = (diags / self.h1).sum().item()

        k = (k0 ** m1) * (k1 + self.c)
        k = k1 + (k0**2) * self.c
        k = k.item()

        if k < 1:
            ValueError('k={}<0'.format(k))
            a = 1

        intermediate_dict = {'k0': k0.item(), 'c': self.c,
                             'P0_proj': P0_proj.item(),
                             'P1_proj': P1_proj.item()}

        return k, intermediate_dict
