import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from model_utils import LinearNet

logger = logging.getLogger(__name__)


class LitLinearNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = LinearNet(cfg.model_degree, output_size=1, intermediate_sizes=cfg.intermediate_sizes)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        logger.info(f'\n{self.model}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.milestones, gamma=0.1)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.squeeze(), y)
        return loss

    def training_epoch_end(self, outputs):
        loss_total = 0
        for output in outputs:
            loss_total += output['loss'].item()
        self.log('train_loss', loss_total / len(outputs), on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'y_hat': y_hat}

    def validation_epoch_end(self, outputs):
        if self.current_epoch % self.cfg.check_val_every_n_epoch == 0:
            y_hats = torch.vstack([output['y_hat'] for output in outputs]).squeeze()

            # SGD
            val_loader = self.val_dataloader()
            train_loader = self.train_dataloader()

            # Compute analytical
            trainset, testset = train_loader.dataset, val_loader.dataset
            y_hats_analytical = testset.phi_train @ trainset.theta_erm

            # Plot
            fig, ax = plt.subplots(1, 1)
            plt.plot(val_loader.dataset.x, y_hats, label='SGD')
            plt.plot(testset.x, y_hats_analytical, label='Analytical')

            ax.plot(self.cfg.x_train, self.cfg.y_train, 'r*', label='Training')
            ax.set_xlabel('t')
            ax.set_ylabel('y')
            ax.grid()
            ax.legend()
            fig.tight_layout()

            tensorboard = self.logger.experiment
            tensorboard.add_figure(f'Prediction epoch {self.current_epoch}', fig)
            plt.savefig('prediction.jpg')
            plt.close(fig)
