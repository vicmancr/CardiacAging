'''
Regressor module with pytorch lightning logs.
'''
import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from blueprints import vgg, resnet, conv_3d, video_resnet


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def init(args, settings):
    return Regressor(**settings), None


class Regressor(pl.LightningModule):
    """
    Regressor implementation.
    Example::
        from pl_bolts.models.regressor import Regressor
        r = Regressor()
        Trainer(gpus=2).fit(r)
    """
    def __init__(
        self,
        model_name: str = 'vgg11_bn',
        learning_rate: float = 0.0001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        reduced: bool = False,
        **kwargs
    ):
        """
        Args:
            learning_rate: the learning rate
            beta1: first momentum of Adam optimizer
        """
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.batch_size = self.kwargs['batch_size']
        self.train_iters_per_epoch = self.kwargs['num_training_samples']//self.batch_size

        if model_name in vgg.__all__:
            module = getattr(vgg, model_name)
            self.model = module(num_classes=1)
        elif model_name in resnet.__all__:
            module = getattr(resnet, model_name)
            self.model = module(num_classes=1)
        elif model_name == 'unet3d':
            module = getattr(conv_3d, 'conv3D')
            self.model = module(num_classes=1)
        elif model_name in ['r3d_18', 'mc3_18', 'r2plus1d_18']:
            module = getattr(video_resnet, model_name)
            self.model = module()

    def forward(self, *args):
        "Prediction/inference action."
        return self.model(*args)

    def shared_step(self, batch, batch_idx, prefix=''):
        'Common step to use for train, val and test.'
        x, y, *_ = batch
        y_hat = self.model(x.float())
        mse = F.mse_loss(y_hat.flatten(), y.float())
        mae = torch.abs(y_hat.flatten() - y.float()).sum().data/len(y)
        err = ((y_hat.flatten() - y.float()) ** 2).sum().data
        var = ((y.mean() - y.float())**2).sum().data
        rsq = 1 - err/var
        if prefix != '':
            # Logging to TensorBoard by default
            self.log('{}_mse_loss'.format(prefix), mse)
            self.log('{}_loss'.format(prefix), mse)
            self.log('{}_mae_loss'.format(prefix), mae, prog_bar=True)
            self.log('{}_rsq'.format(prefix), rsq, prog_bar=True)

        return mse, y_hat, y

    def training_step(self, batch, batch_idx):
        '''
        Training_step defines the train loop.
        It is independent of forward.
        '''
        mse, _, _ = self.shared_step(batch, batch_idx, 'train')
        return mse

    def validation_step(self, batch, batch_idx):
        'Validation step'
        mse, _, _ = self.shared_step(batch, batch_idx, 'val')
        return mse, _, _

    def test_step(self, batch, batch_idx):
        'Test step'
        *_, info = batch
        mse, y_hat, y = self.shared_step(batch, batch_idx, '')
        return mse, y_hat, y, info

    def test_epoch_end(self, test_step_outputs):
        'Operation over results from each test step.'
        results_dir = self.hparams.results_folder

        names = []
        preds = []
        targets = []
        for _, pred, target, info in test_step_outputs:
            names.extend(info)
            preds.extend(pred.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

        # List to tensor
        preds = torch.FloatTensor(preds)
        targets = torch.FloatTensor(targets)
        diff = preds - targets
        # Absolute errors
        aes = torch.abs(diff)
        # Squared errors
        ses = torch.square(diff)
        # Std and Mean
        vmae, mae = torch.std_mean(aes, unbiased=True)
        vmse, mse = torch.std_mean(ses, unbiased=True)

        print('-> MSE for test data {:.3f} (+/- {:.3f})'.format(mse, vmse))
        print('-> MAE for test data {:.3f} (+/- {:.3f})'.format(mae, vmae))

        df = pd.DataFrame(
            np.asarray([names, preds.cpu().numpy(), targets.cpu().numpy()]).T,
            columns=['Filename', 'Prediction', 'Target'])
        print('Saving file to ', os.path.join(results_dir, 'predicted_{}.csv'.format(self.kwargs['target'])))
        filename = os.path.join(results_dir, 'predicted_{}.csv'.format(self.kwargs['target']))
        df.to_csv(filename, index=False)

        return df

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.hparams.scheduler == 'cosinewarmup':
            # Step per iteration
            self.scheduler.step()

    def configure_optimizers(self):
        'Configure optimizer.'
        lrate  = self.hparams.learning_rate
        beta1  = self.hparams.beta1
        beta2  = self.hparams.beta2
        wd     = self.hparams.weight_decay
        warmup = self.hparams.warmup_epochs

        # ------------- Optimizer ------------------------
        try:
            if self.hparams.optimizer == 'adam':
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=lrate, betas=(beta1, beta2), weight_decay=wd)
            elif self.hparams.optimizer == 'radam':
                # Rectified ADAM. Prevents large bias at the start of training.
                # Should work similarly to using a warmup.
                optimizer = torch.optim.RAdam( # https://arxiv.org/pdf/1908.03265.pdf
                    self.parameters(), lr=lrate, betas=(beta1, beta2), weight_decay=wd)
            elif self.hparams.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=lrate, betas=(beta1, beta2), weight_decay=wd)
            elif self.hparams.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    self.parameters(), lr=lrate, weight_decay=wd)
        except AttributeError:
            print('Optimizer not found in model params. Setting ADAM by default.')
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lrate, betas=(beta1, beta2), weight_decay=wd)


        # ------------- Scheduler ------------------------
        if self.hparams.scheduler == None:
            return optimizer
        elif self.hparams.scheduler == 'linear':
            # the scheduler divides the lr by 10 every 10 epoch(s)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1)
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == 'exponential':
            # Exponential decay in LR
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.9)
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == 'cosinewarmup':
            # Cosine decay with warmup
            max_iters = self.train_iters_per_epoch * self.hparams.epochs
            self.scheduler = CosineWarmupScheduler(
                optimizer, warmup=warmup, max_iters=max_iters)
            # We don't return the scheduler because we need to apply it per
            # iteration, not per epoch
            return optimizer
