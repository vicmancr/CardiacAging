'''
Segmentation module with pytorch lightning logs.
'''
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pytorch_lightning as pl

# import cv2
from monai.transforms import KeepLargestConnectedComponent, \
    AddChannel, AsDiscrete, FillHoles, Rotate

sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from blueprints import unet


def dice_coef(y_true, y_pred):
    smooth = 0.1
    intersection = torch.sum(y_true.view(-1) * y_pred.view(-1)) + smooth
    union = torch.sum(y_true) + torch.sum(y_pred) + smooth

    return 2 * intersection / union

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def init(args, settings):
    return Segmentor(**settings), None


class Segmentor(pl.LightningModule):
    """
    Segmentor implementation.
    Example::
        from pl_bolts.models.regressor import Segmentor
        r = Segmentor()
        Trainer(gpus=2).fit(r)
    """
    def __init__(
        self,
        num_channels: int = 1,
        num_classes: int = 3,
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
        self.timeframe = 0
        self.criterion = nn.BCEWithLogitsLoss()

        self.model = unet.UNet(
            self.hparams.n_channels,
            self.hparams.num_classes)

    def forward(self, *args):
        "Prediction/inference action."
        return self.model(*args)

    def shared_step(self, batch, batch_idx, prefix=''):
        'Common step to use for train, val and test.'
        img_seg, info = batch
        img, mask = img_seg['img'], img_seg['seg']
        img = img.float()
        out = self.model(img)

        loss = None
        if prefix != '':
            # Computing loss and logging
            mask = mask.float()
            loss = self.criterion(out.permute((0,2,3,1)), mask.permute((0,2,3,1)))
            self.log('{}_loss'.format(prefix), loss)
            out_sf = F.softmax(out, dim=1)
            dice = dice_coef(mask, out_sf)
            self.log('{}_dice'.format(prefix), dice)

        return loss, out, mask, info

    def training_step(self, batch, batch_idx):
        '''
        Training_step defines the train loop.
        It is independent of forward.
        '''
        bce, *_ = self.shared_step(batch, batch_idx, 'train')
        return bce

    def validation_step(self, batch, batch_idx):
        'Validation step'
        bce, *_ = self.shared_step(batch, batch_idx, 'val')
        return bce

    def test_step(self, batch, batch_idx):
        'Test step'
        bce, out, mask, info = self.shared_step(batch, batch_idx, '')

        return bce, out, mask, info

    def test_step_end(self, test_step_outputs):
        'Operation over results from each test step.'
        results_dir = self.hparams.results_folder

        _, pred, _, info = test_step_outputs
        pred_amax = torch.argmax(pred, dim=1)
        pred_amax = AddChannel()(pred_amax)
        pred_oh = AsDiscrete(to_onehot=True, n_classes=6)(pred_amax)
        pred_aux = FillHoles(applied_labels=[1,2,3,4,5])(pred_oh)
        pred_aux = KeepLargestConnectedComponent(applied_labels=[1,3,4,5])(pred_aux)
        pred_mask = torch.argmax(pred_aux, dim=0).cpu().numpy()

        # Iterate over the batch to get every element
        for i in range(pred.size(0)):
            image_filename = info[i]
            _id = image_filename.split('/')[-2]
            filename = os.path.join(results_dir, '{}_t{}.npy'.format(_id, self.timeframe))
            np.save(filename, pred_mask[i])

    def configure_optimizers(self):
        'Configure optimizer.'
        lrate = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lrate, betas=(beta1, beta2))

        return optimizer
