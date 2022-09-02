'''
Regressor module with pytorch lightning logs.
'''
import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, Specificity
import pytorch_lightning as pl

sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from blueprints import vgg, resnet, conv_3d, temp_conv_3d, inception


def init(args, settings):
    return Classifier(**settings), None


class Classifier(pl.LightningModule):
    """
    Classifier implementation.
    Example::
        from pl_bolts.models.classifier import Classifier
        r = Classifier()
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
        # self.kwargs = kwargs

        if model_name in vgg.__all__:
            module = getattr(vgg, model_name)
        elif model_name in resnet.__all__:
            module = getattr(resnet, model_name)
        elif model_name == 'unet3d':
            module = getattr(conv_3d, 'conv3D')
        elif model_name == 'Inception3':
            module = getattr(inception, model_name)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)

        self.model = module(
            num_classes=self.hparams.num_classes,
            num_channels=self.hparams.n_channels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = MetricCollection([
            Accuracy(num_classes=self.hparams.num_classes),
            Precision(num_classes=self.hparams.num_classes, average='macro'),
            Recall(num_classes=self.hparams.num_classes, average='macro'),
            Specificity(num_classes=self.hparams.num_classes, average='macro')
        ])

    def forward(self, *args):
        "Prediction/inference action."
        return self.model(*args)

    def shared_step(self, batch, batch_idx, prefix=''):
        'Common step to use for train, val and test.'
        x, y, *_ = batch
        # y = y.long()
        # x = x.view(x.size(0), -1)
        y_hat = self.model(x.float())
        bce = self.criterion(y_hat.squeeze(), y)
        mc = self.metric(y_hat.squeeze(), y.long())
        if prefix != '':
            # Logging to TensorBoard by default
            self.log('{}_bce_loss'.format(prefix), bce)
            self.log('{}_loss'.format(prefix), bce)
            self.log('{}_acc_loss'.format(prefix), mc['Accuracy'])
            self.log('{}_precision_loss'.format(prefix), mc['Precision'])
            self.log('{}_recall_loss'.format(prefix), mc['Recall'])
            self.log('{}_specificity_loss'.format(prefix), mc['Specificity'])

        return bce, y_hat, y

    def training_step(self, batch, batch_idx):
        '''
        Training_step defines the train loop.
        It is independent of forward.
        '''
        bce, _, _ = self.shared_step(batch, batch_idx, 'train')
        return bce

    def validation_step(self, batch, batch_idx):
        'Validation step'
        bce, _, _ = self.shared_step(batch, batch_idx, 'val')
        return bce, _, _


    def test_step(self, batch, batch_idx):
        'Test step'
        bce, y_hat, y = self.shared_step(batch, batch_idx, '')
        return bce, y_hat, y

    def test_epoch_end(self, test_step_outputs):
        'Operation over results from each test step.'
        logits = []
        targets = []
        for _, pred, target in test_step_outputs:
            logits.append(pred.squeeze())
            targets.append(target.squeeze())

        # Concatenate tensors
        logits = torch.cat(logits, 0)
        targets = torch.cat(targets, 0)
        # Binary cross entropy
        bce = self.criterion(logits, targets)
        # Accuracy
        mc = self.metric(logits, targets.long())
        # Confusion matrix
        nc = max(2, self.hparams.num_classes)
        confmat = torchmetrics.ConfusionMatrix(num_classes=nc)
        cm = confmat(logits.cpu(), targets.long().cpu())

        print('-> Summary:')
        print('-> BCE for test data {:.3f}'.format(bce))
        print('-> Accuracy for test data {:.3f}'.format(mc['Accuracy']))
        print('-> Precision for test data {:.3f}'.format(mc['Precision']))
        print('-> Recall for test data {:.3f}'.format(mc['Recall']))
        print('-> Specificity for test data {:.3f}'.format(mc['Specificity']))
        print('-> Confusion matrix for test data {}'.format(cm))

        df = pd.DataFrame(
            np.asarray([logits.cpu().numpy(), targets.cpu().numpy()]).T,
            columns=['Prediction', 'Target'])

        return df

    def configure_optimizers(self):
        'Configure optimizer.'
        lrate = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lrate, betas=(beta1, beta2))

        return optimizer
