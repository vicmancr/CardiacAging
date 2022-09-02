'''
Train script for pytorch model
'''
import os
import glob
import logging
import importlib

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from data.loader import load_dataset, load_segmentation_dset
from config.options import parse_options
from models.gan import GAN
from models.regressor import Regressor
from models.classifier import Classifier
from models.segmentor import Segmentor


log = logging.getLogger("proposed_executor")
wd = os.path.dirname(os.path.realpath(__file__))


class InitCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        print("Sending model weights to GPU!")
        pl_module.regressor = [rgr.to(pl_module.device) for rgr in pl_module.regressor]


def main():
    '''Main function.'''
    args, settings = parse_options()
    settings['working_dir'] = wd
    if settings['task_type'] != 'segmentation':
        data_loader = load_dataset 
    else:
        data_loader = load_segmentation_dset

    # GPUs to use
    gpus = [0]
    if args.gpu is not None:
        gpus = np.asarray(args.gpu.split(',')).astype(int)
    # Folder for model checkpoints
    ckpt_folder = os.path.join(args.outf, args.name)
    os.makedirs(ckpt_folder, exist_ok=True)

    if args.train:
        print('-> Training model', args.name)
        # Data loading code
        train_loaders, val_loaders = data_loader(args, settings)

        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            every_n_epochs=20, # Save ckpt every 20 epochs
            dirpath=ckpt_folder,
            filename='model-{epoch:03d}-{train_loss:.2f}',
            save_top_k=30,
            save_last=True,
            mode='min',
        )
        callbacks = [checkpoint_callback]

        # Initialize model: Load model and call init() function
        settings['num_training_samples'] = len(train_loaders)
        settings['num_validation_samples'] = len(val_loaders)
        settings['dataset_file'] = args.dataset
        settings['data_type'] = args.data_type
        init_module = importlib.import_module(
            'models.{}'.format(settings['module_name'].lower()))
        model, clbk = init_module.init(args, settings)
        if clbk:
            callbacks.append(clbk)

        # Train with Pytorch Lightning
        pl.seed_everything(1234, workers=True)

        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(wd, 'logs'), name=args.name)

        # Drop comet logger when testing
        loggers = [tb_logger]
        if args.iters is not None:
            if args.iters <= 10:
                loggers = [tb_logger]

        trainer = pl.Trainer(
            deterministic=True,
            default_root_dir=args.outf,
            max_epochs=settings['epochs'],
            gpus=[*gpus], # GPU id to use (can be [1,3] [ids 1 and 3] or [-1] [all])
            max_steps=args.iters, # default is None (not limited)
            logger=loggers,
            accumulate_grad_batches=settings['accumulated_grad_batches'],
            callbacks=callbacks)

        # Provide confirmation message of settings defined for debugging
        print('Training model for {} epochs with dataset {}.'.format(
            settings['epochs'], args.experiment))

        trainer.fit(model, train_loaders, val_loaders)

    if args.test:
        print('Testing model {} with data from {}'.format(
            args.name, args.dataf))

        # Find checkpoint and hparams file
        if os.path.exists(os.path.join(ckpt_folder, 'last.ckpt')):
            ckpt_model = os.path.join(ckpt_folder, 'last.ckpt')
        else:
            ckpt_name = '*-epoch*.ckpt'
            if args.epochs > 0:
                ckpt_name = '*-epoch={0:03d}*.ckpt'.format(args.epochs)
            print('Looking for model in "{}" with name "{}"'.format(ckpt_folder, ckpt_name))
            ckpt_model = list(sorted(glob.iglob(
                os.path.join(ckpt_folder, ckpt_name)
            )))[-1]
        print('Found model "{}"'.format(ckpt_model))
        hpms_file = list(glob.iglob(
            os.path.join(wd, 'logs', args.name, '*', 'hparams.yaml')))[-1]
        print('hparams file "{}"'.format(hpms_file))

        if settings['task_type'] == 'regression':
            model = Regressor.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None,
                results_folder=settings['results_folder'])
        elif settings['task_type'] == 'classification':
            model = Classifier.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None)
        elif settings['task_type'] == 'segmentation':
            model = Segmentor.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None,
                results_folder=settings['results_folder'])
        elif settings['task_type'] == 'generative':
            model = GAN.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None,
                results_folder=settings['results_folder'])

        # Data loading code
        test_loaders, _ = data_loader(args, settings, split_data=False)

        # Test model
        os.makedirs(
            os.path.join(settings['results_folder'], settings['experiment_name']),
            exist_ok=True)
        trainer = pl.Trainer(gpus=[*gpus])
        trainer.test(model, test_loaders)



if __name__ == '__main__':
    main()
