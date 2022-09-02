'''Fullsize model with tanh activation for the generator output'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.generic_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'reduced': False,
    'use_tanh': False,
    'weight_decay': 1e-4,
    # 'reconstruction_weight': 0.01,
    'regularization_weight': 10,
}

EXPERIMENT_PARAMS.update(**NEW_EXPERIMENT_PARAMS)


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
