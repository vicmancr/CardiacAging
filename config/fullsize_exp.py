'''Fullsize model'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.generic_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'reduced': False,
}

EXPERIMENT_PARAMS.update(**NEW_EXPERIMENT_PARAMS)


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
