'''Fullsize model with tanh activation for the generator output'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.generic_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'reduced': False,
    'use_tanh': False,
    'weight_decay': 1e-4,
    'reconstruction_weight': 0,
    'regularization_weight': 0,
    'cycle_cons_weight': 10,
    'discr_params': {
        'encoding': 'positive'
    },
    'gen_params': {
        'encoding': 'both'
    }
}

# Merge dictionaries
for key, value in NEW_EXPERIMENT_PARAMS.items():
    if isinstance(value, dict) and key in EXPERIMENT_PARAMS.keys():
        EXPERIMENT_PARAMS[key].update(**value)
    else:
        EXPERIMENT_PARAMS[key] = value


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
