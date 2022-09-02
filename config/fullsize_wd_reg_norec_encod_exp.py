'''Fullsize model with tanh activation for the generator output'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.generic_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'cycle_cons_weight': 0,
    'reduced': False,
    'use_tanh': False,
    'weight_decay': 1e-4,
    'reconstruction_weight': 0,
    'regularization_weight': 10,
    'discr_params': {
        'activation': 'lrelu',
        'encoding': 'positive',
        'model': 'DiscriminatorXia',
    },
    'gen_params': {
        'activation': 'lrelu',
        'encoding': 'both',
        'model': 'GeneratorXia',
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
