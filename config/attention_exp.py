'''Fullsize model'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.generic_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'reduced': False,
    'epochs': 300,
    'embedding_max_period': 10000,
    'dims': 2,
    'use_tanh': False,
    'input_shape': (128,128),
    'num_groups_GN': 32,
    'num_res_blocks': 2,
    'model_channels': 32,
    'attention_resolutions': [32, 16, 8],
    'optD': 'adamw',
    'optG': 'adamw',
    'weight_decay': 1e-4,
    'reconstruction_weight': 0,
    'regularization_weight': 10,
    'radial_prior_weight': 0,
    'cycle_cons_weight': 10,
    'discr_params': {
        'activation': 'gelu',
        'model': 'DiscriminatorXiaOld',
        'encoding': 'positive',
    },
    'gen_params': {
        'activation': 'gelu',
        'model': 'UNetModel',
        'encoding': 'both',
        'use_scale_shift_norm': False, # use a FiLM-like conditioning mechanism.
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
