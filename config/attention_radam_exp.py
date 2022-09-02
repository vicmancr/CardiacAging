'''Fullsize model'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.attention_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'optD': 'radam',
    'optG': 'radam',
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
