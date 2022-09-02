'''
Model with attention fully (also for discriminator)
From Diffusion models beat GANs
https://arxiv.org/abs/2105.05233
Model originally introduced by Ho et al.
https://arxiv.org/abs/2006.11239
'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.attention_full_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'embedding_max_period': 100,
    'discr_params': {
        'model': 'EncoderUNetModel',
        'use_scale_shift_norm': True, # use a FiLM-like conditioning mechanism.
    },
    'gen_params': {
        'model': 'UNetModel',
        'use_scale_shift_norm': True, # use a FiLM-like conditioning mechanism.
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