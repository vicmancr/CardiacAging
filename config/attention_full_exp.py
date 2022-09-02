'''
Model with attention fully (also for discriminator)
From Diffusion models beat GANs
https://arxiv.org/abs/2105.05233
Model originally introduced by Ho et al.
https://arxiv.org/abs/2006.11239
'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.generic_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'reduced': False,
    'epochs': 300,
    'embedding_max_period': 10000,
    'dims': 2,
    'use_tanh': False,
    'input_shape': (128,128),
    'num_classes': 1,
    'num_groups_GN': 32,
    'num_res_blocks': 2,
    'model_channels': 32,
    'module_name': 'GAN',
    'attention_resolutions': [32, 16, 8],
    'optD': 'adamw',
    'optG': 'adamw',
    'weight_decay': 1e-4,
    'reconstruction_weight': 0,
    'regularization_weight': 10,
    'radial_prior_weight': 0,
    'cycle_cons_weight': 10,
    'discr_params': {
        'model': 'EncoderUNetModel',
        'num_covariates': 1,
        'use_scale_shift_norm': False, # use a FiLM-like conditioning mechanism.
    },
    'gen_params': {
        'model': 'UNetModel',
        'num_covariates': 1,
        'use_scale_shift_norm': False, # use a FiLM-like conditioning mechanism.
        # Default diffusion config
        # image_size=64,
        # num_channels=192,
        # num_res_blocks=3,
        # channel_mult="",
        # num_heads=1,
        # num_head_channels=64,
        # num_heads_upsample=-1,
        # attention_resolutions="32,16,8",
        # dropout=0.1,
        # text_ctx=128,
        # xf_width=512,
        # xf_layers=16,
        # xf_heads=8,
        # xf_final_ln=True,
        # xf_padding=True,
        # diffusion_steps=1000,
        # noise_schedule="squaredcos_cap_v2",
        # timestep_respacing="",
        # use_scale_shift_norm=True,
        # resblock_updown=True,
        # use_fp16=True,
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
