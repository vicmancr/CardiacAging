'''BMI regression'''
import importlib

EXPERIMENT_PARAMS = importlib.import_module('config.generic_reg_exp').get()

NEW_EXPERIMENT_PARAMS = {
    'target': 'bmi', # Output to predict
}

EXPERIMENT_PARAMS.update(**NEW_EXPERIMENT_PARAMS)


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
