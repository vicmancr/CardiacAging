'''Configuration template for a generic regression experiment'''

EXPERIMENT_PARAMS = {
    # 'augment': True,  # Use data augmentation
    'batch_size': 8,
    'beta1': 0.9,
    'beta2': 0.999,
    'epochs': 30,
    'initial_filters': 32,
    'input_shape': (128,128),
    'kernel_size': (3, 3),
    'learning_rate': 1e-4,  # Learning rate
    'model': 'r2plus1d_18',
    'module_name': 'Regressor',
    'num_classes': 1,
    'n_channels': 1,
    'optimizer': 'adamw',
    'project_name': 'ageingheart',
    'scheduler': 'exponential', # None, linear, exponential, cosinewarmup
    'subcat': '4ch',
    'target': 'age', # Output to predict
    'task_type': 'regression',
    'view': 'la',
    'weight_decay': 0.01,  # Weight decay rate (default in torch's AdamW)
}


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
