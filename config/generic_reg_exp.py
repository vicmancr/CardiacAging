'''Configuration template for a generic regression experiment'''

EXPERIMENT_PARAMS = {
    # 'augment': True,  # Use data augmentation
    'batch_size': 8,
    'beta1': 0.9,
    'beta2': 0.999,
    'epochs': 100,
    'initial_filters': 32,
    'input_shape': (128,128),
    'kernel_size': (3, 3),
    'learning_rate': 1e-4,  # Learning rate
    'module_name': 'Regressor',
    'num_classes': 1,
    'n_channels': 1,
    'optimizer': 'adamw',
    'project_name': 'ageingheart',
    'subcat': '4ch',
    'target': 'age', # Output to predict
    'task_type': 'regression',
    'view': 'la',
    'weight_decay': 0,  # Weight decay rate
}


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
