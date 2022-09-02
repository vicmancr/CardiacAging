'''Configuration template for a generic classification experiment'''

EXPERIMENT_PARAMS = {
    # 'augment': True,  # Use data augmentation
    'balance_classes': False,
    'batch_size': 8,
    'beta1': 0.9,
    'beta2': 0.999,
    'epochs': 100,
    'initial_filters': 32,
    'input_shape': (256,256),
    'kernel_size': (3, 3),
    'learning_rate': 1e-4,  # Learning rate
    'module_name': 'Classifier',
    'num_classes': 1,
    'n_channels': 1,
    'project_name': 'ageingheart',
    'subcat': '4ch',
    'target': 'sex', # Output to predict
    'task_type': 'classification',
    'view': 'la',
    'weight_decay': 0,  # Weight decay rate
}


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
