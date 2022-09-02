'''Configuration template for a generic segmentation experiment'''

EXPERIMENT_PARAMS = {
    # 'augment': True,  # Use data augmentation
    'batch_size': 8,
    'beta1': 0.9,
    'beta2': 0.999,
    'epochs': 500,
    'initial_filters': 32,
    # 'input_shape': (128,128),
    'input_shape': (256,256),
    'kernel_size': (3, 3),
    'learning_rate': 1e-4,  # Learning rate
    'module_name': 'Segmentor',
    # 'num_classes': 7,
    'num_classes': 6,
    'n_channels': 1,
    'project_name': 'segmentation',
    'reduced': True,
    # 'subcat': '4ch_ed',
    'subcat': '4ch',
    'target': 'annot',
    'task_type': 'segmentation',
    'view': 'la',
    'weight_decay': 0,  # Weight decay rate
}


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
