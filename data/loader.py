'''
Utilities to load datasets.
'''
import os
import numpy as np

import torch

from monai.data import ImageDataset, ZipDataset, Dataset
from monai.transforms import AddChannel, Compose, Resize, \
    ScaleIntensity, EnsureType, RandBiasField, RandAdjustContrast, \
    RandHistogramShift, Transpose
from monai.transforms import LoadImaged, AddChanneld, Resized, RandRotated, \
    ScaleIntensityd, EnsureTyped, RandBiasFieldd, RandAdjustContrastd, \
    RandHistogramShiftd, AsDiscreted, ToTensord, Rand2DElasticd

from data import ukbb_loader
from data.paired_data import UnalignedPairedData
from data.transforms import LoadPngImaged, LoadNumpyImaged, Slice2DImage, SliceImageMid, \
    SliceImaged, SliceImage, SliceImageZ, Slice2DImaged

from utils.tensorbacked_array import TensorBackedImmutableStringArray, TensorBackedImmutableNestedArray

import warnings

# MONAI is surprisingly annoying
warnings.simplefilter(action='ignore', category=UserWarning)



def get_files_seg(args, settings, dataset, split_data=True):
    'Get list of files to use in data loaders.'
    if 'target' not in settings.keys():
        settings['target'] = 'age'

    # In segmentation task, split_data=False means that we are generating
    # masks for images without them and need not be filtered out
    masked = split_data
    attributes = not args.test

    # View and data format as input arguments
    file_format = '.nii.gz'
    if 'file_format' in settings.keys():
        file_format = settings['file_format']
    file_list, labels, file_data = ukbb_loader.load_ukbb_data(
        args.dataf, args.maskf, os.path.dirname(dataset), dataset,
        target=settings['target'], view=settings['view'], subcat=settings['subcat'],
        balance_classes=False, num_classes=settings['num_classes'],
        masked=masked, attributes=attributes, file_format=file_format)

    # Shuffle file list so that labels are balanced between train and val datasets
    _order = np.arange(len(file_list))
    np.random.shuffle(_order)
    train_files, train_labels = file_list[_order], labels[_order]
    train_data = file_data[_order]

    val_files = None
    val_labels = None
    val_data = None
    if split_data:
        split = int(0.8*len(file_list))
        val_files, val_labels = file_list[_order][split:], labels[_order][split:]
        val_data = file_data[_order][split:]
        train_files, train_labels = train_files[:split], train_labels[:split]
        train_data = train_data[:split]

        print('\n>> First train', len(train_files), train_files[:5])

    return train_files, train_labels, train_data, val_files, val_labels, val_data


def get_files(args, settings, dataset, split_data=True):
    'Get list of files to use in data loaders.'

    balance_classes = False
    if settings['task_type'] == 'classification':
        balance_classes = settings['balance_classes']
    if 'target' not in settings.keys():
        settings['target'] = 'age'

    # In segmentation task, split_data=False means that we are generating
    # masks for images without them and need not be filtered out
    masked=True
    attributes = True
    if settings['task_type'] == 'regression':
        masked = split_data
        attributes = not args.test
        if not attributes:
            settings['target'] = 'annot'

    # View and data format as input arguments
    file_format = '.nii.gz'
    if 'file_format' in settings.keys():
        file_format = settings['file_format']
    file_list, labels, file_data = ukbb_loader.load_ukbb_data(
        args.dataf, args.maskf, os.path.dirname(dataset), dataset,
        target=settings['target'], view=settings['view'], subcat=settings['subcat'],
        balance_classes=balance_classes, num_classes=settings['num_classes'],
        masked=masked, attributes=attributes, file_format=file_format)

    print('\n-> labels', np.unique(labels, return_counts=True))
    print('\n-> file_list', np.unique(file_list, return_counts=True))
    # One hot encoding if num_classes greater than 1
    # TODO: Add a nicer condition than skipping segmentation
    if settings['num_classes'] > 1:
        labels = np.eye(settings['num_classes'])[labels.astype(int).reshape(-1)]

    # Shuffle file list so that labels are balanced between train and val datasets
    _order = np.arange(len(file_list))
    if split_data:
        np.random.shuffle(_order)
    train_files, train_labels = file_list[_order], labels[_order]
    train_data = file_data[_order]

    val_files = None
    val_labels = None
    val_data = None
    if split_data:
        split = int(0.8*len(file_list))
        val_files, val_labels = file_list[_order][split:], labels[_order][split:]
        val_data = file_data[_order][split:]
        train_files, train_labels = train_files[:split], train_labels[:split]
        train_data = train_data[:split]

        print('\n>> First train', len(train_files), train_files[:5])
        print('\n>> First train', train_data[:5])
        print('\n>> First val', val_files[:5])
        print('\n>> First val', val_data[:5])

    # Issue with increase shared memory usage as training goes on
    # as explained in here 
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    # One solution: use numpy and pandas elements instead of list and dicts.
    # Careful! A numpy array of type object does not work either, as commented in
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-603823029
    # This might make it problematic for readers (might need to decode first the str).
    # Another alternative would be to encode strings to integers and decode back:
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
    # This is a custom tensor-based array from
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-684831789
    train_files = TensorBackedImmutableStringArray(train_files)
    if split_data:
        val_files = TensorBackedImmutableStringArray(val_files)
    if settings['target'] == 'annot':
        if split_data: # = if train (in segm. the labels are all nan for testing)
            train_labels = TensorBackedImmutableStringArray(np.array(train_labels))
            val_labels = TensorBackedImmutableStringArray(np.array(val_labels))
    else:
        train_labels = np.array(train_labels).astype(np.float)
        if split_data:
            val_labels = np.array(val_labels).astype(np.float)

    if isinstance(train_data[0], list) or isinstance(train_data[0], np.ndarray):
        train_data = TensorBackedImmutableNestedArray(train_data)
    else:
        train_data = TensorBackedImmutableStringArray(train_data)

    if split_data:
        val_data = TensorBackedImmutableNestedArray(val_data)

    return train_files, train_labels, train_data, val_files, val_labels, val_data


def load_segmentation_dset(args, settings, split_data=True):
    '''
    Dataset for segmentation models.
    '''
    h, w = settings['input_shape']
    if args.data_type == '2d':
        spatial_size = (h, w)
    elif args.data_type == '3d':
        spatial_size = (h, w, 50)

    # Data augmentation
    keys_tf = ['img', 'seg']
    if args.test:
        keys_tf = ['img']
    # Select image loader depending on the format
    if args.data_format == 'nifti':
        settings['file_format'] = '.nii.gz'
        loader_fns = [
            LoadImaged(keys=keys_tf),
            SliceImaged(keys=keys_tf, slice_pos=0, time_frame=int(args.timeframe))
            # Slice2DImaged(keys=keys_tf)
        ]
    elif args.data_format == 'png':
        settings['file_format'] = '.png'
        loader_fns = [LoadPngImaged(keys=keys_tf)]
    elif args.data_format == 'numpy':
        settings['file_format'] = '.npy'
        loader_fns = [LoadNumpyImaged(keys=keys_tf)]
    else:
        raise Exception(
            'Loader for data format "{}" not implemented!'.format(args.data_format))

    # Transformations for segmentations
    seg_transforms = [
        *loader_fns,
        AddChanneld(keys=keys_tf),
        ScaleIntensityd(keys=["img"])
    ]
    if not args.test:
        seg_transforms.extend(
            [
                RandBiasFieldd(keys=["img"], prob=0.5),
                RandAdjustContrastd(keys=["img"], prob=0.5),
                RandHistogramShiftd(keys=["img"], prob=0.5),
                Rand2DElasticd(
                    keys=["img", "seg"], prob=1, spacing=[20,20],
                    magnitude_range=(1,2), mode=["bilinear", "nearest"]),
                ToTensord(keys=keys_tf),
                # Round mask values to closest integer
                AsDiscreted(keys=["seg"], rounding='torchrounding'),
                # Convert mask labels to num_classes image with 1s
                AsDiscreted(
                    keys=["seg"], to_onehot=True, num_classes=settings['num_classes']
                ),
                RandRotated(
                    keys=keys_tf, prob=0.5, range_x=15,
                    mode=['bilinear', 'nearest']
                )
            ]
        )
    seg_transforms.extend(
        [
            Resized(keys=keys_tf, spatial_size=spatial_size),
            EnsureTyped(keys=keys_tf),
        ]
    )

    train_files, train_labels, train_data, val_files, val_labels, val_data = get_files_seg(
        args, settings, args.dataset, split_data)

    if args.test:
        train_files_dict = [
            {"img": img, "seg": 0} for img in train_files]
    else:
        train_files_dict = [
            {"img": img, "seg": seg} for img, seg in zip(train_files, train_labels)]

    # Data in order is: id, sex, age, bmi
    train_dset = Dataset(train_files_dict, transform=Compose(seg_transforms))
    train_data_dset = Dataset(train_data)
    train_dset = ZipDataset([train_dset, train_data_dset])

    train_loader = torch.utils.data.DataLoader(train_dset,
        batch_size=args.batch_size, shuffle=split_data,
        num_workers=args.workers, pin_memory=True)

    # Validation data
    val_loader = None
    if split_data:

        val_transforms = [
            LoadImaged(keys=["img", "seg"]),
            SliceImaged(keys=["img", "seg"], slice_pos=0, time_frame=0),
            # Slice2DImaged(keys=["img", "seg"]),
            AddChanneld(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img"]),
            ToTensord(keys=["img", "seg"]),
            # Round mask values to closest integer
            AsDiscreted(keys=["seg"], rounding='torchrounding'),
            # Convert mask labels to num_classes image with 1s
            AsDiscreted(
                keys=["seg"], to_onehot=True, num_classes=settings['num_classes']
            ),
            Resized(keys=["img", "seg"], spatial_size=spatial_size),
            EnsureTyped(keys=["img", "seg"]),
        ]

        val_files_dict = [{"img": img, "seg": seg} for img, seg in zip(val_files, val_labels)]

        val_dset = Dataset(val_files_dict, transform=Compose(val_transforms))
        val_data_dset = Dataset(val_data)
        val_dset = ZipDataset([val_dset, val_data_dset])

        val_loader = torch.utils.data.DataLoader(val_dset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)


    return train_loader, val_loader


def load_dataset(args, settings, split_data=True):
    '''
    Prepare dataset for pytorch model
    with corresponding transformation functions.
    '''
    h, w = settings['input_shape']
    if args.data_type == '2d':
        if len(settings['subcat']) < 5:
            slc_im = SliceImage(0, 0)
        else:
            # 2D nifti file. No need to slice time position.
            slc_im = Slice2DImage()
        rsz_im = Resize((h, w))
    elif args.data_type == '3d':
        slc_im = SliceImageZ(0)
        rsz_im = Resize((h, w, 10))

    if settings['view'] == 'sa':
        slc_im = SliceImageMid(0)

    # Select image loader depending on the format
    transf_array = []
    if args.data_format == 'nifti':
        settings['file_format'] = '.nii.gz'
        reader = 'NibabelReader'
        transf_array.append(slc_im)
    elif args.data_format == 'png':
        settings['file_format'] = '.png'
        reader = 'PILReader'
    elif args.data_format == 'numpy':
        settings['file_format'] = '.npy'
        reader = 'NumpyReader'
    else:
        raise Exception(
            'Loader for data format "{}" not implemented!'.format(args.data_format))

    # --------------------------------------
    transf_array.append(AddChannel())
    if args.train:
        transf_array.extend([
            RandBiasField(prob=0.1),
            RandAdjustContrast(prob=0.1),
            RandHistogramShift(prob=0.1),
        ])

    transf_array.extend([
        ScaleIntensity(),
        rsz_im,
        EnsureType()])
    # Tranpose time to be in second position B T H W
    if args.data_type == '3d':
        transf_array.append(Transpose([0,3,1,2]))

    transform = Compose(transf_array)

    train_files, train_labels, train_data, val_files, val_labels, val_data = get_files(
        args, settings, args.dataset, split_data)

    # Data in order is: id, sex, age, bmi
    train_dset = ImageDataset(train_files, transform=transform, labels=train_labels, reader=reader)
    train_data_dset = Dataset(train_data)
    train_dset = ZipDataset([train_dset, train_data_dset])

    train_loader = torch.utils.data.DataLoader(train_dset,
        batch_size=args.batch_size, shuffle=split_data,
        num_workers=args.workers, pin_memory=args.workers>1)

    # Validation data
    val_loader = None
    if split_data:
        val_dset = ImageDataset(val_files, transform=transform, labels=val_labels, reader=reader)
        val_data_dset = Dataset(val_data)
        val_dset = ZipDataset([val_dset, val_data_dset])

        val_loader = torch.utils.data.DataLoader(val_dset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False)

    # For generative models with original and target images
    if settings['task_type'] == 'generative':
        # If a target dataset different from original one is provided
        if args.tg_dataset != '':
            train_files, train_labels, train_data, val_files, val_labels, val_data = get_files(
                args, settings, args.tg_dataset, split_data)

            train_dset = ImageDataset(train_files, transform=transform, labels=train_labels, reader=reader)
            train_data_dset = Dataset(train_data)
            train_dset = ZipDataset([train_dset, train_data_dset])

        # Second loader with shuffled data
        train_loader2 = torch.utils.data.DataLoader(train_dset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=args.workers>1)

        train_loader = UnalignedPairedData(train_loader, train_loader2, return_paths=False)

        if split_data:
            val_dset = ImageDataset(val_files, transform=transform, labels=val_labels, reader=reader)
            val_data_dset = Dataset(val_data)
            val_dset = ZipDataset([val_dset, val_data_dset])

            # Second loader with shuffled validation data
            val_loader2 = torch.utils.data.DataLoader(val_dset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=False)

            val_loader = UnalignedPairedData(val_loader, val_loader2, return_paths=False)


    return train_loader, val_loader
