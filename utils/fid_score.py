"""
Modified from https://github.com/mseitzer/pytorch-fid

Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import cv2
import glob
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionV3

wd = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.join(wd, '..'))
from models.classifier import Classifier


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size to use')
parser.add_argument('--gpu', type=int, default=0,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('experiment', type=str, help='experiment to analyse.')

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms
        self.image_loader = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class ImagePathDatasetv2(torch.utils.data.Dataset):
    'Generalization of ImagePathDataset class with generic data loader.'
    def __init__(self, files, transforms=None, format='png'):
        self.files = files
        self.transforms = transforms
        self.image_loader = None
        if format == 'png':
            self.image_loader = lambda x: Image.open(x).convert('RGB')
        elif format == 'nifti':
            def load_nifti(x):
                aux = nib.load(x).get_fdata()[...,0,0]
                # Rescale to 0,1
                aux = (aux - aux.min())/(aux.max() - aux.min())
                aux = cv2.resize(aux, (256,256))
                return aux
            self.image_loader = lambda x: load_nifti(x)
        elif format == 'numpy':
            def load_numpy(x):
                aux = np.load(x)
                return aux
            self.image_loader = lambda x: load_numpy(x)
        else:
            raise ValueError('Invalid format for image', format)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = self.image_loader(path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    # File format
    if files[0].endswith('.npz') or files[0].endswith('.npy'):
        format = 'numpy'
    elif files[0].endswith('.nii.gz'):
        format = 'nifti'
    else:
        format = 'png'
    dataset = ImagePathDatasetv2(files, transforms=TF.ToTensor(), format=format)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.float()
        batch = batch.to(device)

        with torch.no_grad():
            # Get the logits before the FC layer
            pred = model(batch)[1]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_filelist(
    filelist, model, batch_size, dims, device, num_workers=1):

    if len(filelist) == 0:
        if filelist.endswith('.npz'):
            with np.load(filelist) as f:
                m, s = f['mu'][:], f['sigma'][:]
        else:
            raise ValueError('Invalid file list', filelist)
    else:
        m, s = calculate_activation_statistics(filelist, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_fid_given_filelists(lists_of_files, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_filelist(lists_of_files[0], model, batch_size,
                                            dims, device, num_workers)
    m2, s2 = compute_statistics_of_filelist(lists_of_files[1], model, batch_size,
                                            dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def compute_fid_from_lists(list_of_files, batch_size, dims=2048):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)

    fid_value = calculate_fid_given_filelists(list_of_files,
                                              batch_size,
                                              device,
                                              dims,
                                              num_workers)

    print('FID:', fid_value)
    return fid_value


def compute_fid_for_experiment(name_of_experiment, batch_size, dims=2048, gpu=0):
    device = torch.device('cuda:{}'.format(gpu) if (torch.cuda.is_available()) else 'cpu')

    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 4)

    # FID from ImageNet vs. FID from our dataset is quite different.
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    # model = InceptionV3([block_idx]).to(device)
    ckpt_model = 'repo_directory/checkpoints/sex-class-2d-inception3-v1/model-epoch=099-train_loss=0.00.ckpt'
    hpms_file = 'repo_directory/logs/sex-class-2d-inception3-v1/version_0/hparams.yaml'
    model = Classifier.load_from_checkpoint(
        checkpoint_path=ckpt_model,
        hparams_file=hpms_file,
        map_location=None).to(device)

    _files = [
        'path_to_list_of_testing_ids.csv'
    ]
    _ids = []
    for f in _files:
        _ids.extend(pd.read_csv(f).ID.values.tolist())
    _ids = np.array(_ids)
    _path = 'repo_directory/results/{}/numpy/*_image_step_{}.npy'

    list1 = list(sorted(glob.iglob(_path.format(name_of_experiment, 0))))
    aux = [int(l.split('_')[1]) for l in list1]
    mk = pd.Series(aux).isin(_ids)
    list1 = np.array(list1)[mk].tolist()
    print('list1', len(list1), list1[:3])
    m1, s1 = calculate_activation_statistics(
        list1, model, batch_size, dims, device, num_workers)

    fids = []
    for step in range(-29, 30):
        print('step', step)
        list2 = list(sorted(glob.iglob(_path.format(name_of_experiment, step))))
        aux = [int(l.split('_')[1]) for l in list2]
        mk = pd.Series(aux).isin(_ids)
        list2 = np.array(list2)[mk].tolist()
        print('list2', len(list2))
        m2, s2 = calculate_activation_statistics(
            list2, model, batch_size, dims, device, num_workers)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        fids.append([step, fid_value])

    pd.DataFrame(fids, columns=['step', 'fid']).to_csv(
        'repo_directory/results/{}/csv/fids.csv'.format(name_of_experiment),
        index=False)
    return fid_value


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print('FID:', fid_value)


if __name__ == '__main__':
    args = parser.parse_args()
    compute_fid_for_experiment(args.experiment, args.batch_size, args.dims, args.gpu)
