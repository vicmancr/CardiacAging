import os
import re
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib


def get_file(file_id):
    _file = glob.iglob(
        'repo_directory/data/UKBB/{}/la_4ch.nii.gz'.format(file_id))
    _file = list(_file)
    if len(_file) == 0:
        print('--> No file found for', file_id)
        return False

    return list(_file)[0]


def draw_septum(mask):
    'Find the intraventricular septum and set it to label 7.'
    # Get centroids of LV and RV
    cnt1 = np.mean(np.where(mask == 1), axis=1)
    cnt3 = np.mean(np.where(mask == 3), axis=1)
    # LV apex (smallest point in x axis)
    lvapex = np.array(np.where(mask == 1))
    _ind = np.argmin(lvapex, axis=1)
    lvapex = lvapex[:,_ind[1]]

    new_mask = mask.copy()

    # Compute vectors
    cvec = cnt3 - cnt1 # centroid to centroid
    cvec = cvec / np.linalg.norm(cvec)
    lvvec = cnt1 - lvapex # LV centroid to apex
    lvvec = lvvec / np.linalg.norm(lvvec)
    # Perpendicular to cvec
    pvec = np.array([-cvec[1], cvec[0]])
    # Line perpendicular to lineCnt and crossing lv centroid
    lineLV = lambda x: cnt1[0] + (x - cnt1[1]) * pvec[0] / pvec[1]
    # Line crossing lv apex and lv centroid
    lineApex = lambda x: lvapex[0] + (x - lvapex[1]) * lvvec[0] / lvvec[1]

    # Identify interventricular septum
    a = np.array(np.where(mask == 2))
    # points under LV line
    mk = (a[0] < lineApex(a[1]))
    # points under perpendicular line to interventricular line
    mk = mk & (a[0] < lineLV(a[1]))
    new_mask[tuple(a[:,mk])] = 7

    return new_mask


def extract_measures_from_files(masks_directory, output_directory):
    '''
    Extract measures (volumes) from masks.
    Format of files is
    sample_`1000195`_sex-`0/1`_age-`67`_bmi-`18.7`_`image/map`_step_`+-1`_seg.npy
    for predicted samples from GANs.
    '''
    N_LABELS = 7
    # We don't want to evaluate mappings, but images
    files = sorted(glob.iglob(os.path.join(masks_directory, '*_image_step*.npy')))
    arr = []
    for j, f in enumerate(files):
        s = f.split('_')
        _id = s[1]
        step = s[-2].split('.')[0]

        # Amount of modifications in the image
        mapf = re.sub(r'\_image\_step', '_map_step', f)
        mapf = re.sub(r'segs', 'numpy', mapf)
        mapf = re.sub(r'\_seg', '', mapf)
        _map = np.load(mapf)
        map_weight = np.sum(np.abs(_map - 0.5))

        # Original image
        nii = nib.load(get_file(_id))
        resol = nii.header.structarr['pixdim'][1]
        sx, sy = nii.shape[:2]

        mask = np.load(f)
        # Target shape (original shape times pixel resolution)
        # for a target resolution of 1x1 mm
        tx, ty = int(sx * resol), int(sy * resol)
        mask = np.pad(mask, [(45,45),(0,0)]) # Since we cropped originally the image with [45:-45, :]
        rsz_mask = cv2.resize(mask, (tx,ty), interpolation=cv2.INTER_NEAREST)
        lbs, cnts = np.unique(rsz_mask, return_counts=True)
        aux = [_id, step, resol, sx, sy, tx, ty, map_weight]
        # We count number of pixels with each label value
        for n in range(1, N_LABELS):
            i = np.where(lbs == n)[0]
            if i.size == 0:
                aux.append(0)
                continue
            # Num. of pixels with label n
            aux.append(cnts[i[0]])

        # Compute septum (lb=7) and measure volume
        try:
            rsz_mask = draw_septum(rsz_mask)
            # Compute length of septum
            spt_up = np.array(np.where(rsz_mask == 7))
            _ind = np.argmax(spt_up, axis=1)
            spt_up = spt_up[:,_ind[1]]
            spt_do = np.array(np.where(rsz_mask == 7))
            _ind = np.argmin(spt_do, axis=1)
            spt_do = spt_do[:,_ind[1]]
            length = np.linalg.norm(spt_up - spt_do)
            aux.append(np.sum(rsz_mask == 7))
            aux.append(length)
        except Exception as e:
            # print('\t error for id {} step {}:'.format(_id, step), e)
            aux.extend([0, 0])

        arr.append(aux)

    measures = pd.DataFrame(
        arr, columns=[
            'ID', 'step', 'pixel_resolution', 'org_shape_x', 'org_shape_y', 'cur_shape_x', 'cur_shape_y',
            'map_weight', *np.arange(1, N_LABELS), 7, 'septum_length'])
    measures.to_csv(os.path.join(output_directory, 'volumes_1x1mm.csv'), index=False)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('mkdir', type=str, help='path to masks')
    parser.add_argument('--outdir', default='', type=str, help='output path')
    args = parser.parse_args()


    masks_dir = args.mkdir
    output_dir = args.outdir
    if output_dir == '':
        output_dir = masks_dir
    print('> Extracting volumetric mesures in folder', masks_dir)
    extract_measures_from_files(masks_dir, output_dir)
