'Compute PSNR score = log10 (MAX_I) - log10 (MSE) in dB.'
import glob
import argparse

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('experiment', type=str, help='experiment to analyse.')


def compute_psnr(reference, modified, max_intensity=1):
    ref_im = np.load(reference)
    mod_im = np.load(modified)
    return 20 * np.log10(max_intensity) - 10 * np.log10(np.mean((ref_im - mod_im)**2))


def compute_psnr_for_experiment(experiment):
    print('> Computing PSNR for experiment {}'.format(experiment))

    _files = [
        'path_to_list_of_testing_ids.csv'
    ]
    _ids = []
    for f in _files:
        _ids.extend(pd.read_csv(f).ID.values.tolist())
    _ids = np.array(_ids)

    _path = 'repo_directory/results/{}/numpy/*_image_step_{}.npy'
    ref_list = list(sorted(glob.iglob(_path.format(experiment, 0))))
    _aux = [int(l.split('_')[1]) for l in ref_list]
    _mk = pd.Series(_aux).isin(_ids)
    ref_list = np.array(ref_list)[_mk].tolist()
    
    
    psnr = []
    for step in range(-29, 30):
        if step == 0: continue
        print('step', step)
        synth_list = list(sorted(glob.iglob(_path.format(experiment, step))))
        aux = [int(l.split('_')[1]) for l in synth_list]
        mk = pd.Series(aux).isin(_ids)
        synth_list = np.array(synth_list)[mk].tolist()
        for eli, el in enumerate(synth_list):
            psnr_value = compute_psnr(ref_list[eli], el)
            psnr.append([el, el.split('_')[1], step, psnr_value])

    df = pd.DataFrame(psnr, columns=['filename', 'ID', 'step', 'psnr'])
    df.to_csv(
        'repo_directory/results/{}/csv/psnr.csv'.format(experiment),
        index=False)

    return True


if __name__ == '__main__':
    args = parser.parse_args()
    compute_psnr_for_experiment(args.experiment)
