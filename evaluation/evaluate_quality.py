"""
Compare path tracing and neural render images to the reference image using
SSIM, LPIPS and PSNR metrics and store the results into a CSV file.
"""

import io
import json
import os
import sys
import zipfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import torch
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
)
from skimage.metrics import (
    structural_similarity as ssim,
)
from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity as lpips,
)
from tqdm import tqdm

IMAGES_DIR = os.path.join('data', 'images')
EXPERIMENT_DIRS = [
    (
        os.path.join('evaluation', 'experiments', 'quality', 'front', 'path_tracing'),
        os.path.join('quality', 'front', 'path_tracing'),
    ),
    (
        os.path.join('evaluation', 'experiments', 'quality', 'front', 'neural_render'),
        os.path.join('quality', 'front', 'neural_render'),
    ),
    (
        os.path.join(
            'evaluation', 'experiments', 'quality', 'turntable', 'path_tracing'
        ),
        os.path.join('quality', 'turntable', 'path_tracing'),
    ),
    (
        os.path.join(
            'evaluation', 'experiments', 'quality', 'turntable', 'neural_render'
        ),
        os.path.join('quality', 'turntable', 'neural_render'),
    ),
]
RESULTS_DIR = os.path.join('evaluation', 'results')
BATCH_SIZE = 16

GLOBAL_IMG = 1
INDIRECT_IMG = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    skip_existing = '--skip-existing' in sys.argv
    for experiment_dir, result_dir in EXPERIMENT_DIRS:
        for experiment in os.listdir(experiment_dir):
            experiment_path = os.path.join(experiment_dir, experiment)
            csv_dir = os.path.join(RESULTS_DIR, result_dir)
            csv_path = os.path.join(csv_dir, experiment.split('.')[0] + '.csv')
            if skip_existing and os.path.exists(csv_path):
                print(f'Skipping {experiment_path}')
                continue
            print(f'Running {experiment_path}')
            with open(experiment_path, 'rt') as f:
                experiment_json = json.load(f)
            reference_global = experiment_json['reference'] + '_global.png'
            reference_indirect = experiment_json['reference'] + '_indirect.png'
            image_zip = experiment_json['name'] + '.zip'

            results = evaluate_experiment(
                image_zip, reference_global, reference_indirect
            )
            os.makedirs(csv_dir, exist_ok=True)
            results.to_csv(csv_path, index=False)


def read_images(zip_path, verbose=False):
    """Read images from a quality experiment ZIP.

    The ZIP is organized as: {run}/{mode}/{index}_{elapsed_ms}.png
    Returns a list of runs, where each run is a list of
    (time_seconds, global_image, indirect_image) tuples.
    """
    runs = []
    zip_name = os.path.basename(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        run_dirs = sorted(set(file.split('/')[0] for file in zip_ref.namelist()))
        for run in tqdm(run_dirs, desc=f'Reading {zip_name}', disable=not verbose):
            images = []
            for file in sorted(zip_ref.namelist()):
                if file.startswith(f'{run}/global/') and file.endswith('.png'):
                    img_name = file.split('/')[-1]
                    time = int(img_name[:-4].split('_')[-1]) / 1000

                    with zip_ref.open(file) as img_file:
                        img_g = mpimg.imread(io.BytesIO(img_file.read()))

                    with zip_ref.open(file.replace('global', 'indirect')) as img_file:
                        img_i = mpimg.imread(io.BytesIO(img_file.read()))

                    images.append((time, img_g, img_i))
            runs.append(images)
        return runs


def images_to_batch(images, img_index):
    return torch.stack(
        [torch.tensor(entry[img_index]).permute(2, 0, 1).float() for entry in images]
    ).to(device)


def compute_ssim_batch(reference, images, img_index):
    values = [None] * len(images)
    for i, entry in tqdm(enumerate(images), desc='SSIM', total=len(images)):
        values[i] = ssim(reference, entry[img_index], channel_axis=2, data_range=1.0)
    return values


def compute_lpips_batch(reference, images, img_index):
    ref_tensor = (
        torch.tensor(reference).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    scores = []
    for start in tqdm(range(0, len(images), BATCH_SIZE), desc='LPIPS'):
        batch_images = images[start : start + BATCH_SIZE]
        img_batch = images_to_batch(batch_images, img_index)
        ref_batch = ref_tensor.expand(len(batch_images), -1, -1, -1)
        with torch.no_grad():
            batch_scores = lpips(
                ref_batch, img_batch, net_type='vgg', normalize=True, reduction=None
            )
        scores.append(batch_scores.squeeze())
    return torch.cat(scores).tolist()


def compute_psnr_batch(reference, images, img_index):
    values = [None] * len(images)
    for i, entry in tqdm(enumerate(images), desc='PSNR', total=len(images)):
        values[i] = psnr(reference, entry[img_index], data_range=1.0)
    return values


def evaluate_experiment(image_zip, reference_global, reference_indirect):
    image_zip_path = os.path.join(IMAGES_DIR, image_zip)
    runs = read_images(image_zip_path, verbose=True)
    image_global = plt.imread(reference_global)
    image_indirect = plt.imread(reference_indirect)

    runs_column = [i + 1 for i, run in enumerate(runs) for _ in range(len(run))]
    flat_images = [(t, g, i) for images in runs for t, g, i in images]

    return pd.DataFrame(
        {
            'time': [t for t, _, _ in flat_images],
            'run': runs_column,
            'ssim_global': compute_ssim_batch(image_global, flat_images, GLOBAL_IMG),
            'ssim_indirect': compute_ssim_batch(
                image_indirect, flat_images, INDIRECT_IMG
            ),
            'lpips_global': compute_lpips_batch(image_global, flat_images, GLOBAL_IMG),
            'lpips_indirect': compute_lpips_batch(
                image_indirect, flat_images, INDIRECT_IMG
            ),
            'psnr_global': compute_psnr_batch(image_global, flat_images, GLOBAL_IMG),
            'psnr_indirect': compute_psnr_batch(
                image_indirect, flat_images, INDIRECT_IMG
            ),
        }
    )


if __name__ == '__main__':
    main()
