"""
Reads experiment declarations and pretrains all the needed models.
Run after prepare_radiance.py to ensure radiance data is available.
"""

import hashlib
import json
import logging
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'model'))

from train import train_model
from utils import (
    export_state_dict,
    preprocess_data,
    read_ground_truth_zip,
)

RADIANCE_DIR = 'data/radiance'
EXPERIMENTS_DIRS = [
    'evaluation/experiments/performance',
    'evaluation/experiments/quality/front/neural_render',
]
MODEL_ARGS_FILE = 'data/configs/model_parameters/gpu.json'

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def find_experiments(root):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.json'):
                yield os.path.join(dirpath, filename)


def seed_from_name(name):
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args = json.load(open(MODEL_ARGS_FILE, 'r'))

    for experiments_dir in EXPERIMENTS_DIRS:
        for experiment_path in find_experiments(experiments_dir):
            with open(experiment_path) as f:
                experiment = json.load(f)

            name = experiment.get('name')
            radiance = experiment.get('radiance')
            model = experiment.get('model')

            if not radiance or not model:
                continue

            model_path = model
            if os.path.exists(model_path):
                logging.info(f'[{name}] Skipping: model already exists')
                continue

            radiance_path = os.path.join(RADIANCE_DIR, os.path.basename(radiance))
            if not os.path.exists(radiance_path):
                logging.warning(
                    f'[{name}] Radiance file {radiance_path} not found, skipping'
                )
                continue

            logging.info(f'[{name}] Training model')
            seed = experiment.get('seed') or seed_from_name(name)
            seed_all(seed)
            data, parameters, transfer_function = read_ground_truth_zip(
                radiance_path, verbose=True, include_transfer_function=True
            )
            Xs, ys, masks = preprocess_data(data, device, verbose=True)
            model, *_ = train_model(Xs, ys, device, **model_args, verbose=True)

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            export_state_dict(
                model,
                model_args['model_args'],
                model_path,
                parameters=parameters,
                transfer_function=transfer_function,
            )
            logging.info(f'[{name}] Model saved to {model_path}')


if __name__ == '__main__':
    main()
