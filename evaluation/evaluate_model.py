"""
Trains models from experiments/model/ and saves per-frame training metrics.
"""

import gc
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

import torch
from train import train_model
from utils import (
    export_state_dict,
    preprocess_data,
    read_ground_truth_zip,
)

MODEL_EXPERIMENTS_DIR = 'evaluation/experiments/model'
MODEL_ARGS_FILE = 'data/configs/model_parameters/gpu.json'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args = json.load(open(MODEL_ARGS_FILE, 'r'))

    experiments = []
    for root, _, files in os.walk(MODEL_EXPERIMENTS_DIR):
        for f in files:
            if f.endswith('.json'):
                experiments.append(os.path.join(root, f))

    for experiment_path in experiments:
        with open(experiment_path) as f:
            experiment = json.load(f)

        name = experiment['name']
        radiance = experiment.get('radiance')
        model_key = experiment.get('model')
        train_metrics = experiment.get('train_metrics')

        if not radiance or not model_key:
            print(f'[{name}] Skipping: no radiance or model key defined')
            continue

        if not os.path.exists(radiance):
            print(f'[{name}] Warning: radiance file {radiance} not found, skipping')
            continue

        if train_metrics and os.path.exists(train_metrics):
            print(f'[{name}] Skipping: results already exist at {train_metrics}')
            continue

        print(f'[{name}] Training model')
        seed_all(experiment.get('seed', 1))
        data, parameters, transfer_function = read_ground_truth_zip(
            radiance, verbose=True, include_transfer_function=True
        )
        Xs, ys, masks = preprocess_data(data, device, verbose=True)
        model, train_losses, val_losses, baseline_losses, times = train_model(
            Xs, ys, device, **model_args, verbose=True, return_time=True
        )

        if train_metrics:
            metrics = [
                {
                    'frame': i,
                    'train_loss': train_losses[i],
                    'val_loss': val_losses[i],
                    'baseline_loss': baseline_losses[i],
                    'time_ms': times[i],
                }
                for i in range(len(train_losses))
            ]
            os.makedirs(os.path.dirname(train_metrics), exist_ok=True)
            with open(train_metrics, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f'[{name}] Metrics saved to {train_metrics}')

        os.makedirs(os.path.dirname(model_key), exist_ok=True)
        export_state_dict(
            model,
            model_args,
            model_key,
            parameters=parameters,
            transfer_function=transfer_function,
        )
        print(f'[{name}] Model saved to {model_key}')

        del model, Xs, ys, masks, train_losses, val_losses, baseline_losses, times
        del data, parameters, transfer_function
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == '__main__':
    main()
