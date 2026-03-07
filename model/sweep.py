"""
Usage:
    python sweep.py <data_path> <count> [options]

Options:
    --sweep <sweep_id>               Use existing sweep with given id
    --configuration <config_path>    Use sweep configuration from given path
"""

import json
from sys import argv

import torch
from train import train_model
from utils import preprocess_data, read_ground_truth_zip

import wandb

WANDB_PROJECT = 'neural-volume-illumination'
WANDB_NAME = 'radiance-field-network'


def main():
    if len(argv) < 5:
        print(__doc__)
        exit(1)

    if argv[3] == '--sweep':
        sweep_id = argv[4]
    elif argv[3] == '--configuration':
        sweep_configuration = {
            'name': WANDB_NAME,
            'method': 'bayes',
            'metric': {'goal': 'minimize', 'name': 'val_loss'},
        }
        with open(argv[4], 'r') as f:
            sweep_configuration['parameters'] = json.load(f)
        sweep_id = wandb.sweep(sweep_configuration, project=WANDB_PROJECT)
    else:
        print(__doc__)
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, parameters = read_ground_truth_zip(argv[1], verbose=True)
    Xs, ys, masks = preprocess_data(data, device, verbose=True)

    def train_sweep():
        with wandb.init() as run:
            return train_model(Xs, ys, device, **run.config, run=run, verbose=True)

    print(sweep_id)
    wandb.agent(
        sweep_id, project=WANDB_PROJECT, function=train_sweep, count=int(argv[2])
    )
    wandb.teardown()


if __name__ == '__main__':
    main()
