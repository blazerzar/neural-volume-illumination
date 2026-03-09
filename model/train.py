"""
Usage:
    python train.py <data_paths> <parameters_path> [options]

Options:
    --show-images                    Display images
    --pause                          Pause after each image
    --wait                           Wait for a key press at the start
    --verbose                        Show loading and training progress
"""

import json
import time
from sys import argv, exit

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from radiance_field_network import RadianceFieldNetwork
from tqdm import tqdm
from utils import create_image, preprocess_data, read_ground_truth_zip

import wandb

WANDB_PROJECT = 'neural-volume-illumination'
WANDB_NAME = 'radiance-field-network'


def main():
    if len(argv) < 3:
        print(__doc__)
        exit(1)

    show_images = False
    wait = False
    pause = False
    for i in range(3, len(argv)):
        show_images |= argv[i] == '--show-images'
        pause |= argv[i] == '--pause'
        wait |= argv[i] == '--wait'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(argv[2], 'r') as f:
        model_parameters = json.load(f)
    Xs, ys, masks, parameters = [], [], [], []
    for path in argv[1].split(','):
        data, param = read_ground_truth_zip(path, verbose=True)
        parameters.append(param)
        X, y, m = preprocess_data(data, device, verbose=True)
        Xs.extend(X)
        ys.extend(y)
        masks.extend(m)

    with wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        config={
            'epochs': model_parameters['epochs'],
            'learning_rate': model_parameters['lr'],
            'model_args': model_parameters['model_args'],
        },
    ) as run:
        train_model(
            Xs,
            ys,
            device,
            **model_parameters,
            verbose=True,
            run=run,
            show_images=show_images,
            resolution=parameters[0]['resolution'],
            masks=masks,
            wait=wait,
            pause=pause,
        )


def train_model(
    Xs,
    ys,
    device,
    epochs,
    lr,
    model_args,
    verbose=False,
    run=None,
    show_images=False,
    resolution=None,
    masks=None,
    wait=False,
    pause=False,
):
    """Train the Radiance Field Network on the given data with specified
    hyperparameters.

    Parameters:
        - Xs: List of input tensors for each frame.
        - ys: List of target tensors for each frame.
        - epochs: Number of training epochs for each frame.
        - lr: Learning rate for the optimizer.
        - model_args: Dictionary of arguments to initialize the model.
        - verbose: If True, print training progress and losses.
        - run: Optional Weights & Biases run object for logging.
        - show_images: If True, display all original and predicted images.
        - resolution: Image resolution, required if show_images is True.
        - masks: Boolean masks of valid pixels, required if show_images is True.
        - wait: If True, wait until a key press at the start.
        - pause: If True, pause after displaying each image until user input.

    Returns:
        - model
        - training losses
        - validation losses
        - baseline losses
    """
    assert not show_images or (resolution is not None and masks is not None), (
        'Resolution and masks are required to show images.'
    )

    model = RadianceFieldNetwork(**model_args).to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    baseline_losses = []
    train_losses = []
    val_losses = []

    if show_images:
        cv2.namedWindow('Radiance Field Network', cv2.WINDOW_NORMAL)

    pbar = tqdm(enumerate(zip(Xs, ys)), disable=not verbose, total=len(Xs))
    for i, (X, y) in pbar:
        if model.output == 'log_radiance':
            y = radiance_exponents(y, psi=model.psi)

        # Validate on data before using it for training
        model.eval()
        predicted = model(X)
        val_loss = loss_fn(predicted, y).item()
        if show_images and masks is not None and resolution is not None:
            original = create_image(y, masks[i], resolution)
            predicted_image = create_image(predicted, masks[i], resolution)
            combined = np.hstack(
                [
                    np.pad(original, [(5, 5), (5, 5), (0, 0)]),
                    np.pad(predicted_image, [(5, 5), (5, 5), (0, 0)]),
                ]
            ).astype(np.float32)
            combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.imshow('Radiance Field Network', combined)
            if pause or i == 0 and wait:
                cv2.waitKey(0)
            else:
                cv2.waitKey(10)

        model.train()
        start = time.perf_counter()
        loss = torch.tensor(0)
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        end = time.perf_counter()

        mean_predicts = y.mean(dim=0).unsqueeze(0).expand_as(y)
        baseline_losses.append(loss_fn(mean_predicts, y).item())
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        pbar.set_postfix(
            train_loss=loss.item(),
            val_loss=val_loss,
            ms=(end - start) * 1000,
        )

        if run:
            run.log(
                {
                    'train_loss': train_losses[-1],
                    'val_loss': val_losses[-1],
                    'baseline_loss': baseline_losses[-1],
                    'ms': (end - start) * 1000,
                }
            )

    if show_images:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return model, train_losses, val_losses, baseline_losses


def radiance_exponents(y, psi=4):
    return torch.pow(
        10,
        -torch.where(
            y > 1 / 10**psi,
            -torch.log10(y) / psi,
            1,
        ),
    )


if __name__ == '__main__':
    main()
