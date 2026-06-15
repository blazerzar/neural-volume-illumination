import json
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_utils import colors, set_legend_style

PIXEL_VALUES = 8


def read_model_results(dir_path):
    results = {}
    for filename in os.listdir(dir_path):
        if not filename.endswith('.csv'):
            continue
        volume, extinction, transfer_function = filename[:-4].rsplit('_', 2)
        file_path = os.path.join(dir_path, filename)
        data = pd.read_csv(file_path)
        results[(volume, int(extinction), int(transfer_function))] = data
    return results


def plot_model_loss(
    ax,
    results,
    experiment,
    smoothing_window=5,
    label=None,
    styles=None,
):
    styles = styles or {}
    styles.setdefault('plot', {})
    styles.setdefault('baseline', {})

    values = results[experiment]['val_loss']
    values = np.convolve(
        values, np.ones(smoothing_window) / smoothing_window, mode='valid'
    )
    ax.plot(values, label=label, **styles['plot'])

    baselines = results[experiment]['baseline_loss']
    ax.axhline(
        y=np.mean(baselines),
        **styles['baseline'],
        label=(f'Baseline {label}' if label else None),
    )


def plot_model_loss_all(results, volume, ylims, layers=None, show_legend=True):
    _, ax = plt.subplots(3, 3, figsize=(10, 10))

    results = [results] if layers is None else [results[layer] for layer in layers]
    extinctions = sorted({ext for _, ext, _ in results[0].keys()})
    transfer_functions = sorted({tf for _, _, tf in results[0].keys()})

    baseline_styles = [
        dict(color=colors[2], linestyle='-', linewidth=1.5, zorder=1),
        dict(color=colors[6], linestyle='--', linewidth=1.5, zorder=1),
    ]

    for i, tf in enumerate(transfer_functions):
        for j, extinction in enumerate(extinctions):
            for k, res in enumerate(results):
                plot_model_loss(
                    ax[i, j],
                    res,
                    (volume, extinction, tf),
                    label=layers[k].capitalize() if layers else None,
                    styles={
                        'plot': dict(color='k', linestyle=('-', '--')[k]),
                        'baseline': baseline_styles[k],
                    },
                )

            ax[i, j].set_xlim(0, 500)
            ax[i, j].set_ylim(ylims[i])
            if i < 2:
                ax[i, j].set_xticks([])
            else:
                ax[i, j].set_xticks([0, 100, 200, 300, 400])
            if j > 0:
                ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].annotate(
                    'Extinction ' + str(extinction),
                    xy=(0.5, 1.05),
                    xycoords='axes fraction',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )
            if j == 2:
                ax[i, j].annotate(
                    'Transfer Function ' + str(tf),
                    xy=(1.05, 0.5),
                    xycoords='axes fraction',
                    ha='left',
                    va='center',
                    rotation=270,
                    fontsize=9,
                )

    ax[2, 1].set_xlabel('Frame')
    ax[1, 0].set_ylabel('Loss')

    if show_legend:
        handles, labels = ax[2, 1].get_legend_handles_labels()
        order = ['Front', 'Turntable', 'Baseline Front', 'Baseline Turntable']
        ordered = [
            (h, la) for name in order for h, la in zip(handles, labels) if la == name
        ]
        h_sorted, l_sorted = zip(*ordered)
        legend = ax[2, 1].legend(
            h_sorted, l_sorted, ncol=4, bbox_to_anchor=(0.5, -0.15), loc='upper center'
        )
        set_legend_style(legend)

    plt.subplots_adjust(wspace=0, hspace=0)


def read_ground_truth_zip(file_path, array_file):
    array = None
    with zipfile.ZipFile(file_path, mode='r') as archive:
        files = sorted(archive.namelist())
        parameters = {}

        for file in files:
            if file.endswith('.bin') and file == array_file:
                buffer_bytes = archive.read(file)
                array = np.frombuffer(buffer_bytes, dtype=np.float32)
                array = array.reshape(-1, PIXEL_VALUES)
            elif file == 'parameters.json':
                buffer_bytes = archive.read(file)
                parameters = json.loads(buffer_bytes.decode('utf-8'))

    return array, parameters
