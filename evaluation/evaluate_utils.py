import json
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
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


def read_performance_results(dir_path):
    results = {}
    for filename in os.listdir(dir_path):
        if not filename.endswith('.csv'):
            continue
        volume, extinction, *_ = filename[:-4].rsplit('_', 2)
        file_path = os.path.join(dir_path, filename)
        data = pd.read_csv(file_path)
        results[(volume, int(extinction))] = data
    return results


def plot_performance_stages(ax, results, experiment, stages, colors):
    df = results[experiment]
    ax.stackplot(
        df['time'],
        *[df[c] for c in stages],
        labels=stages,
        colors=colors,
    )


def compute_performance_speedup(results):
    volumes = sorted({volume for volume, _ in results['path_tracing']})
    extinctions = sorted({extinction for _, extinction in results['path_tracing']})

    df = pd.DataFrame(
        [
            {
                'volume': volume,
                'extinction': extinction,
                'ours_Ld': results['neural_render'][volume, extinction][
                    'stage_direct'
                ].mean(),
                'ours_Ld_se': results['neural_render'][volume, extinction][
                    'stage_direct'
                ].std()
                / np.sqrt(len(results['neural_render'][volume, extinction])),
                'ours_Li': results['neural_render'][volume, extinction][
                    'stage_indirect'
                ].mean(),
                'ours_Li_se': results['neural_render'][volume, extinction][
                    'stage_indirect'
                ].std()
                / np.sqrt(len(results['neural_render'][volume, extinction])),
                'ours_L': results['neural_render'][volume, extinction][
                    'stage_direct'
                ].mean()
                + results['neural_render'][volume, extinction]['stage_indirect'].mean(),
                'pt_Ld': results['path_tracing'][volume, extinction][
                    'stage_direct'
                ].mean(),
                'pt_Ld_se': results['path_tracing'][volume, extinction][
                    'stage_direct'
                ].std()
                / np.sqrt(len(results['path_tracing'][volume, extinction])),
                'pt_Li': results['path_tracing'][volume, extinction][
                    'stage_indirect'
                ].mean(),
                'pt_Li_se': results['path_tracing'][volume, extinction][
                    'stage_indirect'
                ].std()
                / np.sqrt(len(results['path_tracing'][volume, extinction])),
                'pt_L': results['path_tracing'][volume, extinction][
                    'stage_direct'
                ].mean()
                + results['path_tracing'][volume, extinction]['stage_indirect'].mean(),
                'ours_ft': results['neural_render'][volume, extinction][
                    'frame_time'
                ].mean(),
                'ours_ft_se': results['neural_render'][volume, extinction][
                    'frame_time'
                ].std()
                / np.sqrt(len(results['neural_render'][volume, extinction])),
                'ours_fps': results['neural_render'][volume, extinction]['fps'].mean(),
                'ours_fps_se': results['neural_render'][volume, extinction]['fps'].std()
                / np.sqrt(len(results['neural_render'][volume, extinction])),
                'pt_ft': results['path_tracing'][volume, extinction][
                    'frame_time'
                ].mean(),
                'pt_ft_se': results['path_tracing'][volume, extinction][
                    'frame_time'
                ].std()
                / np.sqrt(len(results['path_tracing'][volume, extinction])),
                'pt_fps': results['path_tracing'][volume, extinction]['fps'].mean(),
                'pt_fps_se': results['path_tracing'][volume, extinction]['fps'].std()
                / np.sqrt(len(results['path_tracing'][volume, extinction])),
            }
            for volume in volumes
            for extinction in extinctions
        ]
    )

    df['speedup_Li'] = df['pt_Li'] / df['ours_Li']
    df['speedup_L'] = df['pt_L'] / df['ours_L']
    df['speedup_ft'] = df['pt_ft'] / df['ours_ft']

    overall = df[
        ['ours_Ld', 'ours_Li', 'ours_L', 'pt_Ld', 'pt_Li', 'pt_L', 'ours_ft', 'pt_ft']
    ].sum()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        'volume': 'Overall',
                        'extinction': None,
                        'ours_Ld': overall['ours_Ld'],
                        'ours_Ld_se': None,
                        'ours_Li': overall['ours_Li'],
                        'ours_Li_se': None,
                        'ours_L': overall['ours_L'],
                        'pt_Ld': overall['pt_Ld'],
                        'pt_Ld_se': None,
                        'pt_Li': overall['pt_Li'],
                        'pt_Li_se': None,
                        'pt_L': overall['pt_L'],
                        'ours_ft': overall['ours_ft'],
                        'ours_ft_se': None,
                        'ours_fps': None,
                        'ours_fps_se': None,
                        'pt_ft': overall['pt_ft'],
                        'pt_ft_se': None,
                        'pt_fps': None,
                        'pt_fps_se': None,
                        'speedup_Li': overall['pt_Li'] / overall['ours_Li'],
                        'speedup_L': overall['pt_L'] / overall['ours_L'],
                        'speedup_ft': overall['pt_ft'] / overall['ours_ft'],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    return df


def print_stage_timing(timings):
    print(
        f'{"Dataset":<20} {"Ext.":<7} '
        f'{"Ours Ld":<16} {"Ours Li":<16} {"Ours L":<10} '
        f'{"PT Ld":<16} {"PT Li":<16} {"PT L":<10} '
        f'{"S Li":<5} {"S L":<5}'
    )
    for _, row in timings.iterrows():
        if row['volume'] == 'Overall':
            print(
                f'{"Overall":<20} {"":<7} '
                f'{row["ours_Ld"]:<16.2f} {row["ours_Li"]:<16.2f} {row["ours_L"]:<10.2f} '
                f'{row["pt_Ld"]:<16.2f} {row["pt_Li"]:<16.2f} {row["pt_L"]:<10.2f} '
                f'{row["speedup_Li"]:<5.2f} {row["speedup_L"]:<5.2f}'
            )
        else:
            volume_label = (
                row['volume']
                if row.name == timings[timings['volume'] == row['volume']].index[0]
                else ''
            )
            print(
                f'{volume_label:<20} '
                f'{str(row["extinction"]):<7} '
                f'{f"{row['ours_Ld']:.2f} ± {row['ours_Ld_se']:.2f}":<16} '
                f'{f"{row['ours_Li']:.2f} ± {row['ours_Li_se']:.2f}":<16} '
                f'{row["ours_L"]:<10.2f} '
                f'{f"{row['pt_Ld']:.2f} ± {row['pt_Ld_se']:.2f}":<16} '
                f'{f"{row['pt_Li']:.2f} ± {row['pt_Li_se']:.2f}":<16} '
                f'{row["pt_L"]:<10.2f} '
                f'{row["speedup_Li"]:<5.2f} '
                f'{row["speedup_L"]:<5.2f}'
            )


def print_frame_timing(timings):
    print(
        f'{"Dataset":<20} {"Ext.":<7} '
        f'{"Ours Frame Time":<20} {"Ours FPS":<20} '
        f'{"PT Frame Time":<20} {"PT FPS":<20} '
        f'{"Speedup":<7}'
    )
    for _, row in timings.iterrows():
        if row['volume'] == 'Overall':
            print(
                f'{"Overall":<20} {"":<7} '
                f'{row["ours_ft"]:<20.2f} {"":<20} '
                f'{row["pt_ft"]:<20.2f} {"":<20} '
                f'{row["speedup_ft"]:<7.2f}'
            )
        else:
            volume_label = (
                row['volume']
                if row.name == timings[timings['volume'] == row['volume']].index[0]
                else ''
            )
            print(
                f'{volume_label:<20} '
                f'{str(row["extinction"]):<7} '
                f'{f"{row['ours_ft']:.2f} ± {row['ours_ft_se']:.2f}":<20} '
                f'{f"{row['ours_fps']:.2f} ± {row['ours_fps_se']:.2f}":<20} '
                f'{f"{row['pt_ft']:.2f} ± {row['pt_ft_se']:.2f}":<20} '
                f'{f"{row['pt_fps']:.2f} ± {row['pt_fps_se']:.2f}":<20} '
                f'{row["speedup_ft"]:<7.2f}'
            )


def plot_speedups(ax, timings, speedup_col, colors, add_labels=False):
    df = timings[timings['extinction'].notna()]
    volumes = sorted(df['volume'].unique())
    extinctions = sorted(df['extinction'].unique())
    speedups = df.set_index(['volume', 'extinction'])[speedup_col]
    for i, volume in enumerate(reversed(volumes)):
        for j, extinction in enumerate(extinctions):
            ax.barh(
                i - (j - 1) * 0.22,
                speedups[volume, extinction],
                color=colors.get(extinction),
                height=0.22,
                label=str(extinction) if (add_labels and i == 0) else None,
                edgecolor='black',
                linewidth=0.8,
            )
    ax.axvspan(0, 1, color='lightgray', alpha=0.5, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Speedup')
