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


def read_quality_results(dir_path):
    results = {}
    for filename in os.listdir(os.path.join(dir_path, 'path_tracing')):
        if filename.startswith('turntable'):
            _, angle, volume, extinction, tf = filename[:-4].rsplit('_', 4)
            key = volume, int(angle), int(extinction), int(tf)
        else:
            volume, extinction, tf = filename[:-4].rsplit('_', 2)
            key = volume, int(extinction), int(tf)
        results[key] = {
            'path_tracing': pd.read_csv(
                os.path.join(dir_path, 'path_tracing', filename)
            ),
            'neural_render': pd.read_csv(
                os.path.join(dir_path, 'neural_render', filename)
            ),
        }
    return results


def plot_quality_metrics(ax, results, volume, extinction, transfer_function, metric):
    pt = results[(volume, extinction, transfer_function)]['path_tracing']
    nr = results[(volume, extinction, transfer_function)]['neural_render']
    runs = len(pt['run'].unique())

    # Use the smallest run size to handle timing differences
    min_n_pt = pt.groupby('run').size().min()
    min_n_nr = nr.groupby('run').size().min()
    n = min(min_n_pt, min_n_nr)

    # Trim each run to n rows, then concatenate
    pt = pd.concat([grp.iloc[:n] for _, grp in pt.groupby('run')]).reset_index(
        drop=True
    )
    nr = pd.concat([grp.iloc[:n] for _, grp in nr.groupby('run')]).reset_index(
        drop=True
    )

    times = pt[pt['run'] == pt['run'].unique()[0]]['time'].to_numpy()

    stats_pt = pt[metric].to_numpy().reshape(runs, n)
    stats_nr = nr[metric].to_numpy().reshape(runs, n)

    mu_pt = stats_pt.mean(axis=0)
    mu_nr = stats_nr.mean(axis=0)
    se_pt = stats_pt.std(axis=0) / np.sqrt(runs)
    se_nr = stats_nr.std(axis=0) / np.sqrt(runs)

    ax.plot(times, mu_pt, label='Path Tracing', c=colors[2])
    ax.fill_between(times, mu_pt - se_pt, mu_pt + se_pt, color=colors[2], alpha=0.2)
    ax.plot(times, mu_nr, label='Neural Render', c=colors[6])
    ax.fill_between(times, mu_nr - se_nr, mu_nr + se_nr, color=colors[2], alpha=0.2)


def plot_quality_metrics_full(results, volume, extinction, transfer_function, metric):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    experiment = volume, extinction, transfer_function
    plot_quality_metrics(ax[0], results, *experiment, f'{metric}_global')
    plot_quality_metrics(ax[1], results, *experiment, f'{metric}_indirect')

    ax[0].set_title(f'Global {metric.upper()}')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel(metric.upper())
    ax[1].set_title(f'Indirect {metric.upper()}')
    ax[1].set_xlabel('Time [s]')

    legend = ax[0].legend(
        loc='lower center',
        bbox_to_anchor=(1, -0.35),
        ncol=2,
        borderaxespad=0,
    )
    set_legend_style(legend)

    plt.show()


def plot_quality_metrics_turntable(ax, results, metric):
    angles = sorted({angle for _, angle, _, _ in results.keys()})
    times = [0.5, 10]

    mu_pt = np.zeros((len(angles), len(times)))
    mu_nr = np.zeros((len(angles), len(times)))
    se_pt = np.zeros((len(angles), len(times)))
    se_nr = np.zeros((len(angles), len(times)))

    for i, angle in enumerate(angles):
        pt = results[('chameleon', angle, 200, 3)]['path_tracing']
        nr = results[('chameleon', angle, 200, 3)]['neural_render']
        runs = len(pt['run'].unique())

        for j, time in enumerate(times):
            # Find the closest time entry for each run (there are some time variations)
            stats_pt = pt.groupby('run').apply(
                lambda g: g.iloc[(g['time'] - time).abs().argmin()][metric]
            )
            stats_nr = nr.groupby('run').apply(
                lambda g: g.iloc[(g['time'] - time).abs().argmin()][metric]
            )

            mu_pt[i, j] = stats_pt.mean()
            mu_nr[i, j] = stats_nr.mean()
            se_pt[i, j] = stats_pt.std() / np.sqrt(runs)
            se_nr[i, j] = stats_nr.std() / np.sqrt(runs)

    for j, time in enumerate(times):
        ls = '--' if j == 0 else '-'
        ax.plot(
            angles,
            mu_pt[:, j],
            label=f'Path Tracing {time}s',
            ls=ls,
            c=colors[2],
            marker='o',
            markerfacecolor='white',
        )
        ax.fill_between(
            angles,
            mu_pt[:, j] - se_pt[:, j],
            mu_pt[:, j] + se_pt[:, j],
            alpha=0.2,
            color=colors[2],
            edgecolor=None,
        )
        ax.plot(
            angles,
            mu_nr[:, j],
            label=f'Neural Render {time}s',
            ls=ls,
            c=colors[6],
            marker='o',
            markerfacecolor='white',
        )
        ax.fill_between(
            angles,
            mu_nr[:, j] - se_nr[:, j],
            mu_nr[:, j] + se_nr[:, j],
            alpha=0.2,
            color=colors[6],
            edgecolor=None,
        )


def plot_quality_metrics_turntable_full(results):
    angles = sorted({angle for _, angle, _, _ in results.keys()})

    fig, ax = plt.subplots(3, 2, figsize=(7, 9), sharex=True, sharey='row')
    for i, metric in enumerate(['ssim', 'lpips', 'psnr']):
        plot_quality_metrics_turntable(ax[i, 0], results, f'{metric}_global')
        plot_quality_metrics_turntable(ax[i, 1], results, f'{metric}_indirect')

        if i == 0:
            ax[i, 0].annotate(
                'Global Illumination',
                xy=(0.5, 1.05),
                xycoords='axes fraction',
                ha='center',
                va='bottom',
                fontsize=9,
            )
            ax[i, 1].annotate(
                'Indirect Illumination',
                xy=(0.5, 1.05),
                xycoords='axes fraction',
                ha='center',
                va='bottom',
                fontsize=9,
            )

        if i == 2:
            ax[i, 0].set_xlabel('Angle [°]')
            ax[i, 1].set_xlabel('Angle [°]')
            ax[i, 0].set_xticks(angles[::2])
            ax[i, 1].set_xticks(angles[::2])
        else:
            ax[i, 0].set_xticks([])
            ax[i, 1].set_xticks([])

        ax[i, 0].set_ylabel(metric.upper())
        ax[i, 0].set_xlim(angles[0], angles[-1])
        ax[i, 1].set_xlim(angles[0], angles[-1])
        ax[i, 1].annotate(
            metric.upper(),
            xy=(1.05, 0.5),
            xycoords='axes fraction',
            ha='left',
            va='center',
            rotation=270,
            fontsize=9,
        )

    legend = ax[i, 0].legend(ncols=4, loc='upper center', bbox_to_anchor=(1, -0.2))
    set_legend_style(legend)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def compute_metric(results, volume, extinction, transfer_function, column, time):
    pt = results[(volume, extinction, transfer_function)]['path_tracing']
    nr = results[(volume, extinction, transfer_function)]['neural_render']

    # Find the closest time entry for each run (there are some time variations)
    pt = pt.groupby('run').apply(lambda df: df.iloc[(df['time'] - time).abs().argmin()])
    nr = nr.groupby('run').apply(lambda df: df.iloc[(df['time'] - time).abs().argmin()])

    pt = pt[[column]]
    nr = nr[[column]]

    mu_pt = pt.mean().to_numpy()[0].item()
    mu_nr = nr.mean().to_numpy()[0].item()
    se_pt = (pt.std().to_numpy()[0] / np.sqrt(len(pt))).item()
    se_nr = (nr.std().to_numpy()[0] / np.sqrt(len(nr))).item()

    return (mu_pt, se_pt), (mu_nr, se_nr)


def print_quality_metrics(results, volume, time, mode):
    extinctions = sorted({extinction for _, extinction, _ in results.keys()})
    transfer_functions = sorted(
        {transfer_function for _, _, transfer_function in results.keys()}
    )

    print(f'Volume: {volume} ({mode})')
    print(
        f'{"Ext.":<5} {"TF":<4} '
        f'{"SSIM PT":<18} {"SSIM NR":<18} '
        f'{"LPIPS PT":<18} {"LPIPS NR":<18} '
        f'{"PSNR PT":<15} {"PSNR NR":<15}'
    )
    for extinction in extinctions:
        for transfer_function in transfer_functions:
            pt_ssim, nr_ssim = compute_metric(
                results, volume, extinction, transfer_function, f'ssim_{mode}', time
            )
            pt_lpips, nr_lpips = compute_metric(
                results, volume, extinction, transfer_function, f'lpips_{mode}', time
            )
            pt_psnr, nr_psnr = compute_metric(
                results, volume, extinction, transfer_function, f'psnr_{mode}', time
            )

            print(
                f'{extinction:<5} {transfer_function:<4} '
                f'{pt_ssim[0]:.4f} ± {pt_ssim[1]:.4f}    '
                f'{nr_ssim[0]:.4f} ± {nr_ssim[1]:.4f}    '
                f'{pt_lpips[0]:.4f} ± {pt_lpips[1]:.4f}    '
                f'{nr_lpips[0]:.4f} ± {nr_lpips[1]:.4f}    '
                f'{pt_psnr[0]:.2f} ± {pt_psnr[1]:.2f}    '
                f'{nr_psnr[0]:.2f} ± {nr_psnr[1]:.2f}'
            )
