import os

import matplotlib.pyplot as plt
from evaluate_utils import (
    compute_performance_speedup,
    plot_model_loss_all,
    plot_online_loss_tf,
    plot_performance_stages,
    plot_quality_metrics,
    plot_quality_metrics_turntable,
    plot_speedups,
    read_model_results,
    read_performance_results,
    read_quality_results,
)
from plot_utils import colors, set_legend_style, set_thesis_plot_style

PLOTS_DIR = os.path.join('evaluation', 'results', 'thesis_plots')
TEXT_WIDTH = 390 / 72


def main():
    set_thesis_plot_style()

    model_plots()
    online_plots()
    performance_plots()
    quality_plots()


def model_plots():
    dir = os.path.join('evaluation', 'results', 'model')
    cameras = 'front', 'turntable'

    results = {}
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if not os.path.isdir(subdir_path) or subdir not in cameras:
            continue
        results[subdir] = read_model_results(subdir_path)

    ylims = {
        'chameleon': [(0.0085, 0.019), (0.008, 0.116), (0.01, 0.125)],
        'mri_ventricles': [(0.031, 0.064), (0.008, 0.116), (0.022, 0.069)],
    }
    figsize = (TEXT_WIDTH, TEXT_WIDTH * 0.9)

    for volume, ylim in ylims.items():
        fig, ax = plot_model_loss_all(
            results,
            volume,
            ylim,
            layers=['front', 'turntable'],
            figsize=figsize,
        )
        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.18, top=0.93, right=0.93)

        if volume == 'chameleon':
            xs = [80, 135]
            ys = [
                results['turntable']['chameleon', 200, 3].iloc[80]['val_loss'],
                results['turntable']['chameleon', 200, 3].iloc[135]['val_loss'],
            ]
            tags = ['80', '135']

            ax[2, 1].scatter(
                xs,
                ys,
                color='black',
                marker='o',
                facecolors='white',
                s=15,
                zorder=10,
                linewidth=0.8,
            )
            for x, y, tag, xytext in zip(xs, ys, tags, [(4, -10), (10, 3)]):
                ax[2, 1].annotate(
                    tag,
                    xy=(x, y),
                    xytext=xytext,
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                )

        fig.savefig(os.path.join(PLOTS_DIR, f'model_{volume}.pdf'))
        plt.close()


def online_plots():
    dir = os.path.join('evaluation', 'results', 'online')

    results = {}
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        results[subdir] = read_model_results(subdir_path)

    fig, _ = plot_online_loss_tf(
        results,
        'chameleon',
        3,
        (0, 0.125),
        figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.45),
    )
    fig.subplots_adjust(wspace=0, bottom=0.36, right=0.93)
    fig.savefig(os.path.join(PLOTS_DIR, 'online_chameleon.pdf'))
    plt.close()


def performance_plots():
    dir = os.path.join('evaluation', 'results', 'performance')
    results = {}
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        results[subdir] = read_performance_results(subdir_path)

    plot_stages(results)

    timings = compute_performance_speedup(results)
    speedup_colors = {80: colors[0], 200: colors[2], 1000: colors[5]}
    volumes = sorted({v for r in results.values() for v, _ in r})

    plot_stage_speedup(results, speedup_colors, timings, volumes)
    plot_frame_speedup(results, speedup_colors, timings, volumes)


def plot_stages(results):
    example = 'chameleon', 1000
    stage_cols = ['stage_sample_gen', 'stage_direct', 'stage_indirect']
    stage_colors = [colors[5], colors[0], colors[2]]

    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.45), sharey=True)

    for i, subdir in enumerate(['path_tracing', 'neural_render']):
        plot_performance_stages(
            ax[i], results[subdir], example, stage_cols, stage_colors
        )
        subdir = subdir if subdir != 'neural_render' else 'ours'
        ax[i].set_title(subdir.replace('_', ' ').capitalize(), fontsize=12)
        ax[i].set_xlim(1, 59)
        ax[i].set_xlabel('Time [s]', labelpad=10)
        if i == 0:
            ax[i].set_ylabel('Stage time [ms]', labelpad=10)

    handles, labels = ax[0].get_legend_handles_labels()
    labels = ['Sample generation', 'Direct illumination', 'Indirect illumination']
    legend = fig.legend(
        handles,
        labels,
        ncol=4,
        bbox_to_anchor=(0.5, 0.15),
        loc='upper center',
        columnspacing=1,
    )
    set_legend_style(legend)

    fig.subplots_adjust(wspace=0.15, bottom=0.36, top=0.89)
    fig.savefig(os.path.join(PLOTS_DIR, 'stage_times.pdf'))
    plt.close()


def plot_stage_speedup(results, speedup_colors, timings, volumes):
    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.55), sharey=True)

    plot_speedups(ax[0], timings, 'speedup_Li', speedup_colors, add_labels=True)
    plot_speedups(ax[1], timings, 'speedup_L', speedup_colors)

    ax[0].set_title('Indirect illumination', fontsize=12)
    ax[1].set_title('Global illumination', fontsize=12)

    volumes = sorted(timings.loc[timings['extinction'].notna(), 'volume'].unique())
    volumes = [shorten_volume_name(v) for v in volumes]
    labels = [rf'\textsf{{{v}}}' for v in reversed(volumes)]
    ax[0].set_yticks(range(len(volumes)))
    ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].tick_params(axis='y', which='both', length=0)
    ax[1].tick_params(axis='y', which='both', length=0)

    legend = fig.legend(
        loc='upper center',
        ncol=len(speedup_colors),
        bbox_to_anchor=(0.5, 0.12),
    )
    set_legend_style(legend)

    fig.subplots_adjust(wspace=0.1, bottom=0.26, top=0.91, left=0.12, right=0.99)
    fig.savefig(os.path.join(PLOTS_DIR, 'speedups_L.pdf'))
    plt.close()


def plot_frame_speedup(results, speedup_colors, timings, volumes):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.85, TEXT_WIDTH * 0.42))

    plot_speedups(ax, timings, 'speedup_ft', speedup_colors, add_labels=True)

    volumes = sorted(timings.loc[timings['extinction'].notna(), 'volume'].unique())
    volumes = [shorten_volume_name(v) for v in volumes]
    labels = [rf'\textsf{{{v}}}' for v in reversed(volumes)]
    ax.set_yticks(range(len(volumes)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis='y', which='both', length=0)

    legend = ax.legend(
        loc='lower right',
        bbox_to_anchor=(1, 0),
        ncol=1,
        borderaxespad=0,
    )
    set_legend_style(legend)

    fig.subplots_adjust(wspace=0.1, bottom=0.18, top=0.97, left=0.14, right=0.98)
    fig.savefig(os.path.join(PLOTS_DIR, 'speedups_ft.pdf'))
    plt.close()


def quality_plots():
    plot_quality_front('bonsai', 200, 3)
    plot_quality_front('silicium', 1000, 2)
    plot_quality_front('chameleon', 200, 2)
    plot_quality_turntable()


def plot_quality_front(*experiment, nest=False):
    dir = os.path.join('evaluation', 'results', 'quality', 'front')
    results = read_quality_results(dir)

    fig, ax = plt.subplots(
        3, 2, figsize=(TEXT_WIDTH * 0.8, TEXT_WIDTH * 0.6), sharey='row', sharex=True
    )

    for i, metric in enumerate(['ssim', 'lpips', 'psnr']):
        plot_quality_metrics(ax[i, 0], results, *experiment, f'{metric}_global')
        plot_quality_metrics(ax[i, 1], results, *experiment, f'{metric}_indirect')
        ax[i, 0].set_xlim(0.250, 10)
        ax[i, 1].set_xlim(0.250, 10)
        if i == 2:
            ax[i, 0].set_xlabel('Time [s]')
            ax[i, 1].set_xlabel('Time [s]')
            ax[i, 0].set_xticks([2, 4, 6, 8])
            ax[i, 1].set_xticks([2, 4, 6, 8])
        else:
            ax[i, 0].set_xticks([])
            ax[i, 1].set_xticks([])
        if i == 0:
            ax[i, 0].set_title('Global illumination', fontsize=12)
            ax[i, 1].set_title('Indirect illumination', fontsize=12)

        if metric == 'ssim':
            ax[i, 0].set_ylim(top=1.01)
            ax[i, 1].set_ylim(top=1.01)
        elif metric == 'lpips':
            ax[i, 0].set_ylim(bottom=-0.01)
            ax[i, 1].set_ylim(bottom=-0.01)

        ax[i, 0].set_ylabel(metric.upper())

    fig.align_ylabels()

    legend = ax[2, 0].legend(
        loc='lower center',
        bbox_to_anchor=(1, -1.06),
        ncol=2,
        borderaxespad=0,
    )
    set_legend_style(legend)

    fig.subplots_adjust(
        wspace=0, hspace=0, bottom=0.25, top=0.92, left=0.11, right=0.99
    )
    volume, ext, tf = experiment
    if nest:
        file_path = os.path.join(PLOTS_DIR, 'quality_front', f'{volume}_{ext}_{tf}.pdf')
    else:
        file_path = os.path.join(PLOTS_DIR, f'quality_front_{volume}_{ext}_{tf}.pdf')
    fig.savefig(file_path)
    plt.close()


def plot_quality_turntable():
    dir = os.path.join('evaluation', 'results', 'quality', 'turntable')
    results = read_quality_results(dir)
    angles = sorted({angle for _, angle, _, _ in results})

    fig, ax = plt.subplots(
        3, 2, figsize=(TEXT_WIDTH * 0.9, TEXT_WIDTH * 0.8), sharex=True, sharey='row'
    )
    for i, metric in enumerate(['ssim', 'lpips', 'psnr']):
        plot_quality_metrics_turntable(ax[i, 0], results, f'{metric}_global')
        plot_quality_metrics_turntable(ax[i, 1], results, f'{metric}_indirect')

        if i == 0:
            ax[i, 0].set_title('Global illumination', fontsize=12)
            ax[i, 1].set_title('Indirect illumination', fontsize=12)
        if i == 2:
            ax[i, 0].set_xlabel('Angle [°]')
            ax[i, 1].set_xlabel('Angle [°]')
            ax[i, 0].set_xticks(angles[::2][1:])
            ax[i, 1].set_xticks(angles[::2][1:])
        else:
            ax[i, 0].set_xticks([])
            ax[i, 1].set_xticks([])

        ax[i, 0].set_ylabel(metric.upper())
        ax[i, 0].set_xlim(angles[0], angles[-1])
        ax[i, 1].set_xlim(angles[0], angles[-1])

    fig.align_ylabels()

    handles, labels = ax[2, 0].get_legend_handles_labels()
    handles = [handles[1], handles[3], handles[0], handles[2]]
    labels = [labels[1], labels[3], labels[0], labels[2]]
    legend = ax[2, 0].legend(
        handles,
        labels,
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(1, -0.44),
    )
    set_legend_style(legend)

    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.23, top=0.94, left=0.1, right=0.99)
    fig.savefig(os.path.join(PLOTS_DIR, 'quality_turntable.pdf'))
    plt.close()


def shorten_volume_name(name):
    return (
        name.replace('csafe_heptane', 'heptane')
        .replace('mri_ventricles', 'ventricles')
        .replace('marmoset_neurons', 'neurons')
    )


if __name__ == '__main__':
    main()
