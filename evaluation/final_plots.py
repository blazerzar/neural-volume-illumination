import os

from evaluate_utils import (
    plot_model_loss_all,
    plot_online_loss_tf,
    read_model_results,
)
from plot_utils import set_thesis_plot_style

PLOTS_DIR = os.path.join('evaluation', 'results', 'thesis_plots')
TEXT_WIDTH = 390 / 72


def main():
    set_thesis_plot_style()

    model_plots()
    online_plots()


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
            from plot_utils import colors

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


if __name__ == '__main__':
    main()
