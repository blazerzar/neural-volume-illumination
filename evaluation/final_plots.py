import os

from evaluate_utils import (
    plot_model_loss_all,
    read_model_results,
)
from plot_utils import set_thesis_plot_style

PLOTS_DIR = os.path.join('evaluation', 'results', 'thesis_plots')
TEXT_WIDTH = 390 / 72


def main():
    set_thesis_plot_style()

    model_plots()


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
        fig, _ = plot_model_loss_all(
            results,
            volume,
            ylim,
            layers=['front', 'turntable'],
            figsize=figsize,
        )
        fig.savefig(
            os.path.join(PLOTS_DIR, f'model_{volume}.pdf'),
            bbox_inches='tight',
        )


if __name__ == '__main__':
    main()
