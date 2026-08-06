import os

import numpy as np
import pandas as pd
from evaluate_utils import read_model_results


def main():
    training_tables()


def training_tables():
    times = pd.DataFrame(
        columns=['Volume', 'Extinction', 'tf', 'camera', 'type', 'time']
    )
    col = {'model': 'time_ms', 'online': 'train_time_ms'}

    dir = os.path.join('evaluation', 'results')
    for experiment in ['model', 'online']:
        for subdir in os.listdir(os.path.join(dir, experiment)):
            if subdir not in ['front', 'turntable']:
                continue
            results = read_model_results(os.path.join(dir, experiment, subdir))

            for (volume, extinction, tf), df in results.items():
                new_rows = pd.DataFrame(
                    {
                        'Volume': volume,
                        'Extinction': extinction,
                        'tf': tf,
                        'camera': subdir,
                        'type': experiment,
                        'time': df[col[experiment]].to_numpy(),
                    }
                )
                times = pd.concat([times, new_rows], ignore_index=True)

    stats = (
        times.groupby(['Volume', 'Extinction', 'type'])['time']
        .agg(['mean', 'sem'])
        .unstack('type')
    )

    table = pd.DataFrame(index=stats.index)

    # Format numbers
    for kind, label in [('model', 'Offline'), ('online', 'Online')]:
        m = stats[('mean', kind)].map('{:.2f}'.format)
        s = stats[('sem', kind)].map('{:.2f}'.format)
        table[label] = '$' + m + r' \pm ' + s + '$'
    table = (
        table.reset_index().sort_values(['Volume', 'Extinction']).reset_index(drop=True)
    )
    table['Extinction'] = table['Extinction'].map('${:d}$'.format)

    # Only show volume once, split by \cline
    vol = table['Volume']
    label = r'\textsf{' + vol.str.replace('_', ' ') + '}'
    rule = pd.Series(
        np.where(np.arange(len(vol)), r'\cline{1-4}' + '\n', ''), index=vol.index
    )
    table['Volume'] = (rule + label).where(~vol.duplicated(), '')

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format='lrcc',
        label='tab:times',
        caption=r'\captiontimes{}',
    )
    latex = postprocess_latex_table(latex)
    print(latex)


def postprocess_latex_table(latex):
    """
    Move caption and label to the bottom of the table, add centering and
    footnotesize, bold column names, and indent lines.
    """
    lines = latex.splitlines()

    header_line = lines.index(r'\toprule') + 1
    cols = lines[header_line].replace(r' \\', '').split(' & ')
    cols = [r'\textbf{' + col + '}' for col in cols]
    lines[header_line] = ' & '.join(cols) + r' \\'

    caption_line = lines[1]
    label_line = lines[2]

    lines = (
        lines[:1]
        + [r'\centering', r'\footnotesize']
        + lines[3:-1]
        + [caption_line, label_line]
        + lines[-1:]
    )

    for i in range(1, len(lines) - 1):
        indent = 8 if 3 < i < len(lines) - 4 else 4
        lines[i] = indent * ' ' + lines[i]

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
