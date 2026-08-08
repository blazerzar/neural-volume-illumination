import os

import numpy as np
import pandas as pd
from evaluate_utils import (
    compute_performance_speedup,
    read_model_results,
    read_performance_results,
)


def main():
    training_tables()
    speedup_tables()
    high_samples_table()


def training_tables():
    times = pd.DataFrame(
        columns=['Dataset', 'Extinction', 'tf', 'camera', 'type', 'time']
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
                        'Dataset': volume,
                        'Extinction': extinction,
                        'tf': tf,
                        'camera': subdir,
                        'type': experiment,
                        'time': df[col[experiment]].to_numpy(),
                    }
                )
                times = pd.concat([times, new_rows], ignore_index=True)

    stats = (
        times.groupby(['Dataset', 'Extinction', 'type'])['time']
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
        table.reset_index()
        .sort_values(['Dataset', 'Extinction'])
        .reset_index(drop=True)
    )
    table['Extinction'] = table['Extinction'].map('${:d}$'.format)

    # Only show dataset once, split by \cline
    vol = table['Dataset']
    label = r'\textsf{' + vol.str.replace('_', ' ') + '}'
    rule = pd.Series(
        np.where(np.arange(len(vol)), r'\cline{1-4}' + '\n', ''), index=vol.index
    )
    table['Dataset'] = (rule + label).where(~vol.duplicated(), '')

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format='lrcc',
        label='tab:times',
        caption=r'\captiontimes{}',
    )
    latex = postprocess_latex_table(
        latex, ['Dataset', 'Extinction', 'Offline', 'Online']
    )
    print(latex, '\n')


def speedup_tables():
    dir = os.path.join('evaluation', 'results', 'performance')
    results = {}
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        results[subdir] = read_performance_results(subdir_path)

    timings = compute_performance_speedup(results)
    print(stage_speedup_table(timings), '\n')
    print(frame_speedup_table(timings), '\n')


def stage_speedup_table(timings):
    cols = pd.MultiIndex.from_tuples(
        [
            ('', 'Dataset'),
            ('', 'Ext.'),
            ('Ours', '$L_d$'),
            ('Ours', '$L_i$'),
            ('Ours', '$L$'),
            ('PT', '$L_d$'),
            ('PT', '$L_i$'),
            ('PT', '$L$'),
            ('', '$S_{L_i}$'),
            ('', '$S_L$'),
        ]
    )
    num = timings.drop(columns='volume').apply(pd.to_numeric, errors='coerce')
    table = pd.DataFrame('', index=timings.index, columns=cols)

    name = shorten_volume_name(timings.volume.astype(str))
    bold = name == 'Overall'
    first = ~name.duplicated()
    sep = first.copy()
    sep.iloc[0] = False
    dataset = (
        (r'\textsf{' + name.str.replace('_', ' ') + '}')
        .mask(bold, r'\textbf{Overall}')
        .where(first, '')
    )
    table[('', 'Dataset')] = (r'\cline{1-10} ' + dataset).where(sep, dataset)
    table[('', 'Ext.')] = num.extinction.map('{:.0f}'.format).where(
        num.extinction.notna(), ''
    )

    vals = pd.DataFrame(index=timings.index, columns=cols[2:])
    for grp, k in [('Ours', 'ours'), ('PT', 'pt')]:
        for st in 'di':
            vals[(grp, f'$L_{st}$')] = num[f'{k}_L{st}'].map('{:.2f}'.format)
        vals[(grp, '$L$')] = (num[f'{k}_Ld'] + num[f'{k}_Li']).map('{:.2f}'.format)
    vals[('', '$S_{L_i}$')] = num.speedup_Li.map('{:.2f}'.format)
    vals[('', '$S_L$')] = num.speedup_L.map('{:.2f}'.format)
    table[vals.columns] = np.where(
        bold.values[:, None], r'$\mathbf{' + vals + '}$', '$' + vals + '$'
    )

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format=r'l@{\hspace{7pt}}r|'
        + '|'.join([r'c@{\hspace{7pt}}c@{\hspace{7pt}}c'] * 2)
        + '|cc',
        multicolumn=True,
        multicolumn_format='c',
        caption=r'\captionspeedupsL{}',
        label='tab:speedups-L',
    )
    latex = postprocess_latex_table(latex, ['Dataset', 'Ext.', 'Ours', 'PT'])
    return latex


def frame_speedup_table(
    timings, caption=r'\captionspeedupsFT{}', label='tab:speedups-ft'
):
    cols = pd.MultiIndex.from_tuples(
        [
            ('', 'Dataset'),
            ('', 'Ext.'),
            ('Ours', r'\multicolumn{2}{c}{Frame time}'),
            ('Ours', 'SE1'),
            ('Ours', 'FPS'),
            ('PT', r'\multicolumn{2}{c}{Frame time}'),
            ('PT', 'SE2'),
            ('PT', 'FPS'),
            ('', 'Speedup'),
        ]
    )
    num = timings.drop(columns='volume').apply(pd.to_numeric, errors='coerce')
    table = pd.DataFrame('', index=timings.index, columns=cols)

    name = shorten_volume_name(timings.volume.astype(str))
    bold = name == 'Overall'
    first = ~name.duplicated()
    sep = first.copy()
    sep.iloc[0] = False
    dataset = (
        (r'\textsf{' + name.str.replace('_', ' ') + '}')
        .mask(bold, r'\textbf{Overall}')
        .where(first, '')
    )
    table[('', 'Dataset')] = (r'\cline{1-9} ' + dataset).where(sep, dataset)
    table[('', 'Ext.')] = num.extinction.map('{:.0f}'.format).where(
        num.extinction.notna(), ''
    )

    def cell(v):
        s = v.map('{:.2f}'.format)
        return ('$' + s + '$').mask(bold, r'$\mathbf{' + s + '}$').where(v.notna(), '')

    for grp, k, se_col in [('Ours', 'ours', 'SE1'), ('PT', 'pt', 'SE2')]:
        se = num[f'{k}_ft_se'].mask(bold)
        table[(grp, r'\multicolumn{2}{c}{Frame time}')] = cell(num[f'{k}_ft'])
        table[(grp, se_col)] = (r'$\pm\,' + se.map('{:.2f}'.format) + '$').where(
            se.notna(), ''
        )
        table[(grp, 'FPS')] = cell(num[f'{k}_fps'])

    table[('', 'Speedup')] = cell(num.speedup_ft)

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format=r'lr|r@{\hspace{5pt}}lc|r@{\hspace{5pt}}lc|c',
        multicolumn=True,
        multicolumn_format='c',
        caption=caption,
        label=label,
    )
    latex = latex.replace(' & SE1', '').replace(' & SE2', '')
    latex = postprocess_latex_table(latex, ['Dataset', 'Ext.', 'Ours', 'PT', 'Speedup'])
    return latex


def high_samples_table():
    dir = os.path.join('evaluation', 'results', 'high_samples')
    results = {
        'path_tracing': read_performance_results(os.path.join(dir, 'path_tracing')),
        'neural_render': read_performance_results(os.path.join(dir, 'neural_render')),
    }
    timings = compute_performance_speedup(results)
    latex = frame_speedup_table(timings, r'\captionhighsamples{}', 'tab:high-samples')

    # Remove \cline{1-9} except on row with Overall
    lines = latex.splitlines()
    overall_i = lines.index(next(l for l in lines if r'\textbf{Overall}' in l))
    lines = [
        l if i == overall_i else l.replace(r'\cline{1-9} ', '')
        for i, l in enumerate(lines)
    ]
    latex = '\n'.join(lines)
    print(latex, '\n')


def postprocess_latex_table(latex, bold_columns):
    """
    Move caption and label to the bottom of the table, add centering and
    footnotesize, bold column names, and indent lines.
    """
    for col in bold_columns:
        latex = latex.replace(col, r'\textbf{' + col + '}')
    lines = latex.splitlines()

    caption_i = lines.index(next(l for l in lines if l.startswith(r'\caption')))
    label_i = lines.index(next(l for l in lines if l.startswith(r'\label')))
    caption_line = lines[caption_i]
    label_line = lines[label_i]
    lines = [l for i, l in enumerate(lines) if i not in [caption_i, label_i]]

    lines = (
        lines[:1]
        + [r'\centering', r'\footnotesize']
        + lines[1:-1]
        + [caption_line, label_line]
        + lines[-1:]
    )

    for i in range(1, len(lines) - 1):
        indent = 8 if 3 < i < len(lines) - 4 else 4
        lines[i] = indent * ' ' + lines[i]

    return '\n'.join(lines)


def shorten_volume_name(name):
    return (
        name.replace('csafe_heptane', 'heptane')
        .replace('mri_ventricles', 'ventricles')
        .replace('marmoset_neurons', 'neurons')
    )


if __name__ == '__main__':
    main()
