import os

import numpy as np
import pandas as pd
from evaluate_utils import (
    compute_metric,
    compute_performance_speedup,
    read_model_results,
    read_performance_results,
    read_quality_results,
)

TABLES_DIR = os.path.join('evaluation', 'results', 'thesis_tables')


def main():
    training_tables()
    speedup_tables()
    high_samples_table()
    quality_tables()
    filter_tables()


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
    vol = shorten_volume_name(table['Dataset'].astype(str))
    label = r'\textsf{' + vol + '}'
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
    save_table(latex, 'training_times.tex')


def speedup_tables():
    dir = os.path.join('evaluation', 'results', 'performance')
    results = {}
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        results[subdir] = read_performance_results(subdir_path)

    timings = compute_performance_speedup(results)
    save_table(stage_speedup_table(timings), 'stage_speedup.tex')
    save_table(frame_speedup_table(timings), 'frame_speedup.tex')


def stage_speedup_table(timings):
    cols = pd.MultiIndex.from_tuples(
        [
            ('', 'Dataset'),
            ('', 'Ext.'),
            ('Ours', '$L_d$'),
            ('Ours', '$L_i$'),
            ('Ours', '$L$'),
            ('Path tracing', '$L_d$'),
            ('Path tracing', '$L_i$'),
            ('Path tracing', '$L$'),
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
        (r'\textsf{' + name + '}').mask(bold, r'\textbf{Overall}').where(first, '')
    )
    table[('', 'Dataset')] = (r'\cline{1-10} ' + dataset).where(sep, dataset)
    table[('', 'Ext.')] = num.extinction.map('{:.0f}'.format).where(
        num.extinction.notna(), ''
    )

    vals = pd.DataFrame(index=timings.index, columns=cols[2:])
    for grp, k in [('Ours', 'ours'), ('Path tracing', 'pt')]:
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
        position='p',
    )
    latex = postprocess_latex_table(latex, ['Dataset', 'Ext.', 'Ours', 'Path tracing'])
    return latex


def frame_speedup_table(
    timings, caption=r'\captionspeedupsFT{}', label='tab:speedups-ft', position='p'
):
    cols = pd.MultiIndex.from_tuples(
        [
            ('', 'Dataset'),
            ('', 'Ext.'),
            ('Ours', r'\multicolumn{2}{c}{Frame time}'),
            ('Ours', 'SE1'),
            ('Ours', 'FPS'),
            ('Path tracing', r'\multicolumn{2}{c}{Frame time}'),
            ('Path tracing', 'SE2'),
            ('Path tracing', 'FPS'),
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
        (r'\textsf{' + name + '}').mask(bold, r'\textbf{Overall}').where(first, '')
    )
    table[('', 'Dataset')] = (r'\cline{1-9} ' + dataset).where(sep, dataset)
    table[('', 'Ext.')] = num.extinction.map('{:.0f}'.format).where(
        num.extinction.notna(), ''
    )

    def cell(v):
        s = v.map('{:.2f}'.format)
        return ('$' + s + '$').mask(bold, r'$\mathbf{' + s + '}$').where(v.notna(), '')

    for grp, k, se_col in [('Ours', 'ours', 'SE1'), ('Path tracing', 'pt', 'SE2')]:
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
        position=position,
    )
    latex = latex.replace(' & SE1', '').replace(' & SE2', '')
    latex = postprocess_latex_table(
        latex, ['Dataset', 'Ext.', 'Ours', 'Path tracing', 'Speedup']
    )
    return latex


def high_samples_table():
    dir = os.path.join('evaluation', 'results', 'high_samples')
    results = {
        'path_tracing': read_performance_results(os.path.join(dir, 'path_tracing')),
        'neural_render': read_performance_results(os.path.join(dir, 'neural_render')),
    }
    timings = compute_performance_speedup(results)
    latex = frame_speedup_table(
        timings, r'\captionhighsamples{}', 'tab:high-samples', position='h!'
    )

    # Remove \cline{1-9} except on row with Overall
    lines = latex.splitlines()
    overall_i = lines.index(next(l for l in lines if r'\textbf{Overall}' in l))
    lines = [
        l if i == overall_i else l.replace(r'\cline{1-9} ', '')
        for i, l in enumerate(lines)
    ]
    latex = '\n'.join(lines)
    save_table(latex, 'high_samples.tex')


def quality_tables():
    dir = os.path.join('evaluation', 'results', 'quality', 'front')
    results = read_quality_results(dir)

    volumes = sorted({volume for volume, _, _ in results})
    for volume in volumes:
        quality_metrics_table(results, volume, 0.5, 'global')
        quality_metrics_table(results, volume, 0.5, 'indirect')


def quality_metrics_table(results, volume, time, mode):
    metrics = (
        ('ssim', 'SSIM', 3, 'uparrow'),
        ('lpips', 'LPIPS', 3, 'downarrow'),
        ('psnr', 'PSNR', 1, 'uparrow'),
    )
    extinctions = sorted({extinction for _, extinction, _ in results})
    transfer_functions = sorted({tf for _, _, tf in results})

    records = {}
    for extinction in extinctions:
        for transfer_function in transfer_functions:
            row = {}
            for key, label, precision, _ in metrics:
                pt, ours = compute_metric(
                    results,
                    volume,
                    extinction,
                    transfer_function,
                    f'{key}_{mode}',
                    time,
                )
                for group, (mean, std) in (('Path tracing', pt), ('Ours', ours)):
                    digits = max(round(std * 10**precision), 1)
                    row[(group, label)] = f'${mean:.{precision}f} ({digits})$'
            records[(f'${extinction}$', f'${transfer_function}$')] = row

    frame = pd.DataFrame.from_dict(records, orient='index')
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=['Ext.', 'TF'])
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    frame = frame[
        [
            (group, label)
            for group in ('Ours', 'Path tracing')
            for _, label, _, _ in metrics
        ]
    ]

    labels = ' & '.join(f'{label} $\\{arrow}$' for _, label, _, arrow in metrics)
    header = (
        '\\multicolumn{2}{c|}{} & '
        '\\multicolumn{3}{c|}{Ours} & '
        '\\multicolumn{3}{c}{Path tracing} \\\\\n'
        f'Ext. & TF & {labels} & {labels} \\\\'
    )

    latex = (
        frame.style.hide(axis='columns')
        .hide(names=True)
        .to_latex(
            column_format='l@{\\hspace{6pt}}r|ccc|ccc',
            hrules=True,
            sparse_index=True,
            multirow_align='t',
            position='p',
            caption=(
                f'Quality metrics for the \\textsf{{{shorten_volume_name(volume)}}} '
                'dataset with varying extinction coefficients ($80$, $200$, $1000$) '
                f'and transfer functions (TF $1$--$3$), evaluated under '
                f'{mode} illumination after ${time}$ seconds. Values are means '
                'over $10$ runs, with standard errors in parentheses, rounded '
                'to the first significant digit.'
            ),
            label=f'tab:{volume}-{str(time).replace(".", "_")}-{mode}',
        )
    )
    latex = latex.replace('\\toprule\n', f'\\toprule\n{header}\n', 1)

    lines = []
    for line in latex.split('\n'):
        if line.startswith('\\multirow') and lines and '\\midrule' not in lines[-1]:
            lines.append('\\cline{1-8}')
        lines.append(line)
    latex = '\n'.join(lines)

    latex = postprocess_latex_table(latex, ['Ext.', 'TF', 'Ours', 'Path tracing'])
    save_table(latex, f'quality/{volume}_{str(time).replace(".", "-")}_{mode}.tex')


def filter_tables():
    dir = os.path.join('evaluation', 'results', 'filter')
    filter_performance_table(dir)
    filter_quality_table(dir)


def filter_performance_table(dir):
    results = {
        'path_tracing': read_performance_results(
            os.path.join(dir, 'performance', 'path_tracing')
        ),
        'neural_render': read_performance_results(
            os.path.join(dir, 'performance', 'neural_render')
        ),
    }
    timings = compute_performance_speedup(results)
    latex = frame_speedup_table(
        timings, r'\captionfilterspeedup{}', 'tab:filter-speedup', position='t'
    )
    save_table(latex, 'filter_speedup.tex')


def filter_quality_table(dir):
    results = read_quality_results(os.path.join(dir, 'quality'))
    metrics = (
        ('ssim', 'SSIM', 3, 'uparrow'),
        ('lpips', 'LPIPS', 3, 'downarrow'),
        ('psnr', 'PSNR', 1, 'uparrow'),
    )
    modes = (('global', 'Global'), ('indirect', 'Indirect'))
    volumes = sorted({volume for volume, _, _ in results})
    records = {}
    for volume in volumes:
        extinction, transfer_function = min(
            (extinction, transfer_function)
            for key_volume, extinction, transfer_function in results
            if key_volume == volume
        )
        for mode, mode_label in modes:
            row = {}
            for key, label, precision, _ in metrics:
                pt, ours = compute_metric(
                    results,
                    volume,
                    extinction,
                    transfer_function,
                    f'{key}_{mode}',
                    0.5,
                )
                for group, (mean, std) in (('Path tracing', pt), ('Ours', ours)):
                    digits = max(round(std * 10**precision), 1)
                    row[(group, label)] = f'${mean:.{precision}f} ({digits})$'
            records[(f'\\textsf{{{shorten_volume_name(volume)}}}', mode_label)] = row
    frame = pd.DataFrame.from_dict(records, orient='index')
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=['Volume', 'Illum.'])
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    frame = frame[
        [
            (group, label)
            for group in ('Ours', 'Path tracing')
            for _, label, _, _ in metrics
        ]
    ]
    labels = ' & '.join(f'{label} $\\{arrow}$' for _, label, _, arrow in metrics)
    header = (
        '\\multicolumn{2}{c|}{} & '
        '\\multicolumn{3}{c|}{Ours} & '
        '\\multicolumn{3}{c}{Path tracing} \\\\\n'
        f'Volume & Illum. & {labels} & {labels} \\\\'
    )
    latex = (
        frame.style.hide(axis='columns')
        .hide(names=True)
        .to_latex(
            column_format='l@{\\hspace{6pt}}l|ccc|ccc',
            hrules=True,
            sparse_index=True,
            multirow_align='t',
            caption=(
                r'Quality metrics for the \textsf{chameleon}, \textsf{heptane}, '
                r'and \textsf{neurons} datasets with filtering enabled, using '
                f'extinction coefficient ${extinction}$ and transfer function '
                f'${transfer_function}$, evaluated under global and indirect '
                f'illumination after $0.5$ seconds. Values are means over '
                '$10$ runs, with standard errors in parentheses, rounded to the '
                'first significant digit.'
            ),
            label='tab:filter-quality',
        )
    )
    latex = latex.replace('\\toprule\n', f'\\toprule\n{header}\n', 1)
    lines = []
    for line in latex.split('\n'):
        if line.startswith('\\multirow') and lines and '\\midrule' not in lines[-1]:
            lines.append('\\cline{1-8}')
        lines.append(line)
    latex = '\n'.join(lines)
    latex = postprocess_latex_table(latex, ['Volume', 'Illum.', 'Ours', 'Path tracing'])
    save_table(latex, 'filter_quality_0-5.tex')


def postprocess_latex_table(latex, bold_columns):
    """
    Move caption and label to the bottom of the table, add centering and
    footnotesize, bold column names, and indent lines.
    """
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

    latex = '\n'.join(lines)
    for col in bold_columns:
        latex = latex.replace(col, r'\textbf{' + col + '}', 1)

    return latex


def shorten_volume_name(name):
    return (
        name.replace('csafe_heptane', 'heptane')
        .replace('mri_ventricles', 'ventricles')
        .replace('marmoset_neurons', 'neurons')
    )


def save_table(latex, filename):
    file_path = os.path.join(TABLES_DIR, filename)
    with open(file_path, 'wt', encoding='utf-8') as f:
        f.write(latex)


if __name__ == '__main__':
    main()
