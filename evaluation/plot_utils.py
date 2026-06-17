import matplotlib.pyplot as plt

colors = [
    '#FFF5B1',
    '#FFE3B0',
    '#FFD2B1',
    '#FFBAAF',
    '#A1E6D9',
    '#B2ECFF',
    '#B3CDFF',
    '#AEAFFF',
]


def set_plot_style():
    plt.rcParams['lines.linewidth'] = 0.8
    plt.rcParams['font.family'] = 'Serif'
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.labelsize'] = 8


def set_legend_style(legend):
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(0.8)
    frame.set_boxstyle('Square', pad=0)
    frame.set_facecolor('white')
    frame.set_alpha(1.0)
