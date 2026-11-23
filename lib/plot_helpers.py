"""Figures: segment colours, figure style, date axis."""
import matplotlib.dates as mdates

SEG_COLORS = {'before_a': 'tab:blue', 'before_c': 'tab:olive', 'during': 'tab:orange', 'after': 'tab:purple',
              'wide': 'tab:cyan', 'normal_a': 'tab:green', 'normal_b': 'tab:brown', 'narrow': 'tab:pink'}
# segment name → colour (same as the SBND RWM calibration figures)

FULL = 6.3   # figure width [in]
STYLE = {'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7, 'legend.fontsize': 6, 'xtick.labelsize': 7,
         'ytick.labelsize': 7, 'legend.title_fontsize': 6}


def style_time_axis(ax, xlim=None, xlabel='Date-Month-Year', fmt='%d-%b-%y', minor_grid_alpha=0.4):
    """Date formatter, minor ticks, major + dotted minor grid."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=minor_grid_alpha, linestyle=':', linewidth=0.7)
