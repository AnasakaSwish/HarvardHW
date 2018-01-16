import warnings
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    # turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)


warnings.filterwarnings('ignore', message='Polyfit*')

def scatter_by(df, scatterx, scattery, by=None, figure=None, axes=None, labeler={}, mfunc=None,
               setupfunc=None, mms=8):
    cs = copy.deepcopy(colorscale.mpl_colors)
    if not figure:
        figure = plt.figure(figsize=(8, 8))
    if not axes:
        axes = figure.gca()
    x = df[scatterx]
    y = df[scattery]
    if not by:
        col = random.choice(cs)
        axes.scatter(x, y, cmap=colorscale, c=col)
        if setupfunc:
            axeslist = setupfunc(axes, figure)
        else:
            axeslist = [axes]
        if mfunc:
            mfunc(axeslist, x, y, color=col, mms=mms)
    else:
        cs = list(np.linspace(0, 1, len(df.groupby(by))))
        xlimsd = {}
        ylimsd = {}
        xs = {}
        ys = {}
        cold = {}
        for k, g in df.groupby(by):
            col = cs.pop()
            x = g[scatterx]
            y = g[scattery]
            xs[k] = x
            ys[k] = y
            c = colorscale.mpl_colormap(col)
            cold[k] = c
            axes.scatter(x, y, c=c, label=labeler.get(k, k), s=40, alpha=0.3);
            xlimsd[k] = axes.get_xlim()
            ylimsd[k] = axes.get_ylim()
        xlims = [min([xlimsd[k][0] for k in xlimsd.keys()]), max([xlimsd[k][1] for k in xlimsd.keys()])]
        ylims = [min([ylimsd[k][0] for k in ylimsd.keys()]), max([ylimsd[k][1] for k in ylimsd.keys()])]
        axes.set_xlim(xlims)
        axes.set_ylim(ylims)
        if setupfunc:
            axeslist = setupfunc(axes, figure)
        else:
            axeslist = [axes]
        if mfunc:
            for k in xs.keys():
                mfunc(axeslist, xs[k], ys[k], color=cold[k], mms=mms);
    axes.set_xlabel(scatterx);
    axes.set_ylabel(scattery);

    return axes

def make_rug(axeslist, x, y, color='b', mms=8):
    axes = axeslist[0]
    zerosx1 = np.zeros(len(x))
    zerosx2 = np.zeros(len(x))
    xlims = axes.get_xlim()
    ylims = axes.get_ylim()
    zerosx1.fill(ylims[1])
    zerosx2.fill(xlims[1])
    axes.plot(x, zerosx1, marker='|', color=color, ms=mms)
    axes.plot(zerosx2, y, marker='_', color=color, ms=mms)
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    return axes

np.random.seed(42)

def rmse(p,x,y):
    yfit = np.polyval(p, x)
    return np.sqrt(np.mean((y - yfit) ** 2))

def generate_curve(x, sigma):
    return np.random.normal(10 - 1. / (x + 0.1), sigma)
x = 10 ** np.linspace(-2, 0, 8)
intrinsic_error=1.
y=generate_curve(x, intrinsic_error)
plt.scatter(x,y)
plt.show()