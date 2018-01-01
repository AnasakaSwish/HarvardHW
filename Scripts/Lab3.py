import brewer2mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image

#colorbrewer2 Dark2 qualitative color table
dark2_cmap = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
dark2_colors = dark2_cmap.mpl_colors

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

imagefp = "../hwandsolutions/labs/lab3/Italy.png"
olivefp = "../hwandsolutions/labs/lab3/data/olive.csv"

Image(filename=imagefp)
df = pd.read_csv(olivefp)
df.rename(columns={df.columns[0]: 'areastring'}, inplace=True)
df.areastring = df.areastring.map(lambda x: x.split('.')[-1])

acidlist = ['palmitic', 'palmitoleic', 'stearic', 'oleic', 'linoleic', 'linolenic', 'arachidic', 'eicosenoic']


def divide100(x):
    return x / 100.


dfsub = df[acidlist].apply(divide100)
df[acidlist] = dfsub

xacids = ['oleic', 'linolenic', 'eicosenoic']
yacids = ['stearic', 'arachidic']
#
# fig, axis = plt.subplots(3, 2)
#
# for i, xacid in enumerate(xacids):
#     for j, yacid in enumerate(yacids):
#         axis[i, j].scatter(df[xacid], df[yacid])
#         axis[i, j].set_xlabel(xacid)
#         axis[i, j].set_ylabel(yacid)
#
# plt.show()

# region_groupby = df.groupby('region')
# dfbystd = df.groupby('region').std()
# dfbymean = region_groupby.aggregate(np.mean)
# renamedict_std = {k: k + "_std" for k in acidlist}
# renamedict_mean = {k: k + "_mean" for k in acidlist}
# dfbystd.rename(inplace=True, columns=renamedict_std)
# dfbymean.rename(inplace=True, columns=renamedict_mean)
# dfpalmiticmean = dfbymean[['palmitic_mean']]
# dfpalmiticstd = dfbystd[['palmitic_std']]
#
# newdfbyregion = dfpalmiticmean.join(dfpalmiticstd)
# weights = np.random.uniform(size=df.shape[0])
# smallerdf = df[['palmitic']].copy()
# otherdf = df[['region']].copy()
# otherdf = otherdf.assign(weights=pd.Series(weights))
#
# smallerdf=smallerdf.join(otherdf)
# wavg=((smallerdf.palmitic*smallerdf.weights).sum()/ smallerdf.weights.sum())
# print(wavg)
# print(region_groupby.agg(np.sum))

rkeys=[1,2,3]
rvals=['South','Sardinia','North']
rmap={e[0]:e[1] for e in zip(rkeys,rvals)}
mdf2=df.groupby('region').aggregate(np.mean)
mdf2=mdf2[acidlist]
acidlist=['palmitic', 'palmitoleic', 'stearic', 'linoleic', 'linolenic', 'arachidic', 'eicosenoic']
# fig, axes=plt.subplots(figsize=(10,20), nrows=len(acidlist), ncols=1)
# i=0
# colors=[dark2_cmap.mpl_colormap(col) for col in [1.0,0.5,0.0]]

# def make2d(df, scatterx, scattery, by="region", labeler={}):
#     figure=plt.figure(figsize=(8,8))
#     ax=plt.gca()
#     cs=list(np.linspace(0,1,len(df.groupby(by))))
#     xlimsd={}
#     ylimsd={}
#     xs={}
#     ys={}
#     cold={}
#     for k,g in df.groupby(by):
#         col=cs.pop()
#         x=g[scatterx]
#         y=g[scattery]
#         xs[k]=x
#         ys[k]=y
#         c=dark2_cmap.mpl_colormap(col)
#         cold[k]=c
#         ax.scatter(x, y, c=c, label=labeler.get(k,k), s=40, alpha=0.4);
#         xlimsd[k]=ax.get_xlim()
#         ylimsd[k]=ax.get_ylim()
#     xlims=[min([xlimsd[k][0] for k in xlimsd.keys()]), max([xlimsd[k][1] for k in xlimsd.keys()])]
#     ylims=[min([ylimsd[k][0] for k in ylimsd.keys()]), max([ylimsd[k][1] for k in ylimsd.keys()])]
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
#     ax.set_xlabel(scatterx)
#     ax.set_ylabel(scattery)
#     ax.grid(False)
#     return ax
# a=make2d(df, "linoleic","arachidic", labeler=rmap)
# a.legend(loc='upper right')
# plt.show()

dfcopy=df.copy()
dfcopy['region']=dfcopy['region'].map(rmap)
imap={e[0]:e[1] for e in zip (df.area.unique(), df.areastring.unique())}
dfcopy['area']=dfcopy['area'].map(imap)
# plot = rplot.RPlot(dfcopy, x='linoleic', y='oleic')
# plot.add(rplot.TrellisGrid(['region', '.']))
# plot.add(rplot.GeomPoint(size=40.0, alpha=0.3, colour=rplot.ScaleRandomColour('area')));

import seaborn as sns

plot=sns.FacetGrid(dfcopy, col="region",hue="area",palette=None)
plot=(plot.map(plt.scatter,"linoleic","oleic",alpha=0.3)).add_legend()
plt.show()
