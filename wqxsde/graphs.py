
"""
Code from
@article {GWAT:GWAT12118,
author = {Peeters, Luk},
title = {A Background use_color Scheme for Piper Plots to Spatially Visualize Hydrochemical Patterns},
journal = {Groundwater},
volume = {52},
number = {1},
publisher = {Blackwell Publishing Ltd},
issn = {1745-6584},
url = {http://dx.doi.org/10.1111/gwat.12118},
doi = {10.1111/gwat.12118},
pages = {2--6},
year = {2014},
}

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from collections import OrderedDict
from datetime import datetime, timedelta
# Load required packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as interpolate


# Define plotting functions hsvtorgb and piper
def hsvtorgb(H, S, V):
    """Converts hsv use_colorspace to rgb

    :param H: [mxn] matrix of hue ( between 0 and 2pi )
    :param S: [mxn] matrix of saturation ( between 0 and 1 )
    :param V: [mxn] matrix of value ( between 0 and 1 )
    :return:
        R: [mxn] matrix of red ( between 0 and 1 )
        G: [mxn] matrix of green ( between 0 and 1 )
        B: [mxn] matrix of blue ( between 0 and 1 )
    """

    # conversion (from http://en.wikipedia.org/wiki/HSL_and_HSV)
    C = V * S
    Hs = H / (np.pi / 3)
    X = C * (1 - np.abs(np.mod(Hs, 2.0 * np.ones_like(Hs)) - 1))
    N = np.zeros_like(H)
    # create empty RGB matrices
    R = np.zeros_like(H)
    B = np.zeros_like(H)
    G = np.zeros_like(H)
    # assign values
    h = np.floor(Hs)
    # h=0
    R[h == 0] = C[h == 0]
    G[h == 0] = X[h == 0]
    B[h == 0] = N[h == 0]
    # h=1
    R[h == 1] = X[h == 1]
    G[h == 1] = C[h == 1]
    B[h == 1] = N[h == 1]
    # h=2
    R[h == 2] = N[h == 2]
    G[h == 2] = C[h == 2]
    B[h == 2] = X[h == 2]
    # h=3
    R[h == 3] = N[h == 3]
    G[h == 3] = X[h == 3]
    B[h == 3] = C[h == 3]
    # h=4
    R[h == 4] = X[h == 4]
    G[h == 4] = N[h == 4]
    B[h == 4] = C[h == 4]
    # h=5
    R[h == 5] = C[h == 5]
    G[h == 5] = N[h == 5]
    B[h == 5] = X[h == 5]
    # match values
    m = V - C
    R = R + m
    G = G + m
    B = B + m
    return (R, G, B)


def rgb2hex(r, g, b):
    hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return hex


def piper(arrays, plottitle, use_color, fig=None, ax=None, *args, **kwargs):
    """Create a Piper plot:

    Args:
        arrays (ndarray, or see below): n x 8 ndarray with columns corresponding
            to Ca Mg Na K HCO3 CO3 Cl SO4 data. See below for a different format
            for this argument if you want to plot different subsets of data with
            different marker styles.
        plottitle (str): title of Piper plot
        use_color (bool): use background use_coloring of Piper plot
        fig (Figure): matplotlib Figure to use, one will be created if None

    If you would like to plot different sets of data e.g. from different aquifers
    with different marker styles, you can pass a list of tuples as the first
    arguments. The first item of each tuple should be an n x 8 ndarray, as usual.
    The second item should be a dictionary of keyword arguments to ``plt.scatter``.
    Any keyword arguments for ``plt.scatter`` that are in common to all the
    subsets can be passed to the ``piper()`` function directly. By default the
    markers are plotted with: ``plt.scatter(..., marker=".", color="k", alpha=1)``.

    Returns a dictionary with:
            if use_color = False:
                cat: [nx3] meq% of cations, order: Ca Mg Na+K
                an:  [nx3] meq% of anions,  order: HCO3+CO3 SO4 Cl
            if use_color = True:
                cat: [nx3] RGB triple cations
                an:  [nx3] RGB triple anions
                diamond: [nx3] RGB triple central diamond
    """
    kwargs["marker"] = kwargs.get("marker", ".")
    kwargs["alpha"] = kwargs.get("alpha", 1)
    kwargs["facecolor"] = kwargs.get("facecolor", "k")

    try:
        shp = arrays.shape
        if shp[1] == 8:
            arrays = [(arrays, {})]
    except:
        pass

    if fig is None:
        fig = plt.figure()
    # Basic shape of piper plot
    offset = 0.05
    offsety = offset * np.tan(np.pi / 3)
    h = 0.5 * np.tan(np.pi / 3)
    ltriangle_x = np.array([0, 0.5, 1, 0])
    ltriangle_y = np.array([0, h, 0, 0])
    rtriangle_x = ltriangle_x + 2 * offset + 1
    rtriangle_y = ltriangle_y
    diamond_x = np.array([0.5, 1, 1.5, 1, 0.5]) + offset
    diamond_y = h * (np.array([1, 2, 1, 0, 1])) + (offset * np.tan(np.pi / 3))

    if ax:
        pass
    else:
        ax = fig.add_subplot(111, aspect='equal', frameon=False, xticks=[], yticks=[])
    ax.plot(ltriangle_x, ltriangle_y, '-k')
    ax.plot(rtriangle_x, rtriangle_y, '-k')
    ax.plot(diamond_x, diamond_y, '-k')
    # labels and title
    ax.set_title(plottitle)
    ax.text(-offset, -offset, u'$Ca^{2+}$', horizontalalignment='left', verticalalignment='center')
    ax.text(0.5, h + offset, u'$Mg^{2+}$', horizontalalignment='center', verticalalignment='center')
    ax.text(1 + offset, -offset, u'$Na^+ + K^+$', horizontalalignment='right', verticalalignment='center')
    ax.text(1 + offset, -offset, u'$HCO_3^- + CO_3^{2-}$', horizontalalignment='left', verticalalignment='center')
    ax.text(1.5 + 2 * offset, h + offset, u'$SO_4^{2-}$', horizontalalignment='center', verticalalignment='center')
    ax.text(2 + 3 * offset, -offset, u'$Cl^-$', horizontalalignment='right', verticalalignment='center')

    # Convert chemistry into plot coordinates
    gmol = np.array([40.078, 24.305, 22.989768, 39.0983, 61.01714, 60.0092, 35.4527, 96.0636])
    eqmol = np.array([2., 2., 1., 1., 1., 2., 1., 2.])
    for dat_piper, plt_kws in arrays:
        n = dat_piper.shape[0]
        meqL = (dat_piper / gmol) * eqmol
        sumcat = np.sum(meqL[:, 0:4], axis=1)
        suman = np.sum(meqL[:, 4:8], axis=1)
        cat = np.zeros((n, 3))
        an = np.zeros((n, 3))
        cat[:, 0] = meqL[:, 0] / sumcat  # Ca
        cat[:, 1] = meqL[:, 1] / sumcat  # Mg
        cat[:, 2] = (meqL[:, 2] + meqL[:, 3]) / sumcat  # Na+K
        an[:, 0] = (meqL[:, 4] + meqL[:, 5]) / suman  # HCO3 + CO3
        an[:, 2] = meqL[:, 6] / suman  # Cl
        an[:, 1] = meqL[:, 7] / suman  # SO4

        # Convert into cartesian coordinates
        cat_x = 0.5 * (2 * cat[:, 2] + cat[:, 1])
        cat_y = h * cat[:, 1]
        an_x = 1 + 2 * offset + 0.5 * (2 * an[:, 2] + an[:, 1])
        an_y = h * an[:, 1]
        d_x = an_y / (4 * h) + 0.5 * an_x - cat_y / (4 * h) + 0.5 * cat_x
        d_y = 0.5 * an_y + h * an_x + 0.5 * cat_y - h * cat_x

        # plot data
        kws = dict(kwargs)
        kws.update(plt_kws)
        ax.scatter(cat_x, cat_y, **kws)
        ax.scatter(an_x, an_y, **{k: v for k, v in kws.items() if not k == "label"})
        ax.scatter(d_x, d_y, **{k: v for k, v in kws.items() if not k == "label"})

    # use_color coding Piper plot
    if use_color == False:
        # add density use_color bar if alphalevel < 1
        if kwargs.get("alpha", 1) < 1.0:
            ax1 = fig.add_axes([0.75, 0.4, 0.01, 0.2])
            cmap = plt.cm.gray_r
            norm = mpl.use_colors.Normalize(vmin=0, vmax=1 / kwargs["alpha"])
            cb1 = mpl.use_colorbar.use_colorbarBase(ax1, cmap=cmap,
                                                    norm=norm,
                                                    orientation='vertical')
            cb1.set_label('Dot Density')

        return (dict(cat=cat, an=an))
    else:

        # create empty grids to interpolate to
        x0 = 0.5
        y0 = x0 * np.tan(np.pi / 6)
        X = np.reshape(np.repeat(np.linspace(0, 2 + 2 * offset, 1000), 1000), (1000, 1000), 'F')
        Y = np.reshape(np.repeat(np.linspace(0, 2 * h + offsety, 1000), 1000), (1000, 1000), 'C')
        H = np.nan * np.zeros_like(X)
        S = np.nan * np.zeros_like(X)
        V = np.nan * np.ones_like(X)
        A = np.nan * np.ones_like(X)
        # create masks for cation, anion triangle and upper and lower diamond
        ind_cat = np.logical_or(np.logical_and(X < 0.5, Y < 2 * h * X),
                                np.logical_and(X > 0.5, Y < (2 * h * (1 - X))))
        ind_an = np.logical_or(np.logical_and(X < 1.5 + (2 * offset), Y < 2 * h * (X - 1 - 2 * offset)),
                               np.logical_and(X > 1.5 + (2 * offset), Y < (2 * h * (1 - (X - 1 - 2 * offset)))))
        ind_ld = np.logical_and(
            np.logical_or(np.logical_and(X < 1.0 + offset, Y > -2 * h * X + 2 * h * (1 + 2 * offset)),
                          np.logical_and(X > 1.0 + offset, Y > 2 * h * X - 2 * h)),
            Y < h + offsety)
        ind_ud = np.logical_and(np.logical_or(np.logical_and(X < 1.0 + offset, Y < 2 * h * X),
                                              np.logical_and(X > 1.0 + offset, Y < -2 * h * X + 4 * h * (1 + offset))),
                                Y > h + offsety)
        ind_d = np.logical_or(ind_ld == 1, ind_ud == 1)

        # Hue: convert x,y to polar coordinates
        # (angle between 0,0 to x0,y0 and x,y to x0,y0)
        H[ind_cat] = np.pi + np.arctan2(Y[ind_cat] - y0, X[ind_cat] - x0)
        H[ind_cat] = np.mod(H[ind_cat] - np.pi / 6, 2 * np.pi)
        H[ind_an] = np.pi + np.arctan2(Y[ind_an] - y0, X[ind_an] - (x0 + 1 + (2 * offset)))
        H[ind_an] = np.mod(H[ind_an] - np.pi / 6, 2 * np.pi)
        H[ind_d] = np.pi + np.arctan2(Y[ind_d] - (h + offsety), X[ind_d] - (1 + offset))
        # Saturation: 1 at edge of triangle, 0 in the centre,
        # Clough Tocher interpolation, square root to reduce central white region
        xy_cat = np.array([[0.0, 0.0],
                           [x0, h],
                           [1.0, 0.0],
                           [x0, y0]])
        xy_an = np.array([[1 + (2 * offset), 0.0],
                          [x0 + 1 + (2 * offset), h],
                          [2 + (2 * offset), 0.0],
                          [x0 + 1 + (2 * offset), y0]])
        xy_d = np.array([[x0 + offset, h + offsety],
                         [1 + offset, 2 * h + offsety],
                         [x0 + 1 + offset, h + offsety],
                         [1 + offset, offsety],
                         [1 + offset, h + offsety]])
        z_cat = np.array([1.0, 1.0, 1.0, 0.0])
        z_an = np.array([1.0, 1.0, 1.0, 0.0])
        z_d = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        s_cat = interpolate.CloughTocher2DInterpolator(xy_cat, z_cat)
        s_an = interpolate.CloughTocher2DInterpolator(xy_an, z_an)
        s_d = interpolate.CloughTocher2DInterpolator(xy_d, z_d)
        S[ind_cat] = s_cat.__call__(X[ind_cat], Y[ind_cat])
        S[ind_an] = s_an.__call__(X[ind_an], Y[ind_an])
        S[ind_d] = s_d.__call__(X[ind_d], Y[ind_d])
        # Value: 1 everywhere
        V[ind_cat] = 1.0
        V[ind_an] = 1.0
        V[ind_d] = 1.0
        # Alpha: 1 everywhere
        A[ind_cat] = 1.0
        A[ind_an] = 1.0
        A[ind_d] = 1.0
        # convert HSV to RGB
        R, G, B = hsvtorgb(H, S ** 0.5, V)
        RGBA = np.dstack((R, G, B, A))
        # visualise
        ax.imshow(RGBA,
                   origin='lower',
                   aspect=1.0,
                   extent=(0, 2 + 2 * offset, 0, 2 * h + offsety))
        # calculate RGB triples for data points
        # hue
        hcat = np.pi + np.arctan2(cat_y - y0, cat_x - x0)
        hcat = np.mod(hcat - np.pi / 6, 2 * np.pi)
        han = np.pi + np.arctan2(an_y - y0, an_x - (x0 + 1 + (2 * offset)))
        han = np.mod(han - np.pi / 6, 2 * np.pi)
        hd = np.pi + np.arctan2(d_y - (h + offsety), d_x - (1 + offset))
        # saturation
        scat = s_cat.__call__(cat_x, cat_y) ** 0.5
        san = s_an.__call__(an_x, an_y) ** 0.5
        sd = s_d.__call__(d_x, d_y) ** 0.5
        # value
        v = np.ones_like(hd)
        # rgb
        cat = np.vstack((hsvtorgb(hcat, scat, v))).T
        an = np.vstack((hsvtorgb(han, san, v))).T
        d = np.vstack((hsvtorgb(hd, sd, v))).T
        return (dict(cat=cat,
                     an=an,
                     diamond=d))


if __name__ == "__main__":
    ###
    # Plot example data
    ###

    # Load data
    dat = np.loadtxt('CondamineData.csv',
                     delimiter=',',
                     skiprows=1)
    # define extent map
    bbox = np.array([np.min(dat[:, 1]),
                     np.min(dat[:, 0]),
                     np.max(dat[:, 1]),
                     np.max(dat[:, 0])])
    bbox = bbox + np.array([-0.05 * (bbox[2] - bbox[0]),
                            -0.05 * (bbox[3] - bbox[1]),
                            0.05 * (bbox[2] - bbox[0]),
                            0.05 * (bbox[3] - bbox[1])])
    # Plot example data
    # Piper plot
    rgb = piper(dat[:, 2:10], 'Condamine Alluvium', use_color=True)
    # Maps (only data points, no background shape files)
    fig = plt.figure()
    # cations
    ax1 = fig.add_subplot(131,
                          aspect='equal',
                          axisbg=[0.75, 0.75, 0.75, 1])
    ax1.scatter(dat[:, 1],
                dat[:, 0],
                s=10,
                c=rgb['cat'],
                edgeuse_color='None',
                zorder=3)
    plt.title('Condamine Cations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.xlim(bbox[0], bbox[2])
    plt.ylim(bbox[1], bbox[3])
    # central diamond
    ax2 = fig.add_subplot(132,
                          aspect='equal',
                          axisbg=[0.75, 0.75, 0.75, 1])
    ax2.scatter(dat[:, 1],
                dat[:, 0],
                s=10,
                c=rgb['diamond'],
                edgeuse_color='None',
                zorder=3)
    plt.title('Condamine Central Diamond')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(bbox[0], bbox[2])
    plt.ylim(bbox[1], bbox[3])
    plt.grid()
    # anions
    ax3 = fig.add_subplot(133,
                          aspect='equal',
                          axisbg=[0.75, 0.75, 0.75, 1])
    ax3.scatter(dat[:, 1],
                dat[:, 0],
                s=10,
                c=rgb['an'],
                edgeuse_color='None',
                zorder=3)
    plt.title('Condamine Anions')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(bbox[0], bbox[2])
    plt.ylim(bbox[1], bbox[3])
    plt.grid()


class RectPiper(object):
    """Class that generates rectangular piper diagrams.
    :param df: DataFrame containing chemistry data; must have fields labeled as abbreviations of the major ions; Na, K,
    NaK, Ca, Mg, Cl, HCO3, CO3, and SO4
    :type df: pandas.core.frame.DataFrame
    :param type_col: Name of field that designates the sample type (optional); defaults to ''
    :type type_col: str
    :param var_col: Name of field that contains a scalar variable to be designated by color (optional); defaults to ''
    :type var_col: str
    .. note::
    Hydrochemistry - Construct Rectangular Piper plot
    Adopted from: Ray and Mukherjee, 2008, Groundwater 46(6): 893-896 and from code found at:
    http://python.hydrology-amsterdam.nl/scripts/piper_rectangular.py
    Based on code by:
    B.M. van Breukelen <b.m.vanbreukelen@vu.nl>

    """

    def __init__(self, df, type_col='', var_col=''):

        self.fieldnames = [u'Na', u'K', u'NaK', u'Ca', u'Mg', u'Cl', u'HCO3', u'CO3', u'SO4']
        self.anions = ['Cl', 'HCO3', 'CO3', 'SO4']
        self.cations = ['Na', 'K', 'Ca', 'Mg', 'NaK']
        self.piperplot(df, type_col, var_col)

    def fillMissing(self, df):

        # fill in nulls with 0
        for col in df.columns:
            if col in self.fieldnames:
                for i in df.index:
                    if df.loc[i, col] is None or df.loc[i, col] == '' or np.isnan(df.loc[i, col]):
                        df.loc[i, col] = 0
            else:
                df.col = 0

        # add missing columns
        for name in self.fieldnames:
            if name in df.columns:
                pass
            else:
                print(name)
                df[name] = 0

        return df

    def check_nak(self, x):
        if x[0] == 0 and x[2] > 0:
            return x[2]
        else:
            return x[0] + x[1]

    def convertIons(self, df):
        """Convert major ion concentrations from mg/L to meq
        This function uses conversion factors to convert the concentrations of major ions from mg/L to meq.  It also
        appends a field to the input database listing the Cation-Anion pair that have the highest meq concentrations.
        :param df: DataFrame containing chemistry data; must have fields labeled as abbreviations of the major ions; Na, K,
        NaK, Ca, Mg, Cl, HCO3, CO3, and SO4
        :returns: appends convert fields onto DataFrame with the suffix `_meq` and adds the fields 'water type', 'CBE'
        (charge balance), 'EC' (Sum(anions+cations))
        """
        # Conversion factors from mg/L to meq/L
        d = {'Ca': 0.04990269, 'Mg': 0.082287595, 'Na': 0.043497608, 'K': 0.02557656, 'Cl': 0.028206596,
             'NaK': 0.043497608,
             'HCO3': 0.016388838, 'CO3': 0.033328223, 'SO4': 0.020833333, 'NO2': 0.021736513, 'NO3': 0.016129032}

        df1 = df

        for name in self.fieldnames:
            if name in df.columns:
                df1[name + '_meq'] = df1[name].apply(lambda x: float(d.get(name, 0)) * x, 1)

        df1['NaK_meq'] = df1[['Na_meq', 'K_meq', 'NaK_meq']].apply(lambda x: self.check_nak(x), 1)

        df1['anions'] = 0
        df1['cations'] = 0

        for ion in self.anions:
            if ion in df.columns:
                df1['anions'] += df1[ion + '_meq']
        for ion in self.cations:
            if ion in df1.columns:
                df1['cations'] += df1[ion + '_meq']

        df1['total_ions'] = df1['cations'] + df1['anions']
        df1['EC'] = df1['anions'] - df1['cations']
        df1['CBE'] = df1['EC'] / (df1['anions'] + df1['cations'])
        df1['maj_cation'] = df1[['Ca_meq', 'Mg_meq', 'Na_meq', 'K_meq', 'NaK_meq']].idxmax(axis=1)
        df1['maj_anion'] = df1[['Cl_meq', 'SO4_meq', 'HCO3_meq', 'CO3_meq']].idxmax(axis=1)
        df1['water_type'] = df1[['maj_cation', 'maj_anion']].apply(lambda x: str(x[0])[:-4] + '-' + str(x[1])[:-4], 1)
        return df1

    def ionPercentage(self, df):
        """Determines percentage of charge for each ion for display on the piper plot"""
        for ion in self.anions:
            df[ion + 'EC'] = df[[ion + '_meq', 'anions']].apply(lambda x: 100 * x[0] / x[1], 1)
        for ion in self.cations:
            df[ion + 'EC'] = df[[ion + '_meq', 'cations']].apply(lambda x: 100 * x[0] / x[1], 1)

        return df

    def piperplot(self, df, type_col, var_col):
        """Generates a rectangular piper diagram"""
        self.fillMissing(df)
        self.convertIons(df)
        self.ionPercentage(df)

        CaEC = df['CaEC'].values
        MgEC = df['MgEC'].values
        ClEC = df['ClEC'].values
        SO4EC = df['SO4EC'].values
        NaKEC = df['NaKEC'].values
        SO4ClEC = df[['ClEC', 'SO4EC']].apply(lambda x: x[0] + x[1], 1).values

        num_samps = len(df)
        if var_col == '':
            Elev = ''
        else:
            Elev = df[var_col].values

        if type_col == '':
            typ = ['Station'] * num_samps
            stationtypes = ['Station']
        else:
            stationtypes = list(df[type_col].unique())
            typ = df[type_col].values

        # Change default settings for figures
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('font', size=12)
        plt.rc('legend', fontsize=12)
        plt.rc('figure', figsize=(14, 5.5))  # defines size of Figure window orig (14,4.5)

        markSize = 30
        lineW = 0.5

        # Make Figure
        fig = plt.figure()
        # add title
        # fig.suptitle(piperTitle, x=0.20,y=.98, fontsize=14 )
        # Colormap and Saving Options for Figure

        if len(Elev) > 0:
            vart = Elev
        else:
            vart = [1] * num_samps
        cNorm = plt.Normalize(vmin=min(vart), vmax=max(vart))
        cmap = plt.cm.coolwarm
        # pdf = PdfPages(fileplace)

        mrkrSymbl = ['v', '^', '+', 's', '.', 'o', '*', 'v', '^', '+', 's', ',', '.', 'o', '*', 'v', '^', '+', 's', ',',
                     '.', 'o', '*', 'v', '^', '+', 's', ',', '.', 'o', '*']

        # count variable for legend (n)
        unique, counts = np.unique(typ, return_counts=True)
        nstatTypesDict = dict(zip(unique, counts))

        typdict = {}
        for i in range(len(stationtypes)):
            typdict[stationtypes[i]] = mrkrSymbl[i]

        # CATIONS-----------------------------------------------------------------------------
        # 2 lines below needed to create 2nd y-axis (ax1b) for first subplot
        ax1 = fig.add_subplot(131)
        ax1b = ax1.twinx()

        ax1.fill([100, 0, 100, 100], [0, 100, 100, 0], color=(0.8, 0.8, 0.8))
        ax1.plot([100, 0], [0, 100], 'k')
        ax1.plot([50, 0, 50, 50], [0, 50, 50, 0], 'k--')
        ax1.text(25, 15, 'Na type')
        ax1.text(75, 15, 'Ca type')
        ax1.text(25, 65, 'Mg type')

        if len(typ) > 0:
            for j in range(len(typ)):
                ax1.scatter(CaEC[j], MgEC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm, marker=typdict[typ[j]],
                            linewidths=lineW)
        else:
            ax1.scatter(CaEC, MgEC, s=markSize, c=vart, cmap=cmap, norm=cNorm, linewidths=lineW)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1b.set_ylim(0, 100)
        ax1.set_xlabel('<= Ca (% meq)')
        ax1b.set_ylabel('Mg (% meq) =>')
        plt.setp(ax1, yticklabels=[])

        # next line needed to reverse x axis:
        ax1.set_xlim(ax1.get_xlim()[::-1])

        # ANIONS----------------------------------------------------------------------------
        ax = fig.add_subplot(1, 3, 3)
        ax.fill([100, 100, 0, 100], [0, 100, 100, 0], color=(0.8, 0.8, 0.8))
        ax.plot([0, 100], [100, 0], 'k')
        ax.plot([50, 50, 0, 50], [0, 50, 50, 0], 'k--')
        ax.text(55, 15, 'Cl type')
        ax.text(5, 15, 'HCO3 type')
        ax.text(5, 65, 'SO4 type')

        if len(typ) > 0:
            for j in range(len(typ)):
                labs = "{:} n= {:}".format(typ[j], nstatTypesDict[typ[j]])
                if float(nstatTypesDict[typ[j]]) > 1:
                    s = ax.scatter(ClEC[j], SO4EC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm,
                                   marker=typdict[typ[j]], label=labs, linewidths=lineW)
                else:
                    s = ax.scatter(ClEC[j], SO4EC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm,
                                   marker=typdict[typ[j]], label=typ[j], linewidths=lineW)
        else:
            s = ax.scatter(ClEC, SO4EC, s=markSize, c=vart, cmap=cmap, norm=cNorm, label='Sample', linewidths=lineW)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Cl (% meq) =>')
        ax.set_ylabel('SO4 (% meq) =>')

        # CATIONS AND ANIONS COMBINED ---------------------------------------------------------------
        # 2 lines below needed to create 2nd y-axis (ax1b) for first subplot
        ax2 = fig.add_subplot(132)
        ax2b = ax2.twinx()

        ax2.plot([0, 100], [10, 10], 'k--')
        ax2.plot([0, 100], [50, 50], 'k--')
        ax2.plot([0, 100], [90, 90], 'k--')
        ax2.plot([10, 10], [0, 100], 'k--')
        ax2.plot([50, 50], [0, 100], 'k--')
        ax2.plot([90, 90], [0, 100], 'k--')

        if len(typ) > 0:
            for j in range(len(typ)):
                ax2.scatter(NaKEC[j], SO4ClEC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm, marker=typdict[typ[j]],
                            linewidths=lineW)
        else:
            ax2.scatter(NaKEC, SO4ClEC, s=markSize, c=vart, cmap=cmap, norm=cNorm, linewidths=lineW)

        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel('Na+K (% meq) =>')
        ax2.set_ylabel('SO4+Cl (% meq) =>')
        ax2.set_title('<= Ca+Mg (% meq)', fontsize=12)
        ax2b.set_ylabel('<= CO3+HCO3 (% meq)')
        ax2b.set_ylim(0, 100)

        # next two lines needed to reverse 2nd y axis:
        ax2b.set_ylim(ax2b.get_ylim()[::-1])

        # Align plots
        plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.90, wspace=0.4, hspace=0.0)

        # Legend-----------------------------------------------------------------------------------------

        # Add colorbar below legend
        # [left, bottom, width, height] where all quantities are in fractions of figure width and height

        if len(typ) > 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))

            plt.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, shadow=False, fancybox=True,
                       bbox_to_anchor=(0.5, -0.3), scatterpoints=1)

        if len(Elev) > 0:
            cax = fig.add_axes([0.25, 0.10, 0.50, 0.02])
            cb1 = plt.colorbar(s, cax=cax, cmap=cmap, norm=cNorm, orientation='horizontal')  # use_gridspec=True
            cb1.set_label(var_col, size=8)

        self.plot = fig
        self.df = df