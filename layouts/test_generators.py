# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import os
import sys
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from shutil import rmtree
from matplotlib.ticker import FormatStrFormatter
from math import radians, degrees, pi, ceil, sin, asin, log, log10, floor
from utilities.plotting import save_fig
import seaborn
seaborn.set_style('ticks')


def plot_stations(layout, name, r_min, r_max, station_d,
                  results_dir='temp_layouts', b_max=[]):
    """Plot the station coords from the layout"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for x_, y_ in zip(layout['x'], layout['y']):
        c = plt.Circle((x_, y_), station_d / 2, fill=True, color='k',
                       alpha=0.6, lw=0.0)
        ax.add_artist(c)
    c = plt.Circle((0.0, 0.0), r_max, fill=False, color='r', linestyle='-',
                   linewidth=1.0, alpha=0.3)
    ax.add_artist(c)
    c = plt.Circle((0.0, 0.0), r_min, fill=False, color='r', linestyle='-',
                   linewidth=1.0, alpha=0.3)
    ax.add_artist(c)
    if 'cx' in layout and 'cy' in layout and 'cr' in layout:
        for x_, y_ in zip(layout['cx'], layout['cy']):
            c = plt.Circle((x_, y_), layout['cr'], fill=True, color='b',
                           alpha=0.1, lw=0.5)
            ax.add_artist(c)
    ax.set_xlabel('east (m)')
    ax.set_ylabel('north (m)')
    b_max.append(r_max * 1.05)
    for i, lim in enumerate(b_max):
        print('', i, lim)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        save_fig(fig, 'layout_%s_%06.1fm.png' % (name.lower(), lim),
                 [results_dir])
    plt.show()
    plt.close(fig)


def main():
    from utilities.generators import (inner_arms,
                                      inner_arms_clusters,
                                      inner_arms_rand_uniform)
    from utilities.layout import rand_uniform_2d_trials

    results_dir = 'temp_layouts'
    station_d = 35.0
    layouts = dict()

    r_min, r_max = 0.0, 5000.0

    # r_min, r_max = 500.0, 5000.0
    # b, num_arms, n = 0.5, 6, 12
    # layouts['spiral'] = inner_arms(b, num_arms, n, r_min, r_max)

    # n, r_min, r_max = 72, 500.0, 5000.0
    # layouts['rand_uniform'] = inner_arms_rand_uniform(
    #     n, station_d, r_min, r_max)

    # r_min, r_max = 500.0, 5000.0
    # b, num_arms, clusters_per_arm, stations_per_cluster = 0.5, 3, 4, 6
    # cluster_d = 160.0
    # layouts['clusters'] = inner_arms_clusters(
    #     b, num_arms, clusters_per_arm, stations_per_cluster, cluster_d,
    #     station_d, r_min, r_max, trail_timeout=3.0, tries_per_cluster=3)

    n = 300
    r_max = 500.0
    x, y, _ = rand_uniform_2d_trials(n, r_max, min_sep=station_d,
                                     trial_timeout=2.0, num_trials=5,
                                     seed=None, r_min=0.0, verbose=False)
    layouts['core'] = {'x': x, 'y': y}

    for i, name in enumerate(layouts):
        print(i, '-', name, layouts[name]['x'].shape[0], 'stations')
        plot_stations(layouts[name], name, r_min, r_max, station_d,
                      results_dir, b_max=[2000, 1000])


if __name__ == '__main__':
    main()
