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
import time
import seaborn
seaborn.set_style('ticks')


def plot_stations(layout, name, r_min, r_max, station_d,
                  results_dir='temp_layouts', b_max=[], min_sep=None):
    """Plot the station coords from the layout"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for x_, y_ in zip(layout['x'], layout['y']):
        c = plt.Circle((x_, y_), station_d / 2, fill=True, color='k',
                       alpha=0.6, lw=0.0)
        ax.add_artist(c)
        if min_sep and 'taper_func' in layout:
            r = (x_**2 + y_**2)**0.5
            c = plt.Circle((x_, y_), (min_sep / 2) * (1 / layout['taper_func'](r / r_max)),
                           fill=False, color='r', alpha=0.5, lw=0.8)
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

    if 'trials' in layout:
        xt = layout['trials'][:, 0]
        yt = layout['trials'][:, 1]
        ax.plot(xt, yt, 'b.', ms=3.0, alpha=0.5)

    ax.set_xlabel('east (m)')
    ax.set_ylabel('north (m)')
    b_max.append(r_max * 1.05)
    for i, lim in enumerate(b_max):
        print('', i, lim)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        save_fig(fig, 'layout_%s_%06.1fm.png' % (name.lower(), lim),
                 [results_dir])
    # plt.show()
    plt.close(fig)


def plot_layouts(layouts, station_d, results_dir, plot_r=[]):
    print('*' * 20)
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    ax.set_aspect('equal')
    r_max = 0.0
    for name in layouts:
        print(name, layouts[name].keys())
        x = layouts[name]['x']
        y = layouts[name]['y']
        r = (x**2 + y**2)**0.5
        r_max = max(r_max, r.max())
        for p in zip(x, y):
            ax.add_artist(plt.Circle(p, station_d / 2, fill=False, color='k'))
    ax.grid()
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    plt.savefig(join(results_dir, 'layouts_%05.0fm.png' % r_max))
    for r in plot_r:
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        plt.savefig(join(results_dir, 'layouts_%05.0fm.png' % r))
    plt.close(fig)


def main():
    from utilities.generators import (inner_arms,
                                      inner_arms_clusters,
                                      inner_arms_rand_uniform,
                                      uniform_core,
                                      tapered_core)
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

    v4a_ss_enu_file = join('models', 'v7ska1lowN1v2rev3R.enu.94x4.fixed.txt')
    v4a_ss_enu = np.loadtxt(v4a_ss_enu_file)
    r = (v4a_ss_enu[:, 1]**2 + v4a_ss_enu[:, 2]**2)**0.5
    v5_enu_file = join('models', 'v5.tm', 'layout.txt')
    v5_enu = np.loadtxt(v5_enu_file)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    for p in zip(v4a_ss_enu[:, 1], v4a_ss_enu[:, 2]):
        ax.add_artist(plt.Circle(p, 90 / 2, fill=False, color='k'))
    for p in zip(v5_enu[:, 0], v5_enu[:, 1]):
        ax.add_artist(plt.Circle(p, 35 / 2, fill=False, color='b'))
        ax.add_artist(plt.Circle(p, 35 / 2, fill=True, color='b', alpha=0.5,
                                 lw=0.0))
    ax.add_artist(plt.Circle((0, 0), 1700.0, fill=False, color='r'))
    ax.add_artist(plt.Circle((0, 0), 7000.0, fill=False, color='r'))
    ax.grid(True)
    ax.set_xlim(-r.max(), r.max())
    ax.set_ylim(-r.max(), r.max())
    plt.show()

    return
    #layouts['outer_arms'] =

    cluster_d = 100.0
    r_min, r_max = 550.0 + cluster_d / 2, 5000.0
    b, num_arms, clusters_per_arm, stations_per_cluster = 0.5, 3, 8, 3
    layouts['clusters'] = inner_arms_clusters(
        b, num_arms, clusters_per_arm, stations_per_cluster, cluster_d,
        station_d, r_min, r_max, trail_timeout=3.0, tries_per_cluster=3)

    # n = 300
    # r_max = 500.0
    # layouts['core'] = uniform_core(n, r_max, station_d)

    # def taper_func(r):
    #     if r < 0.5:
    #         return 1
    #     elif r < 0.75:
    #         return 0.75
    #     else:
    #         return 0.5

    def taper_func(r):
        return 1 - (r / 2)**1.1

    n, r_min, r_max = 250, 0.0, 550.0
    # TODO(BM) multi try or multi r_max version of this function
    # TODO(BM) taper functions... taylor, etc
    t0 = time.time()
    layouts['core_tapered'] = tapered_core(n, r_max, station_d, taper_func,
                                           trial_timeout=5.0, r_min=r_min)
    print('time taken = %.3f s' % (time.time() - t0))

    for i, name in enumerate(layouts):
        print(i, '-', name, layouts[name]['x'].shape[0], 'stations',
              layouts[name].keys())
        plot_stations(layouts[name], name, r_min, r_max, station_d,
                      results_dir, b_max=[1000, 2000, 5000],
                      min_sep=station_d)

    plot_layouts(layouts, station_d, results_dir, plot_r=[5000, 1000, 500])


if __name__ == '__main__':
    main()
