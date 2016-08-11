# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import time
from utilities.telescope import Telescope, taylor_win
from utilities.analysis import SKA1_low
from utilities.telescope_set import TelescopeSet
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import constants as const
from math import log, floor
# from numba import jit


def test1():
    tel = SKA1_low()
    tel.dec_deg = tel.lat_deg
    tel.obs_length_h = 0
    tel.num_times = 1
    tel.grid_cell_size_m = tel.station_diameter_m / 2

    def taper_r_profile(r, num_amps, amps):
        """0.0 <= r < 1.0"""
        i = int(r * num_amps)
        return amps[i]

    # from scipy import interpolate
    n = 5000
    y = taylor_win(n, -28)
    # x = np.linspace(0, 1, n)
    # f = interpolate.interp1d(x, y, kind='cubic')
    opts = dict(num_amps=n, amps=y)
    tel.num_trials = 1
    tel.trail_timeout_s = 2


    tel.add_ska1_v5(r_max=5000)
    # tel.uv_sensitivity(100, 100, 5000, log_bins=True, )
    tel.plot_grid(show=True)
    # tel.eval_psf_rms_r(100)
    # tel.uv_hist(100, b_max=1000, plot=True, log_bins=True, bar=True)
    # tel.uv_sensitivity(100)
    # tel.eval_psf_rms_r(100)
    tel.eval_psf(plot1d=True)

    # FIXME(BM) test comparing two telescopes
    # This could be done by creating two telescope objects and modifying their
    # plot functions to plot into a specified axis as an alternative
    # to creating the figure.

    # tel.network_graph()
    # tel.plot_network()
    # tel.add_tapered_core(200, 500, f)
    # tel.add_tapered_core(200, 500, taper_r_profile, **opts)
    # try_id = tel.layouts['tapered_core']['info']['attempt_id']
    # info = tel.layouts['tapered_core']['info'][try_id]
    # print('- try id:', try_id)
    # print('- Time taken:', info['time_taken'], 's')
    # print('- Time taken:', info['total_tries'], 's')
    # print('- tries per second:', info['total_tries'] / info['time_taken'])
    # print(info['trials'].shape)
    # tel.plot_layout(show=True, plot_decorations=True)


def test2():
    # FIXME(BM) standard taper functions should be methods on the telescope
    # or layout class
    def gauss_taper(r, hwhm=1.5):
        c = hwhm / (2 * log(2))**0.5
        return np.exp(-r**2 / (2 * c**2))

    def add_legend(ax, names, colors):
        # Create legend
        opts = dict(marker='o', linestyle='none', ms=5, mfc='none', mew=1.1)
        handles = list()
        for color in colors:
            handles.append(plt.Line2D([0], [0], mec=color, **opts))
        ax.legend(handles, names, numpoints=1, loc='best', fontsize='medium')

    names = ['SKA1 low reference', 'tel2']
    colors = ['b', 'r']
    fig, ax = plt.subplots(figsize=(8, 8))
    tel = SKA1_low()

    for i, name in enumerate(names):
        tel.name = name
        if i == 7:
            tel.add_ska1_v5(r_max=1700)
            tel.plot_layout(mpl_ax=ax, station_color=colors[0])
        elif i == 1:
            tel.add_log_spiral_clusters(
                num_clusters=5 * 3, num_arms=3, r0=420, r1=1700, b=0.513,
                stations_per_cluster=6, cluster_radius_m=100, theta0_deg=-35)
            tel.add_tapered_core(190, 400, gauss_taper, hwhm=1.5)
            tel.plot_layout(mpl_ax=ax, station_color=colors[i])
        tel.clear_layouts()
    add_legend(ax, names, colors)
    plt.show()


def test3():
    tel1 = SKA1_low('reference')
    tel1.add_ska1_v5(r_max=1700)

    def gauss_taper(r, hwhm=1.5):
        c = hwhm / (2 * log(2))**0.5
        return np.exp(-r**2 / (2 * c**2))

    tel2 = SKA1_low('mod1')
    tel2.add_tapered_core(190, 400, gauss_taper, hwhm=1.5)
    tel2.plot_taper(gauss_taper, hwhm=1.5)
    # tel2.add_log_spiral_clusters(num_clusters=5 * 3, num_arms=3, r0=420,
    #                              r1=1700, b=0.513, stations_per_cluster=6,
    #                              cluster_radius_m=100, theta0_deg=-10)
    tel2.plot_layout(plot_decorations=True)

    # set = TelescopeSet()
    # set.add_telescope(tel1)
    # set.add_telescope(tel2)
    # set.plot_my_metric()


def test4():
    tel = SKA1_low('ref')
    tel.add_ska1_v5(r_max=1700)
    tel.uv_hist(num_bins=50, log_bins=False, bar=True)
    tel.uv_sensitivity(num_bins=50, log_bins=False)


def test5():
    tel = SKA1_low('ref')
    # tel.add_hex_core(300, theta0_deg=60)
    tel.add_uniform_core(10, 500)
    tel.plot_layout(plot_radii=[300])


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    test5()
