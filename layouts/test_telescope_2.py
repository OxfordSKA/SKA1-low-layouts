# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import time
from utilities.telescope import Telescope, taylor_win
from utilities.analysis import SKA1_low
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
    # tel.add_tapered_core(200, 500, f)
    tel.add_tapered_core(220, 500, taper_r_profile, **opts)
    tel.add_log_spiral()
    tel.add_ska1_v5(r)




    # try_id = tel.layouts['tapered_core']['info']['attempt_id']
    # info = tel.layouts['tapered_core']['info'][try_id]
    # print('- try id:', try_id)
    # print('- Time taken:', info['time_taken'], 's')
    # print('- Time taken:', info['total_tries'], 's')
    # print('- tries per second:', info['total_tries'] / info['time_taken'])
    # print(info['trials'].shape)
    # tel.plot_layout(show=True, plot_decorations=True)


if __name__ == '__main__':
    test1()
