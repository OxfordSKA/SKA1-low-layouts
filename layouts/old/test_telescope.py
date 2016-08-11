# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import time
from utilities.telescope import Telescope, taylor_win
from utilities.analysis import TelescopeAnalysis, SKA1_v5, SKA1_low
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import constants as const
from math import log


def test1():
    tel = Telescope()
    tel.add_uniform_core(200, 1000)
    fig = tel.plot_layout()
    ax = fig.gca()
    ax.add_artist(plt.Circle((0, 0), radius=1000, fill=False, color='r'))
    plt.show()
    plt.close(fig)


def test2():
    tel = Telescope()
    tel.add_uniform_core(200, 1000)
    tel.add_ska1_v5(r_min=1000)
    tel.add_ska1_v5(r_min = 500, r_max = 1000)
    fig = tel.plot_layout()
    ax = fig.gca()
    ax.add_artist(plt.Circle((0, 0), radius=500, fill=False, color='r'))
    ax.add_artist(plt.Circle((0, 0), radius=1000, fill=False, color='r'))
    plt.show()
    plt.close(fig)


def test3():
    tel = Telescope()
    tel.add_uniform_core(368, 5000)
    tel.add_ska1_v5(r_min=5000)
    x, y, z = tel.coords()
    fig, ax = plt.subplots()
    for xy in zip(x, y):
        ax.add_artist(plt.Circle(xy, radius=(tel.station_diameter_m / 2),
                                 fill=False))
    ax.set_aspect('equal')
    ax.set_xlim(-8000, 8000)
    ax.set_ylim(-8000, 8000)
    plt.show()
    plt.close(fig)


def test4():
    ska1 = SKA1_v5()
    ska1.plot_layout(plot_r = 10000, show=True)


def test5():
    tel = TelescopeAnalysis()
    tel.add_ska1_v5(r_min=0, r_max=1000)
    # tel.add_uniform_core(368, 5000)
    fig = tel.plot_layout(filename='zzz_layout.png')
    tel.lon_deg = 116.631289
    tel.lat_deg = -26.697024
    tel.dec_deg = tel.lat_deg
    tel.obs_length_h = 4
    tel.num_times = int((tel.obs_length_h * 3600) / 30)
    tel.grid_cell_size_m = tel.station_diameter_m / 2
    tel.gen_uvw_coords()
    tel.grid_uvw_coords()
    fig = tel.plot_grid()
    fig.savefig('zzz_uv_grid_%02.1fh_%i.png' % (tel.obs_length_h, tel.num_times))
    plt.close(fig)


def test6():
    tel = TelescopeAnalysis()
    tel.add_uniform_core(400, 500)
    tel.add_log_spiral(12 * 3, 535, 1500, 0.5, 3)
    tel.lon_deg = 116.631289
    tel.lat_deg = -26.697024
    tel.dec_deg = tel.lat_deg
    # tel.plot_layout(filename='zzz_layout.png')
    # tel.obs_length_h = 4
    # tel.num_times = int((tel.obs_length_h * 3600) / 30)
    tel.obs_length_h = 0
    tel.num_times = 1
    tel.grid_cell_size_m = tel.station_diameter_m / 2
    tel.gen_uvw_coords()
    # tel.grid_uvw_coords()
    # tel.plot_grid('zzz_uv_grid_%02.1fh_%i.png' %
    #               (tel.obs_length_h, tel.num_times))
    tel.uv_hist(100)


def test7():
    tel = TelescopeAnalysis()

    # Settings
    tel.lon_deg = 116.631289
    tel.lat_deg = -26.697024
    tel.dec_deg = tel.lat_deg
    tel.obs_length_h = 0
    tel.num_times = 1
    tel.grid_cell_size_m = tel.station_diameter_m / 2

    # Telescope 1: ska v5 < 1500m
    tel.add_ska1_v5(r_max=1500)
    tel.plot_layout(filename='zzz_layout_ref.png', plot_decorations=False,
                    x_lim=[-1.5e3, 1.5e3], y_lim=[-1.5e3, 1.5e3],
                    plot_radii=[500, 1500])
    tel.gen_uvw_coords()
    tel.grid_uvw_coords()
    tel.plot_grid(filename='zzz_uvw_ref_%02.1fh.png' % tel.obs_length_h,
                  plot_radii=[1500],
                  x_lim=[-3e3, 3e3], y_lim=[-3e3, 3e3])
    tel.obs_length_h = 4
    tel.num_times = int((tel.obs_length_h * 3600) / 60)
    tel.gen_uvw_coords()
    tel.grid_uvw_coords()
    tel.plot_grid(filename='zzz_uvw_ref_%02.1fh.png' % tel.obs_length_h,
                  plot_radii=[1500],
                  x_lim=[-3e3, 3e3], y_lim=[-3e3, 3e3])

    # Clear the telescope model
    tel.clear()

    # Telescope 2
    tel.add_ska1_v5(r_max=500)
    # tel.add_log_spiral_clusters(3 * 3, 3, 535, 1500, 0.5, 6, 70)
    tel.add_log_spiral(3 * 6 * 3, 535, 1500, 0.5, 3)
    tel.plot_layout(plot_decorations=False, filename='zzz_layout_2.png',
                    x_lim=[-1.5e3, 1.5e3], y_lim=[-1.5e3, 1.5e3],
                    plot_radii=[500, 1500])
    tel.gen_uvw_coords()
    tel.grid_uvw_coords()
    tel.plot_grid(filename='zzz_uvw_2_%02.1fh.png' % tel.obs_length_h,
                  plot_radii=[1500],
                  x_lim=[-3e3, 3e3], y_lim=[-3e3, 3e3])
    tel.obs_length_h = 4
    tel.num_times = int((tel.obs_length_h * 3600) / 60)
    tel.gen_uvw_coords()
    tel.grid_uvw_coords()
    tel.plot_grid(filename='zzz_uvw_2_%02.1fh.png' % tel.obs_length_h,
                  plot_radii=[1500],
                  x_lim=[-3e3, 3e3], y_lim=[-3e3, 3e3])


def test8():
    tel = SKA1_low()
    tel.dec_deg = tel.lat_deg
    tel.obs_length_h = 0
    tel.num_times = 1
    tel.grid_cell_size_m = tel.station_diameter_m / 2

    def taper_func(r):
        return 1 - (r / 2)**1.1

    tel.add_tapered_core(195, 500, taper_func)
    tel.add_ska1_v5(r_min=500, r_max=1500)

    tel.plot_layout(filename='zzz_test_8_layout.png', plot_decorations=True,
                    plot_radii=[500])


def test9():
    def taper_func_1(r):
        return 1 - (r / 2)**1.5

    def taper_func_2(r, hwhm=1.5):
        c = hwhm / (2 * log(2))**0.5
        return np.exp(-r**2 / (2 * c**2))

    def taper_r_profile(r, amps):
        n = amps.shape[0]
        # Find the index nearest to r
        # FIXME(BM) check this is right...
        i = np.round(r * (n - 1)).astype(np.int)
        return amps[i]

    def get_taper_profile(taper_func, r, **kwargs):
        if kwargs is not None:
            t = taper_func(r, **kwargs)
        else:
            t = taper_func(r)
        return t

    r = np.linspace(0, 1, 100)
    # r = np.random.rand(100)
    t1 = get_taper_profile(taper_func_1, r)
    t2 = get_taper_profile(taper_func_2, r, hwhm=0.7)
    t3 = get_taper_profile(taper_r_profile, r, amps=taylor_win(1000, -22))

    fig, ax = plt.subplots()
    ax.plot(r, t1, 'b.', label='t1')
    ax.plot(r, t2, 'r.', label='gauss')
    ax.plot(r, t3, 'g.', label='r profile (taylor)')

    ax.plot(ax.get_xlim(), [0.5, 0.5], 'r--')
    ax.plot([0.7, 0.7], ax.get_ylim(), 'r--')
    ax.grid()
    ax.legend()
    plt.show()

    return
    # r_max = 500
    # n = 11
    # sll = -28
    # nbar = int(np.ceil(2.0 * (np.arccosh(10**(-sll / 20.0)) / np.pi)**2 + 0.5))

    # # r = np.arange(w.shape[0]) * ((r_max * 2) / (n + 1))
    # r = np.linspace(0, 1, n)
    # print(r.shape, w.shape)

    fig, ax = plt.subplots()
    ax.plot(r, w, '.-')
    ax.grid()
    plt.show()


def test10():
    n = 5
    x = np.linspace(0, 1, n)
    y = taylor_win(n, -28)

    from scipy import interpolate
    f = interpolate.interp1d(x, y, kind='cubic')
    x_new = np.random.rand(10000)
    x_new.sort()
    fig, ax = plt.subplots()
    ax.plot(x, y, '+-')
    ax.plot(x_new, f(x_new), '-')
    plt.show()


def test11():
    tel = SKA1_low()
    tel.dec_deg = tel.lat_deg
    tel.obs_length_h = 0
    tel.num_times = 1
    tel.grid_cell_size_m = tel.station_diameter_m / 2

    def gauss_taper(r, hwhm=1.5):
        c = hwhm / (2 * log(2))**0.5
        return np.exp(-r**2 / (2 * c**2))

    # tel.seed = 56332004
    tel.plot_min_sep(500, gauss_taper, hwhm=0.9)

    tel.add_tapered_core(80, 500, gauss_taper, hwhm=0.5)
    print('seed', tel.layouts['tapered_core']['info']['final_seed'])
    print(tel.layouts['tapered_core']['info']['attempt_id'])


    # tel.plot_layout(filename='zzz_test_11_layout.png', plot_decorations=True,
    #                 plot_radii=[500], x_lim=[-500, 500], y_lim=[-500, 500])
    tel.plot_layout(show=True, plot_decorations=True,
                    plot_radii=[500], x_lim=[-500, 500], y_lim=[-500, 500])
    # tel.plot_taper(gauss_taper, hwhm=0.5)


def test12():
    tel = SKA1_low()
    tel.dec_deg = tel.lat_deg
    tel.obs_length_h = 0
    tel.num_times = 1
    tel.grid_cell_size_m = tel.station_diameter_m / 2

    def taper_r_profile(r, amps):
        n = amps.shape[0]
        i = np.round(r * (n - 1)).astype(np.int)
        return amps[i]

    from scipy import interpolate
    n = 100
    x = np.linspace(0, 1, n)
    y = taylor_win(n, -28)
    f = interpolate.interp1d(x, y, kind='cubic')

    # tel.plot_min_sep(500, f)
    tel.num_trials = 3
    tel.trail_timeout_s = 20
    tel.add_tapered_core(150, 500, f)
    try_id = tel.layouts['tapered_core']['info']['attempt_id']
    info = tel.layouts['tapered_core']['info'][try_id]
    print(try_id, info.keys(), info['max_tries'], info['total_tries'],
          info['time_taken'], info['total_tries']/info['time_taken'])
    # tel.plot_layout(show=True, plot_decorations=True,
    #                 plot_radii=[500], x_lim=[-500, 500], y_lim=[-500, 500])
    # tel.plot_layout(show=True, plot_decorations=False,
    #                 plot_radii=[500], x_lim=[-500, 500], y_lim=[-500, 500])


def test13():

    import pandas as pd
    from bokeh.plotting import figure, show
    import datashader as ds
    from datashader import transfer_functions as tf
    from functools import partial

    import datashader as ds
    from datashader import transfer_functions as tf
    from datashader.colors import Greys9, Hot
    from datashader.bokeh_ext import InteractiveImage
    from datashader import reductions
    import matplotlib

    Greys9_r = list(reversed(Greys9))[:-2]

    tel = SKA1_low()
    tel.add_ska1_v5(r_max=1000)
    tel.dec_deg = tel.lat_deg
    tel.obs_length_h = 4
    tel.num_times = int((tel.obs_length_h * 3600) / 60)
    tel.gen_uvw_coords()
    tel.grid_uvw_coords()
    # print(tel.num_coords())

    uu = np.hstack([tel.uu_m, -tel.uu_m])
    vv = np.hstack([tel.vv_m, -tel.vv_m])

    df = pd.DataFrame(np.vstack([uu, vv, np.ones_like(uu)]).transpose(),
                      columns=list('uvc'))
    print(df.tail())

    img = tf.interpolate(ds.Canvas().points(df, 'u', 'v'))
    print(img.data.min(), img.data.max())
    print(img.values.min(), img.values.max())
    print(tel.uv_grid.min(), tel.uv_grid.max())

    # img.plot(cmap='gray_r')
    # plt.show()



    # x_range = (min(df['u']), max(df['u']))
    # y_range = (min(df['v']), max(df['v']))

    # cvs = ds.Canvas(x_range=x_range, y_range=y_range)
    # agg = cvs.points(df, 'u', 'v')
    # cmap = cmap=matplotlib.cm.gray_r
    # img = tf.interpolate(agg, cmap=cmap, how='log')
    # img.plot(cmap='gray_r')
    # plt.show()
    # print(img.data.shape)
    # print(img.values.shape)
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(img.data, interpolation='nearest',
    #                origin='lower', cmap=cmap)
    # fig.colorbar(im, ax=ax)
    # plt.show()


def test14():
    import pandas as pd
    import numpy as np
    import xarray as xr
    import datashader as ds
    import datashader.glyphs
    import datashader.transfer_functions as tf
    from collections import OrderedDict
    import matplotlib.pyplot as plt
    from datashader import reductions

    np.random.seed(1)
    num = 10000
    dists = {cat: pd.DataFrame(dict(x=np.random.normal(x, s, num),
                                    y=np.random.normal(y, s, num),
                                    val=val, cat=cat))
             for x, y, s, val, cat, in
             [(2, 2, 0.01, 10, "d1"), (2, -2, 0.1, 20, "d2"),
              (-2, -2, 0.5, 30, "d3"), (-2, 2, 1.0, 40, "d4"),
              (0, 0, 3, 50, "d5")]
             }
    df = pd.concat(dists, ignore_index=True)
    df['cat'] = df['cat'].astype('category')
    print(df.tail())
    print('-' * 20)
    plt.plot(df['x'], df['y'], 'k.', ms=3, alpha=0.1)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()

    cvs = ds.Canvas(plot_width=200, plot_height=200, x_range=(-8, 8), y_range=(-8, 8))
    agg = cvs.points(df, 'x', 'y', agg=reductions.count())
    img = tf.interpolate(agg)
    plt.imshow(img)
    plt.show()


    # img.plot(cmap='gray')
    # plt.show()


if __name__ == '__main__':
    test6()
