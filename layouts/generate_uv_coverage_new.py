# -*- coding: utf-8 -*-
from __future__ import print_function

import shutil
from math import radians
from os.path import join

import pyuvwsim
import numpy
from utilities.layout_utils import generate_baseline_uvw
import time
import os
from utilities.plotting_layout import plot_layouts_2
from utilities.plotting_psf import plot_psf_2
from utilities.plotting_uv_scatter import uv_plot_2
import shutil
from collections import OrderedDict
import matplotlib.pyplot as pyplot


def main():
    out_dir = 'TEMP_0h'
    layouts = OrderedDict()
    layouts['v5'] = {'filename': join('v5.tm', 'layout.txt')}
    layouts['v5c'] = {'filename': join('v5c.tm', 'layout_enu_stations.txt')}
    station_radius_m = 40.0 / 2.0
    freq_hz = 100.0e6
    wave_length = 299792458.0 / freq_hz
    lon = radians(116.63128900)
    lat = radians(-26.69702400)
    ra = radians(68.698903779331502)
    dec = radians(-26.568851215532160)
    mjd_mid = 57443.4375000000
    snapshot = True
    if snapshot:
        mjd_start = mjd_mid
        obs_length = 0.0
        dt_s = 0.0
        num_times = 1
    else:
        obs_length = 4.0 * 3600.0  # seconds
        num_times = int(obs_length / (1 * 60.0))
        dt_s = obs_length / float(num_times)
        mjd_start = mjd_mid - (obs_length / 2.0) / (3600.0 * 24.0)

    # Create output directory and copy script =================================
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    shutil.copy(__file__, join(out_dir, 'copy_' + os.path.basename(__file__)))

    # Load station coordinates & generate baseline coordinates ================
    for name in layouts:
        coords = numpy.loadtxt(layouts[name]['filename'])
        layouts[name]['station_coords'] = coords
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        # TODO(BM) remove stations > xx km
        r = (x**2 + y**2)**0.5
        sort_idx = numpy.argsort(r)
        r = r[sort_idx]
        x = x[sort_idx]
        y = y[sort_idx]
        z = z[sort_idx]
        x = x[r <= 5000.0]
        y = y[r <= 5000.0]
        z = z[r <= 5000.0]
        print(name, x.shape)
        x, y, z = pyuvwsim.convert_enu_to_ecef(x, y, z, lon, lat)
        uu, vv, ww = generate_baseline_uvw(x, y, z, ra, dec, num_times,
                                           mjd_start, dt_s)
        layouts[name]['uu'] = uu
        layouts[name]['vv'] = vv
        layouts[name]['ww'] = ww

    print('- UV coordinate generation complete.')
    print('- obs_length = %.2f s (%.2f h)' % (obs_length, obs_length / 3600.0))
    print('- num_times =', num_times)

    # Plotting ================================================================
    t0 = time.time()
    plot_layouts_2(layouts, station_radius_m, join(out_dir, 'layouts'))
    uv_plot_2(layouts, join(out_dir, 'uv_scatter'))
    plot_psf_2(layouts, freq_hz, 60.0, 4096, join(out_dir, 'psf'))
    plot_psf_2(layouts, freq_hz, 30.0, 4096, join(out_dir, 'psf'))
    plot_psf_2(layouts, freq_hz, 5.0, 4096, join(out_dir, 'psf'))
    plot_psf_2(layouts, freq_hz, 1.0, 4096, join(out_dir, 'psf'))
    # plot_uv_hist(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length,
    #              join(out_dir, 'uv_hist'))
    # plot_uv_images(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length,
    #                station_radius_m, join(out_dir, 'uv_images'))
    # plot_az_rms_2(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length,
    #               join(out_dir, 'uv_az'))
    print('- Plotting took %.2f s' % (time.time() - t0))

if __name__ == '__main__':
    main()
