# -*- coding: utf-8 -*-
from __future__ import print_function
from os.path import join
import numpy
from math import radians
import shutil
try:
    from pyuvwsim import (load_station_coords, convert_enu_to_ecef,
                          evaluate_baseline_uvw)
    uvwsim_found = True
except ImportError:
    print('pyuvwsim not found, skipping uvw co-ordiante generation.')
    print('see: https://github.com/SKA-ScienceDataProcessor/uvwsim, pyuvwsim.rst')
    uvwsim_found = False
from layout_utils import (generate_baseline_uvw)
import time
import os
try:
    from oskar.imager import Imager
    oskar_imager_found = True
except ImportError:
    print('OSKAR python imager not found, PSF images wont be made.')
    oskar_imager_found = False
from plotting_psf import plot_psf


def main():
    # Load station positions
    t0 = time.time()
    v4d_file = join('v4d.tm', 'layout_enu_stations.txt')
    v4o1_file = join('v4o1.tm', 'layout_enu_stations.txt')
    v4d = numpy.loadtxt(v4d_file)
    v4o1 = numpy.loadtxt(v4o1_file)
    station_radius_m = 35.0 / 2.0
    num_stations = v4d.shape[0]
    assert(v4o1.shape[0] == v4d.shape[0])
    print('- loading coordinates took %.2f s' % (time.time() - t0))

    freq = 100.0e6
    wave_length = 299792458.0 / freq
    lon = radians(116.63128900)
    lat = radians(-26.69702400)
    alt = 0.0
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
        num_times = int(obs_length / (3 * 60.0))
        dt_s = obs_length / float(num_times)
        mjd_start = mjd_mid - (obs_length / 2.0) / (3600.0 * 24.0)

    print('- obs_length = %.2f s (%.2f h)' % (obs_length, obs_length / 3600.0))
    print('- num_times =', num_times)

    num_baselines = num_stations * (num_stations - 1) / 2
    out_dir = 'uv_%3.1fh' % (obs_length / 3600.0)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    # if not os.path.isdir(out_dir):
    #     os.makedirs(out_dir)

    # plot_layouts(v4d, v4o1, station_radius_m, out_dir)
    t0 = time.time()
    x, y, z = convert_enu_to_ecef(v4d[:, 0], v4d[:, 1], v4d[:, 2],
                                  lon, lat, alt)
    uu_v4d, vv_v4d, ww_v4d = \
        generate_baseline_uvw(x, y, z, ra, dec, num_times, num_baselines,
                              mjd_start, dt_s)
    x, y, z = convert_enu_to_ecef(v4o1[:, 0], v4o1[:, 1], v4o1[:, 2],
                                  lon, lat, alt)
    uu_v4o1, vv_v4o1, ww_v4o1 = \
        generate_baseline_uvw(x, y, z, ra, dec, num_times, num_baselines,
                              mjd_start, dt_s)
    print('- coordinate generation took %.2f s' % (time.time() - t0))
    print('- num vis = %i' % uu_v4d.shape[0])

    plot_psf(uu_v4d, vv_v4d, ww_v4d, uu_v4o1, vv_v4o1, ww_v4o1,
             freq, join(out_dir, 'psf'))


if __name__ == '__main__':
    main()
