# -*- coding: utf-8 -*-
from __future__ import print_function
from os.path import join
import numpy
import matplotlib.pyplot as pyplot
from math import radians
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


def plot_layouts(v4d, v5d, station_radius_m):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(v4d.shape[0]):
        circle = pyplot.Circle((v4d[i, 0], v4d[i, 1]), station_radius_m,
                               color='b', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_title('v4d')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(True)
    pyplot.show()

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(v4d.shape[0]):
        circle = pyplot.Circle((v5d[i, 0], v5d[i, 1]), station_radius_m,
                               color='b', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_title('v5d')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(True)
    pyplot.show()


def uv_plot(uu_v4d, vv_v4d, uu_v5d, vv_v5d, out_dir):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4d, vv_v4d, 'k.', alpha=0.1, ms=2.0)
    ax.plot(-uu_v4d, -vv_v4d, 'k.', alpha=0.1, ms=2.0)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_title('v4d')
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    pyplot.savefig(join(out_dir, 'v4d_1000m.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v5d, vv_v5d, 'k.', alpha=0.1, ms=2.0)
    ax.plot(-uu_v5d, -vv_v5d, 'k.', alpha=0.1, ms=2.0)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_title('v5d')
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    pyplot.savefig(join(out_dir, 'v5d_1000m.png'))
    pyplot.close(fig)

if __name__ == '__main__':
    # Load station positions
    t0 = time.time()
    v4d_file = join('v4d.tm', 'layout_enu_stations.txt')
    v5d_file = join('v5d.tm', 'layout_enu_stations.txt')
    v4d = numpy.loadtxt(v4d_file)
    v5d = numpy.loadtxt(v5d_file)
    station_radius_m = 35.0 / 2.0
    num_stations = v4d.shape[0]
    assert(v5d.shape[0] == v4d.shape[0])
    print('- loading coordinates took %.2f s' % (time.time() - t0))

    freq = 100.0e6
    wave_length = 299792458.0 / freq
    lon = radians(116.63128900)
    lat = radians(-26.69702400)
    alt = 0.0
    ra = radians(68.698903779331502)
    dec = radians(-26.568851215532160)
    mjd_mid = 57443.4375000000
    # obs_length = 4.0 * 3600.0  # seconds
    # num_times = int(obs_length / (3 * 60.0))
    # # print('num_times =', num_times)
    # dt_s = obs_length / float(num_times)
    # mjd_start = mjd_mid - (obs_length / 2.0) / (3600.0 * 24.0)
    mjd_start = mjd_mid
    obs_length = 0.0
    dt_s = 0.0
    num_times = 1
    num_baselines = num_stations * (num_stations - 1) / 2
    out_dir = 'uv_%3.1fh' % (obs_length / 3600.0)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    t0 = time.time()
    x, y, z = convert_enu_to_ecef(v4d[:, 0], v4d[:, 1], v4d[:, 2],
                                  lon, lat, alt)
    uu_v4d, vv_v4d, ww_v4d = \
        generate_baseline_uvw(x, y, z, ra, dec, num_times, num_baselines,
                              mjd_start, dt_s)
    x, y, z = convert_enu_to_ecef(v5d[:, 0], v5d[:, 1], v5d[:, 2],
                                  lon, lat, alt)
    uu_v5d, vv_v5d, ww_v5d = \
        generate_baseline_uvw(x, y, z, ra, dec, num_times, num_baselines,
                              mjd_start, dt_s)
    print('- coordinate generation took %.2f s' % (time.time() - t0))

    uv_plot(uu_v4d, vv_v4d, uu_v5d, vv_v5d, out_dir)

    print('wavelength = %.2f m' % wave_length)
    uv_dist = (uu_v4d**2 + vv_v4d**2)**0.5
    uv_dist /= wave_length
    # Remove all points larger than x
    uv_dist.sort()
    uv_dist = uv_dist[uv_dist < 500.0]
    uv_dist_range = uv_dist.max()  # - uv_dist.min()
    bin_width = 1.0  # wavelengths
    num_bins = numpy.ceil(uv_dist_range / bin_width)
    print('radius max = %f' % uv_dist.max())
    print('num_bins = %i' % num_bins)
    bin_inc = uv_dist_range / num_bins
    bins = numpy.arange(num_bins) * bin_inc
    hist, bin_edges = numpy.histogram(uv_dist, bins=bins, density=False)
    # x = bin_edges[:-1] + numpy.diff(bin_edges) / 2.0
    x = bin_edges[:-1]
    y = hist
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    # ax.loglog(x, y, '-')
    # ax.hist(x, data=y)
    ax.bar(x, y, width=numpy.diff(bin_edges), color='none')

    uv_dist = (uu_v5d**2 + vv_v5d**2)**0.5
    uv_dist /= wave_length
    # Remove all points larger than x
    uv_dist.sort()
    uv_dist = uv_dist[uv_dist < 500.0]
    uv_dist_range = uv_dist.max()  # - uv_dist.min()
    bin_width = 1.0  # wavelengths
    num_bins = numpy.ceil(uv_dist_range / bin_width)
    print('radius max = %f' % uv_dist.max())
    print('num_bins = %i' % num_bins)
    bin_inc = uv_dist_range / num_bins
    bins = numpy.arange(num_bins) * bin_inc
    hist, bin_edges = numpy.histogram(uv_dist, bins=bins, density=False)
    # x = bin_edges[:-1] + numpy.diff(bin_edges) / 2.0
    x = bin_edges[:-1]
    y = hist

    ax.bar(x, y, width=numpy.diff(bin_edges), color='r', edgecolor='r', alpha=0.2)

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 600)
    pyplot.show()
    pyplot.close(fig)

    # TODO-BM: Generate plots to a folder based on observation length.
    # TODO-BM: Histogram
    # TODO-BM: UV image plots
    # TODO-BM: PSF images
