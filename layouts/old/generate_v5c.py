# -*- coding: utf-8 -*-
"""Script to generate v5c station coordinates.

Changes:
    03/03/2016: Initial version.
"""
from __future__ import print_function
import numpy
from numpy.random import random, rand
import matplotlib.pyplot as pyplot
try:
    from pyuvwsim import (load_station_coords, convert_enu_to_ecef,
                          evaluate_baseline_uvw)
    uvwsim_found = True
except ImportError:
    print('pyuvwsim not found, skipping uvw co-ordiante generation.')
    print('see: https://github.com/SKA-ScienceDataProcessor/uvwsim, pyuvwsim.rst')
    uvwsim_found = False
from math import radians
import os
from os.path import join
from math import floor
from layout_utils import (gridgen_taylor_padded, gridgen,
                          plot_hist, plot_uv_dist,
                          generate_baseline_uvw)
import math
import pyfits
from oskar.imager import Imager


def save_fits_image_2(file_name, image, cell_size_deg, ra0, dec0, freq_hz):

    # FIXME-BM get ra0, dec0 from the oskar vis file?!
    data = numpy.reshape(image, (1, 1, image.shape[0], image.shape[1]))

    header = pyfits.header.Header()
    header.append(('BUNIT', 'Jy/beam'))
    header.append(('CTYPE1', 'RA--SIN'))
    header.append(('CRVAL1', ra0))
    header.append(('CDELT1', -cell_size_deg))
    header.append(('CUNIT1', 'deg'))
    # Note: Assumes even image dims and that image pixels start at 1 (not 0)
    header.append(('CRPIX1', image.shape[1] / 2 + 1))
    header.append(('CTYPE2', 'DEC-SIN'))
    header.append(('CRVAL2', dec0))
    header.append(('CDELT2', cell_size_deg))
    header.append(('CRPIX2', image.shape[0] / 2 + 1))
    header.append(('CUNIT2', 'deg'))
    header.append(('CTYPE3', 'FREQ'))
    header.append(('CRVAL3', freq_hz))
    header.append(('CDELT3', 1.0))
    header.append(('CRPIX3', 1.0))
    header.append(('CUNIT3', 'Hz'))
    header.append(('CTYPE4', 'STOKES'))
    header.append(('CRVAL4', 1.0))
    header.append(('CDELT4', 1.0))
    header.append(('CRPIX4', 1.0))
    header.append(('CUNIT4', ''))
    if os.path.exists(file_name):
        print('- WARNING: Overwriting FITS file: %s' % file_name)
        os.remove(file_name)
    print('- Saving FITS image:', file_name)
    pyfits.writeto(file_name, data, header)



def generate_random_core(num_stations, core_radius_m, inner_core_radius_m,
                         sll, station_radius_m,
                         num_tries, seed):
    print('Generating %i core stations ...' % num_stations)
    for t in range(num_tries):
        numpy.random.seed(seed + t)
        x, y, miss_count, weights, r_weights = \
            gridgen_taylor_padded(num_stations, core_radius_m * 2.0,
                                  inner_core_radius_m * 2.0,
                                  station_radius_m * 2.0, sll, 5000)

        if x.shape[0] == num_stations:
            print('Done. seed = %i (%i)' % (seed, miss_count.max()))
            break
        else:
            print('%i' % miss_count.shape[0], end=' ')
        if (x.shape[0] / float(num_stations)) < 0.7:
            raise RuntimeError('Failed to generate enough stations in the '
                               'core. %i / %i generated **'
                               % (x.shape[0], num_stations))
    if not x.shape[0] == num_stations:
        raise RuntimeError('Failed to generate enough stations in the '
                           'core. %i / %i generated'
                           % (x.shape[0], num_stations))
    print()
    return x, y, weights, r_weights


def generate_core_arms(num_arms, core_arm_count, stations_per_cluster,
                       arm_cluster_radius, station_radius_m,
                       a, b, delta_theta, arm_offsets, num_tries):
    # Position of station clusters.
    num_clusters = num_arms * core_arm_count
    cluster_x = numpy.zeros(num_clusters)
    cluster_y = numpy.zeros(num_clusters)
    for i in range(num_arms):
        t = numpy.arange(2, core_arm_count + 2) * delta_theta
        tmp = a * numpy.exp(b * t)
        cx = tmp * numpy.cos(t + arm_offsets[i])
        cy = tmp * numpy.sin(t + arm_offsets[i])
        i0 = i * core_arm_count
        i1 = i0 + core_arm_count
        cluster_x[i0:i1] = cx
        cluster_y[i0:i1] = cy

    # Generate stations at the cluster positions.
    num_stations = num_clusters * stations_per_cluster
    arm_x = numpy.zeros(num_stations)
    arm_y = numpy.zeros(num_stations)
    for i in range(num_clusters):
        for t in range(num_tries):
            x, y, _ = gridgen(stations_per_cluster, arm_cluster_radius * 2.0,
                              station_radius_m * 2.0, 10000)
            if x.shape[0] == stations_per_cluster:
                break
            else:
                print('.', end='')
        if not x.shape[0] == stations_per_cluster:
            raise RuntimeError('Did not generate enough stations in arm '
                               'cluster. %i / %i'
                               % (x.shape[0], stations_per_cluster))
        i0 = i * stations_per_cluster
        i1 = i0 + stations_per_cluster
        arm_x[i0:i1] = x + cluster_x[i]
        arm_y[i0:i1] = y + cluster_y[i]

    return arm_x, arm_y, cluster_x, cluster_y


def generate_outer_arms(v4a_ss_enu_file, num_clusters_outer,
                        stations_per_outer_cluster,
                        outer_arm_cluster_radius, station_radius_m,
                        num_tries):
    v4a_ss_enu = numpy.loadtxt(v4a_ss_enu_file)
    v4a_ss_enu = v4a_ss_enu[:, 1:]
    v4a_ss_r = (v4a_ss_enu[:, 0]**2 + v4a_ss_enu[:, 1]**2)**0.5
    sort_idx = numpy.argsort(v4a_ss_r)
    v4a_ss_enu = v4a_ss_enu[sort_idx[::-1], :]
    cluster_x = v4a_ss_enu[:num_clusters_outer, 0]
    cluster_y = v4a_ss_enu[:num_clusters_outer, 1]

    # Generate stations at the cluster positions.
    num_stations = num_clusters_outer * stations_per_outer_cluster
    arm_x = numpy.zeros(num_stations)
    arm_y = numpy.zeros(num_stations)
    for i in range(num_clusters_outer):
        for t in range(num_tries):
            x, y, _ = gridgen(stations_per_outer_cluster,
                              outer_arm_cluster_radius * 2.0,
                              station_radius_m * 2.0, 10000)
            if x.shape[0] == stations_per_outer_cluster:
                break
            else:
                print('.', end='')
        if not x.shape[0] == stations_per_outer_cluster:
            raise RuntimeError('Did not generate enough stations in outer arm '
                               'cluster. %i / %i'
                               % (x.shape[0], stations_per_outer_cluster))
        i0 = i * stations_per_outer_cluster
        i1 = i0 + stations_per_outer_cluster
        arm_x[i0:i1] = x + cluster_x[i]
        arm_y[i0:i1] = y + cluster_y[i]

    return arm_x, arm_y, cluster_x, cluster_y


def plot_layout(x_core, y_core, x_arm, y_arm, x_arm_outer, y_arm_outer,
                cx_arm, cy_arm, cx_outer, cy_outer, station_radius_m,
                inner_core_radius_m, core_radius_m, arm_cluster_radius,
                outer_arm_cluster_radius, out_dir):
    # Plotting
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')

    circle = pyplot.Circle((0.0, 0.0),
                           core_radius_m, color='k',
                           fill=False, alpha=0.5, linewidth=1.0)
    ax.add_artist(circle)
    circle = pyplot.Circle((0.0, 0.0),
                           inner_core_radius_m, color='k',
                           fill=False, alpha=0.5, linewidth=1.0)
    ax.add_artist(circle)
    for i in range(x_core.shape[0]):
        circle = pyplot.Circle((x_core[i], y_core[i]),
                               station_radius_m, color='r',
                               fill=True, alpha=0.4, linewidth=1.0)
        ax.add_artist(circle)
    for i in range(cx_arm.shape[0]):
        circle = pyplot.Circle((cx_arm[i], cy_arm[i]),
                               arm_cluster_radius, color='k',
                               fill=False, alpha=0.5, linewidth=1.0)
        ax.add_artist(circle)
    for i in range(x_arm.shape[0]):
        circle = pyplot.Circle((x_arm[i], y_arm[i]),
                               station_radius_m, color='g',
                               fill=True, alpha=0.4, linewidth=1.0)
        ax.add_artist(circle)

    for i in range(cx_outer.shape[0]):
        circle = pyplot.Circle((cx_outer[i], cy_outer[i]),
                               outer_arm_cluster_radius, color='k',
                               fill=False, alpha=0.5, linewidth=1.0)
        ax.add_artist(circle)
    for i in range(x_arm_outer.shape[0]):
        circle = pyplot.Circle((x_arm_outer[i], y_arm_outer[i]),
                               station_radius_m, color='y',
                               fill=True, alpha=0.4, linewidth=1.0)
        ax.add_artist(circle)

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1.0)
    ax.set_ylabel('North [m]')
    ax.set_xlabel('East [m]')
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    pyplot.savefig(join(out_dir, 'layout_00.5km.png'))
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    pyplot.savefig(join(out_dir, 'layout_01.5km.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'layout_03.0km.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'layout_05.0km.png'))
    ax.set_xlim(-50000, 50000)
    ax.set_ylim(-50000, 50000)
    pyplot.savefig(join(out_dir, 'layout_50.0km.png'))
    pyplot.close(fig)


def plot_core_thinning_profile(r_weights, weights, core_radius_m,
                               inner_core_radius_m, out_dir):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(r_weights, weights)
    ax.set_ylim(0, 1.1)
    ax.plot([inner_core_radius_m, inner_core_radius_m], ax.get_ylim(), 'r-')
    ax.plot([core_radius_m, core_radius_m], ax.get_ylim(), 'r-')
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('Thinning weight')
    pyplot.savefig(join(out_dir, 'core_thinning_weights.png'))
    pyplot.close(fig)


def main():
    """
    1. Generate a large-ish core of stations using random generator.
         a. overlap some stations in the core to have a very dense station
            region
    2. After core area start using arms but generate some randomness in the arms
       by placing antennas randomly near the outer stations keeping them along
       the spiral
    3. Remove radius redundancy in the spiral arms
    """

    # =========================================================================

    # ====== Core
    seed = 1
    num_tries = 10
    num_core_stations = (1 + 5 + 11 + 17) * 6 + (3 * 6)
    core_radius_m = 480.0
    inner_core_radius_m = 280.0
    station_radius_m = 35.0 / 2.0
    sll = -28
    # ====== Core arms
    num_arms = 3
    core_arm_count = 4
    stations_per_arm_cluster = 6
    arm_cluster_radius = 75.0
    # a = 300.0
    # b = 0.513
    a = 300.0
    b = 0.513
    delta_theta = math.radians(37.0)
    arm_offsets = numpy.radians([35.0, 155.0, 270.0])
    num_core_arm_stations = num_arms * core_arm_count * stations_per_arm_cluster
    # ====== Outer arms
    outer_arm_count = 12
    stations_per_outer_cluster = 6
    num_clusters_outer = outer_arm_count * num_arms
    v4a_ss_enu_file = 'v7ska1lowN1v2rev3R.enu.94x4.fixed.txt'
    outer_arm_cluster_radius = 80.0

    # ===== uvw coordinate generation.
    lon = radians(116.63128900)
    lat = radians(-26.69702400)
    alt = 0.0
    ra = radians(68.698903779331502)
    dec = radians(-26.568851215532160)
    mjd_mid = 57443.4375000000

    obs_length = 0.0
    mjd_start = mjd_mid
    dt_s = 0.0
    num_times = 1

    obs_length = 2.0 * 3600.0  # seconds
    num_times = int(obs_length / (3.0 * 60.0))
    dt_s = obs_length / float(num_times)
    mjd_start = mjd_mid - ((obs_length / 2.0) / 3600.0 * 24.0)
    print('num times = %i' % num_times)

    out_dir = 'v5c-2h'
    # =========================================================================
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Generate core stations
    x_core, y_core, weights, r_weights = \
        generate_random_core(num_core_stations, core_radius_m,
                             inner_core_radius_m, sll, station_radius_m,
                             num_tries, seed)

    # Core arms
    x_arm, y_arm, cx_arm, cy_arm = \
        generate_core_arms(num_arms, core_arm_count, stations_per_arm_cluster,
                           arm_cluster_radius, station_radius_m,
                           a, b, delta_theta, arm_offsets, num_tries)

    # Outer stations.
    x_arm_outer, y_arm_outer, cx_outer, cy_outer = \
        generate_outer_arms(v4a_ss_enu_file, num_clusters_outer,
                            stations_per_outer_cluster,
                            outer_arm_cluster_radius, station_radius_m,
                            num_tries)

    # Plotting
    plot_layout(x_core, y_core, x_arm, y_arm, x_arm_outer, y_arm_outer,
                cx_arm, cy_arm, cx_outer, cy_outer, station_radius_m,
                inner_core_radius_m, core_radius_m, arm_cluster_radius,
                outer_arm_cluster_radius, out_dir)
    plot_core_thinning_profile(r_weights, weights, core_radius_m,
                               inner_core_radius_m, out_dir)

    if uvwsim_found:
        x = numpy.hstack((x_core, x_arm, x_arm_outer))
        y = numpy.hstack((y_core, y_arm, y_arm_outer))
        print('total stations = %i' % x.shape[0])
        num_stations = x.shape[0]
        z = numpy.zeros_like(x)

        num_baselines = num_stations * (num_stations - 1) / 2
        x, y, z = convert_enu_to_ecef(x, y, z, lon, lat, alt)
        uu, vv, ww = generate_baseline_uvw(x, y, z, ra, dec, num_times,
                                           num_baselines, mjd_start,
                                           dt_s)
        plot_hist(uu, vv, join(out_dir, 'uv_hist_%.2fh.png'
                               % (obs_length/3600.0)),
                  'v5c %.2f h' % (obs_length/3600.0))
        plot_uv_dist(uu, vv, station_radius_m, join(out_dir, 'uv_%.2fh'
                                                    % (obs_length/3600.0)))
        # TODO-BM see ALMA memo for plots?
        # TODO-BM Plot of azimuthal variation
        # TODO-BM movie of uv coverage histogram improvement with time?
        # TODO-BM convolve uv response with station beam?!

    print('making image...')
    imager = Imager('single')
    fov = 1.0
    im_size = 2048
    freq = 150.0e6
    wavelength = 299792458.0 / freq
    uu /= wavelength
    vv /= wavelength
    ww /= wavelength
    amp = numpy.ones(uu.shape, dtype='c16')
    weight = numpy.ones(uu.shape, dtype='f8')
    image = imager.make_image(uu, vv, ww, amp, weight, fov, im_size)
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(image, interpolation='nearest')
    pyplot.show()

    cell = math.degrees(imager.fov_to_cellsize(math.radians(fov), im_size))
    save_fits_image_2(join(out_dir, 'psf.fits'), image, cell, math.degrees(ra),
                      math.degrees(dec), freq)


if __name__ == '__main__':
    main()
