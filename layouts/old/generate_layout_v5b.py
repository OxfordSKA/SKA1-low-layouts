# -*- coding: utf-8 -*-
"""Script to generate v5b station coordinates.

Changes:
    20/05/2016: Initial version.
"""
from __future__ import print_function

import math
import os
import sys
from os.path import join

import matplotlib.pyplot as pyplot
import numpy
from utilities.layout_utils import (gridgen_taylor_padded, gridgen,
                                    rotate_coords)


def generate_random_core(num_stations, core_radius_m, inner_core_radius_m,
                         sll, station_radius_m,
                         num_tries, seed):
    print('Generating %i core stations ... (%i tries)' %
          (num_stations, num_tries))
    sys.stdout.flush()
    for t in range(num_tries):
        numpy.random.seed(seed + t)
        x, y, miss_count, weights, r_weights = \
            gridgen_taylor_padded(num_stations, core_radius_m * 2.0,
                                  inner_core_radius_m * 2.0,
                                  station_radius_m * 2.0, sll, 50000)
        if x.shape[0] == num_stations:
            print('Done generating core. seed = %i (%i)'
                  % (seed + t, miss_count.max()))
            sys.stdout.flush()
            break
        else:
            print('%i/%i' % (miss_count.shape[0], num_stations), end=' ')
            sys.stdout.flush()
        if (x.shape[0] / float(num_stations)) < 0.7:
            raise RuntimeError('Exiting as far too few stations were generated '
                               '(%i / %i).'
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
    # Position of station clusters. (generate single arm and rotate to position
    # with stride 3)
    num_clusters = num_arms * core_arm_count
    t = numpy.arange(num_clusters) * delta_theta
    tmp = a * numpy.exp(b * t)
    cluster_x = tmp * numpy.cos(t)
    cluster_y = tmp * numpy.sin(t)
    for i in range(num_arms):
        cluster_x[i::3], cluster_y[i::3] = \
            rotate_coords(cluster_x[i::3], cluster_y[i::3],
                          math.degrees(arm_offsets[i]))
    # TODO(BM) shrink arm cluster radius with cluster telescope radius
    # Generate stations at the cluster positions.
    num_stations = num_clusters * stations_per_cluster
    arm_x = numpy.zeros(num_stations)
    arm_y = numpy.zeros(num_stations)

    arm_cluster_radius_m_inc = (arm_cluster_radius * 1.0) / (num_clusters - 1)

    for i in range(num_clusters):
        cluster_radius_m = arm_cluster_radius_m_inc * (num_clusters - 1 - i)
        cluster_radius_m += arm_cluster_radius
        cluster_radius_m = arm_cluster_radius
        print(i, cluster_radius_m)
        for t in range(num_tries):
            # x, y, _ = gridgen(stations_per_cluster, arm_cluster_radius * 2.0,
            #                   station_radius_m * 2.0, 30000)
            x, y, _ = gridgen(stations_per_cluster, cluster_radius_m * 2.0,
                              station_radius_m * 2.0, 30000)
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


def plot_layout(x_core, y_core, x_arm, y_arm, x_arm_outer, y_arm_outer,
                cx_arm, cy_arm, cx_outer, cy_outer, station_radius_m,
                inner_core_radius_m, core_radius_m, arm_cluster_radius,
                outer_arm_cluster_radius, out_dir):
    # Plotting
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')

    # Core --------------------------------
    circle = pyplot.Circle((0.0, 0.0),
                           core_radius_m, color='k', linestyle=':',
                           fill=False, alpha=0.5, linewidth=1.0)
    ax.add_artist(circle)
    # circle = pyplot.Circle((0.0, 0.0),
    #                        inner_core_radius_m, color='k',
    #                        fill=False, alpha=0.5, linewidth=1.0)
    # ax.add_artist(circle)
    for i in range(x_core.shape[0]):
        circle = pyplot.Circle((x_core[i], y_core[i]),
                               station_radius_m, color='r',
                               fill=True, alpha=0.4, linewidth=1.0)
        ax.add_artist(circle)

    # Inner arms --------------------------------
    arm_cluster_radius_m_inc = (arm_cluster_radius * 1.0) / (cx_arm.shape[0] - 1)
    for i in range(cx_arm.shape[0]):
        color = 'k'
        cluster_radius_m = arm_cluster_radius_m_inc * (cx_arm.shape[0] - 1 - i)
        cluster_radius_m += arm_cluster_radius
        cluster_radius_m = arm_cluster_radius
        circle = pyplot.Circle((cx_arm[i], cy_arm[i]),
                               cluster_radius_m, color=color, linestyle=':',
                               fill=False, alpha=0.5, linewidth=1.0)
        ax.add_artist(circle)
        # ax.text(cx_arm[i], cy_arm[i] + arm_cluster_radius*1.05, '%i' % i,
        #         ha='center', va='bottom', color='k', fontsize='small')
    for i in range(x_arm.shape[0]):
        if i < 12 * 6:
            color = 'g'
        else:
            color = 'r'
        circle = pyplot.Circle((x_arm[i], y_arm[i]),
                               station_radius_m, color=color,
                               fill=True, alpha=0.4, linewidth=1.0)
        ax.add_artist(circle)
    circle = pyplot.Circle((0, 0), 1700.0,
                           color='k',
                           fill=False, alpha=0.5, linewidth=1.0,
                           linestyle=':')
    ax.add_artist(circle)

    # Outer arms --------------------------------
    for i in range(cx_outer.shape[0]):
        circle = pyplot.Circle((cx_outer[i], cy_outer[i]),
                               outer_arm_cluster_radius, color='k',
                               linestyle=':', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)

    for i in range(x_arm_outer.shape[0]):
        circle = pyplot.Circle((x_arm_outer[i], y_arm_outer[i]),
                               station_radius_m, color='b',
                               fill=True, alpha=0.6, linewidth=1.0)
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
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    pyplot.savefig(join(out_dir, 'layout_02.0km.png'))
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    pyplot.savefig(join(out_dir, 'layout_02.0km.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'layout_03.0km.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'layout_05.0km.png'))
    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-10000, 10000)
    pyplot.savefig(join(out_dir, 'layout_10.0km.png'))
    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    pyplot.savefig(join(out_dir, 'layout_15.0km.png'))
    ax.set_xlim(-50000, 50000)
    ax.set_ylim(-50000, 50000)
    pyplot.savefig(join(out_dir, 'layout_50.0km.png'))
    pyplot.close(fig)


def main():
    """
    Core: 224 stations
    Inner arms: 72 stations (4 clusters per arm, 3 arms, 6 stations per cluster)
    Outer arms: 216 stations (12 clusters per arm, 3 arms, 6 stations per cluster)

    TODO(BM)
        - tweak core profile / radius
        - Arm clusters patch size shrink with radius (larger clusters near
          the core)
        - 35m or 45m stations?
        - Copy script to the telescope directory.
        - Add jitter to inner arm rotations?
        - Work out arm paramters a, b, delta_theta based on outer and inner
          arm station radii
    """

    # Common -----------------------------------------------------------------
    station_radius_m = 40.0 / 2.0
    stations_per_cluster = 6

    # Core --------------------------------------------------------------------
    seed = numpy.random.randint(1, 20000)
    seed = 19335
    print('seed0 = %i' % seed)
    num_tries = 2
    num_core_stations = 224
    core_radius_m = 500.0
    inner_core_radius_m = 270.0
    sll = -22

    # Core arms ---------------------------------------------------------------
    arm_cluster_radius = 120.0
    num_arms = 3
    core_arm_count = 4
    # a = core_radius_m + arm_cluster_radius * 1.6
    a = core_radius_m + 80.0 * 1.6
    b = 0.513
    delta_theta = math.radians(9.5)
    # delta_theta = math.radians(2.0)
    arm_offsets = numpy.arange(3) * 0.0 - 3.0
    arm_offsets = numpy.radians(arm_offsets)

    # Outer arms --------------------------------------------------------------
    outer_arm_count = 12
    num_clusters_outer = outer_arm_count * num_arms
    v4a_ss_enu_file = 'v7ska1lowN1v2rev3R.enu.94x4.fixed.txt'
    outer_arm_cluster_radius = 80.0

    out_dir = 'v5b_v4_TEMP.tm'
    # =========================================================================

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Generate core stations
    x_core, y_core, weights, r_weights = \
        generate_random_core(num_core_stations, core_radius_m,
                             inner_core_radius_m, sll, station_radius_m,
                             num_tries, seed)

    # Core arms (single spiral, split between arms)
    x_arm, y_arm, cx_arm, cy_arm = \
        generate_core_arms(num_arms, core_arm_count,
                           stations_per_cluster,
                           arm_cluster_radius, station_radius_m,
                           a, b, delta_theta, arm_offsets, num_tries)

    # Outer stations.
    x_arm_outer, y_arm_outer, cx_outer, cy_outer = \
        generate_outer_arms(v4a_ss_enu_file, num_clusters_outer,
                            stations_per_cluster,
                            outer_arm_cluster_radius, station_radius_m,
                            num_tries)

    st_x = numpy.hstack((x_core, x_arm, x_arm_outer))
    st_y = numpy.hstack((y_core, y_arm, y_arm_outer))
    num_stations = st_x.shape[0]
    v4d_st_enu = numpy.zeros((num_stations, 3))
    v4d_st_enu[:, 0] = st_x
    v4d_st_enu[:, 1] = st_y
    numpy.savetxt(join(out_dir, 'layout_enu_stations.txt'), v4d_st_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    # Plotting
    plot_layout(x_core, y_core, x_arm, y_arm,
                x_arm_outer, y_arm_outer, cx_arm, cy_arm,
                cx_outer, cy_outer, station_radius_m,
                inner_core_radius_m, core_radius_m, arm_cluster_radius,
                outer_arm_cluster_radius, out_dir)
    # plot_core_thinning_profile(r_weights, weights, core_radius_m,
    #                            inner_core_radius_m, out_dir)


if __name__ == '__main__':
    main()
