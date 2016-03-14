# -*- coding: utf-8 -*-
"""Script to generate v5d station coordinates.

Changes:
    04/03/2016: Initial version.
"""
from __future__ import print_function
import numpy
import matplotlib.pyplot as pyplot
import os
from os.path import join
from layout_utils import (gridgen_taylor_padded, gridgen,
                          rotate_coords)
import math


def generate_random_core(num_stations, core_radius_m, inner_core_radius_m,
                         sll, station_radius_m,
                         num_tries, seed):
    print('Generating %i core stations ...' % num_stations)
    for t in range(num_tries):
        numpy.random.seed(seed + t)
        x, y, miss_count, weights, r_weights = \
            gridgen_taylor_padded(num_stations, core_radius_m * 2.0,
                                  inner_core_radius_m * 2.0,
                                  station_radius_m * 2.0, sll, 50000)
        if x.shape[0] == num_stations:
            print('Done generating core. seed = %i (%i)'
                  % (seed + t, miss_count.max()))
            break
        else:
            print('%i/%i' % (miss_count.shape[0], num_stations), end=' ')
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
        t = numpy.arange(1, core_arm_count + 1) * delta_theta
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


def generate_core_arms_2(num_arms, core_arm_count, stations_per_cluster,
                         arm_cluster_radius, station_radius_m,
                         a, b, delta_theta, arm_offsets, num_tries):
    # Position of station clusters. (generate single arm and rotate to position
    # with stride 3)
    num_clusters = num_arms * core_arm_count
    t = numpy.arange(num_clusters) * delta_theta
    tmp = a * numpy.exp(b * t)
    cluster_x = tmp * numpy.cos(t)
    cluster_y = tmp * numpy.sin(t)
    # cluster_x, cluster_y = rotate_coords(cluster_x, cluster_y, 90.0)
    # cluster_x, cluster_y = rotate_coords(cluster_x, cluster_y, 90.0 + 120.0)
    # cluster_x, cluster_y = rotate_coords(cluster_x, cluster_y, 90.0 + 120.0 + 120.0)
    for i in range(num_arms):
        cluster_x[i::3], cluster_y[i::3] = \
            rotate_coords(cluster_x[i::3], cluster_y[i::3],
                          math.degrees(arm_offsets[i]))

    # Generate stations at the cluster positions.
    num_stations = num_clusters * stations_per_cluster
    arm_x = numpy.zeros(num_stations)
    arm_y = numpy.zeros(num_stations)
    for i in range(num_clusters):
        for t in range(num_tries):
            x, y, _ = gridgen(stations_per_cluster, arm_cluster_radius * 2.0,
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


def plot_layout(x_core, y_core, x_arm, y_arm, x_arm_2, y_arm_2,
                x_arm_outer, y_arm_outer,
                cx_arm, cy_arm, cx_arm_2, cy_arm_2, cx_outer,
                cy_outer, station_radius_m,
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
                               arm_cluster_radius, color='r',
                               fill=True, alpha=0.1, linewidth=0.0)
        ax.add_artist(circle)
    circle = pyplot.Circle((0, 0),
                           (cx_arm[-1]**2 + cy_arm[-1]**2)**0.5,
                           color='r',
                           fill=False, alpha=0.5, linewidth=2.0,
                           linestyle=':')
    ax.add_artist(circle)

    for i in range(cx_arm_2.shape[0]):
        circle = pyplot.Circle((cx_arm_2[i], cy_arm_2[i]),
                               arm_cluster_radius, color='k',
                               fill=False, alpha=0.5, linewidth=1.0)
        ax.add_artist(circle)
    circle = pyplot.Circle((0, 0),
                           (cx_arm_2[-1]**2 + cy_arm_2[-1]**2)**0.5,
                           color='g',
                           fill=False, alpha=0.3, linewidth=1.0,
                           linestyle='--')
    ax.add_artist(circle)

    for i in range(x_arm.shape[0]):
        circle = pyplot.Circle((x_arm_2[i], y_arm_2[i]),
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
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    pyplot.savefig(join(out_dir, 'layout_02.0km.png'))
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
    seed = numpy.random.randint(1, 20000)
    seed = 4407
    num_tries = 10
    num_core_stations = (1 + 5 + 11 + 17) * 6
    core_radius_m = 420.0
    inner_core_radius_m = 230.0
    station_radius_m = 35.0 / 2.0
    sll = -22
    # ====== Core arms
    num_arms = 3
    core_arm_count = 5
    stations_per_arm_cluster = 6
    arm_cluster_radius = 80.0
    a = 300.0
    b = 0.513
    delta_theta = math.radians(37.0)
    arm_offsets = numpy.radians([35.0, 155.0, 270.0])

    num_arms_2 = 3
    core_arm_count_2 = 5
    # a2 = core_radius_m + arm_cluster_radius
    a2 = 410.0 + 80.0
    b2 = 0.513
    delta_theta2 = math.radians(9.25)
    arm_offsets2 = numpy.radians([90.0, 210.0, 325.0])

    num_core_arm_stations = num_arms * core_arm_count * stations_per_arm_cluster
    # ====== Outer arms
    outer_arm_count = 12
    stations_per_outer_cluster = 6
    num_clusters_outer = outer_arm_count * num_arms
    v4a_ss_enu_file = 'v7ska1lowN1v2rev3R.enu.94x4.fixed.txt'
    outer_arm_cluster_radius = 80.0

    out_dir = 'v5d.tm'
    # =========================================================================
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Generate core stations
    x_core, y_core, weights, r_weights = \
        generate_random_core(num_core_stations, core_radius_m,
                             inner_core_radius_m, sll, station_radius_m,
                             num_tries, seed)

    # Core arms (original spiral arms)
    x_arm, y_arm, cx_arm, cy_arm = \
        generate_core_arms(num_arms, core_arm_count, stations_per_arm_cluster,
                           arm_cluster_radius, station_radius_m,
                           a, b, delta_theta, arm_offsets, num_tries)

    # Core arms (single spiral, split between arms)
    x_arm_2, y_arm_2, cx_arm_2, cy_arm_2 = \
        generate_core_arms_2(num_arms_2, core_arm_count_2,
                             stations_per_arm_cluster,
                             arm_cluster_radius, station_radius_m,
                             a2, b2, delta_theta2, arm_offsets2, num_tries)

    # Outer stations.
    x_arm_outer, y_arm_outer, cx_outer, cy_outer = \
        generate_outer_arms(v4a_ss_enu_file, num_clusters_outer,
                            stations_per_outer_cluster,
                            outer_arm_cluster_radius, station_radius_m,
                            num_tries)

    st_x = numpy.hstack((x_core, x_arm_2, x_arm_outer))
    st_y = numpy.hstack((y_core, y_arm_2, y_arm_outer))
    num_stations = st_x.shape[0]
    v4d_st_enu = numpy.zeros((num_stations, 3))
    v4d_st_enu[:, 0] = st_x
    v4d_st_enu[:, 1] = st_y
    numpy.savetxt(join(out_dir, 'layout_enu_stations.txt'), v4d_st_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    # Plotting
    plot_layout(x_core, y_core,
                x_arm, y_arm, x_arm_2, y_arm_2,
                x_arm_outer, y_arm_outer,
                cx_arm, cy_arm, cx_arm_2, cy_arm_2,
                cx_outer, cy_outer, station_radius_m,
                inner_core_radius_m, core_radius_m, arm_cluster_radius,
                outer_arm_cluster_radius, out_dir)
    plot_core_thinning_profile(r_weights, weights, core_radius_m,
                               inner_core_radius_m, out_dir)


if __name__ == '__main__':
    main()
