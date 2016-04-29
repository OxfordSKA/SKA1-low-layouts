"""Module to generate super-stations for trail v4d spec. layouts"""
# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib.pyplot as pyplot
import numpy
from numpy.random import rand
import shutil
import os
from os.path import join
from math import radians
from taper_function import taylor_win
from numpy.random import rand
from math import floor


def rotate_coords(x, y, angle):
    """Rotate array of x, y coordinates counter clockwise by angle, in deg."""
    xr = x * numpy.cos(radians(angle)) - y * numpy.sin(radians(angle))
    yr = x * numpy.sin(radians(angle)) + y * numpy.cos(radians(angle))
    return xr, yr


def gridgen_taylor(num_points, diameter, min_dist, sll=-28, n_miss_max=1000):
    """Generate uniform random positions within a specified diameter which
    are no closer than a specified minimum distance.

    Uses and algorithm where the area is split into a grid sectors
    so that when checking for minimum distance, only nearby points need to be
    considered.
    """
    def get_trail_position(r):
        x = -r + 2.0 * r * rand()
        y = -r + 2.0 * r * rand()
        return x, y

    def grid_position(x, y, scale, r):
        jx = int(floor((x + r) * scale))
        jy = int(floor((y + r) * scale))
        return jx, jy

    # Fix seed to study closest match fails (with fixed seed can
    # print problematic indices)
    # seed(2)

    r = diameter / 2.0  # Radius

    # Initialise taylor taper.
    nbar = int(numpy.ceil(2.0 * (numpy.arccosh(10**(-sll / 20.0)) /
                                 numpy.pi)**2 + 0.5))
    n_taylor = 10000
    w_taylor = taylor_win(n_taylor + 1, nbar, sll)
    w_taylor /= w_taylor.max()
    w_taylor = w_taylor[n_taylor/2:]
    r_taylor = numpy.arange(w_taylor.shape[0]) * (diameter / (n_taylor + 1))
    n_taylor = w_taylor.shape[0]

    p = 1.0 / w_taylor[-1]
    max_dist = p * min_dist

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / max_dist)))
    grid_size += grid_size % 2
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 1
    # print('- Station d: %f' % diameter)
    # print('- Grid size: %i' % grid_size)
    # print('- Min dist: %f' % min_dist)
    # print('- Max dist: %f' % max_dist)
    # print('- Grid cell: %f' % grid_cell)

    # Pre-allocate coordinate arrays
    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    # Grid meta-data
    # First index in the grid
    grid_i_start = numpy.zeros((grid_size, grid_size), dtype='i8')
    # Last index in the grid
    grid_i_end = numpy.zeros((grid_size, grid_size), dtype='i8')
    # Points in grid cell.
    grid_count = numpy.zeros((grid_size, grid_size), dtype='i8')
    # Next coordinate index.
    grid_next = numpy.zeros(num_points, dtype='i8')

    n = num_points
    n_req = num_points
    num_miss = 0
    max_num_miss = 0
    miss_count = []
    j = 0
    space_remaining = True
    while space_remaining:
        done = False
        while not done:
            # Generate a trail position
            xt, yt = get_trail_position(r)
            rt = (xt**2 + yt**2)**0.5

            # Check if the point is inside the diameter.
            if rt + min_dist / 2.0 > r:
                num_miss += 1

            # Check if min distance is met.
            else:
                iw = int(round((rt / r) * n_taylor))
                ant_r = min_dist / (2.0 * w_taylor[iw])

                jx, jy = grid_position(xt, yt, scale, r)
                y0 = max(0, jy - check_width)
                y1 = min(grid_size, jy + check_width + 1)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size, jx + check_width + 1)
                dmin = diameter  # Set initial min to diameter.
                for ky in range(y0, y1):
                    for kx in range(x0, x1):
                        if grid_count[kx, ky] > 0:
                            i_other = grid_i_start[kx, ky]
                            for num_other in range(grid_count[kx, ky]):
                                dx = xt - x[i_other]
                                dy = yt - y[i_other]
                                dr = (dx**2 + dy**2)**0.5
                                r_other = (x[i_other]**2 + y[i_other]**2)**0.5
                                iw = int(round(r_other / r * n_taylor))
                                ant_r_other = min_dist / (2.0 * w_taylor[iw])

                                if dr - ant_r_other <= dmin:
                                    dmin = dr - ant_r_other
                                i_other = grid_next[i_other]

                iw = int(round(rt / r * n_taylor))
                scaled_min_dist_3 = (min_dist / 2.0) / w_taylor[iw]

                if dmin >= scaled_min_dist_3:
                    x[j] = xt
                    y[j] = yt

                    if grid_count[jx, jy] == 0:
                        grid_i_start[jx, jy] = j
                    else:
                        grid_next[grid_i_end[jx, jy]] = j
                    grid_i_end[jx, jy] = j
                    grid_count[jx, jy] += 1
                    miss_count.append(num_miss)
                    max_num_miss = max(max_num_miss, num_miss)
                    num_miss = 0
                    done = True
                    j += 1
                else:
                    num_miss += 1

            if num_miss >= n_miss_max:
                n = j - 1
                done = True

        if num_miss >= n_miss_max or j >= num_points:
            max_num_miss = max(max_num_miss, num_miss)
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    return x, y, miss_count, w_taylor, r_taylor, n_taylor


def gridgen(num_points, diameter, min_dist, max_trials=1000):
    def grid_position(x, y, scale, grid_size):
        jx = int(round(x * scale)) + grid_size / 2
        jy = int(round(y * scale)) + grid_size / 2
        return jx, jy

    def get_trail_position(r):
        x = -r + 2.0 * r * rand()
        y = -r + 2.0 * r * rand()
        return x, y

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / min_dist)))
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 1

    r = diameter / 2.0  # Radius
    r_sq = r**2  # Radius, squared
    min_dist_sq = min_dist**2  # minimum distance, squared
    r_ant = min_dist / 2.0

    # Pre-allocate coordinate arrays
    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    # Grid meta-data
    grid_i_start = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_i_end = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_count = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_i_next = numpy.zeros(num_points, dtype='i8')

    n = num_points
    n_req = num_points
    num_tries = 0
    try_count = list()
    for j in range(n_req):

        done = False
        while not done:

            # Generate a trail position
            xt, yt = get_trail_position(r)
            rt = (xt**2 + yt**2)**0.5

            # Check if the point is inside the diameter.
            if rt + r_ant > r:
                num_tries += 1

            # Check if min distance is met.
            else:
                jx, jy = grid_position(xt, yt, scale, grid_size)
                y0 = max(0, jy - check_width)
                y1 = min(grid_size, jy + check_width + 1)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size, jx + check_width + 1)
                d_min = diameter  # Set initial min to diameter.
                for ky in range(y0, y1):
                    for kx in range(x0, x1):
                        if grid_count[kx, ky] > 0:
                            kh1 = grid_i_start[kx, ky]
                            for kh in range(grid_count[kx, ky]):
                                dx = xt - x[kh1]
                                dy = yt - y[kh1]
                                d_min = min((dx**2 + dy**2)**0.5, d_min)
                                kh1 = grid_i_next[kh1]

                if d_min >= min_dist:
                    x[j] = xt
                    y[j] = yt
                    if grid_count[jx, jy] == 0:
                        grid_i_start[jx, jy] = j
                    else:
                        grid_i_next[grid_i_end[jx, jy]] = j
                    grid_i_end[jx, jy] = j
                    grid_count[jx, jy] += 1
                    try_count.append(num_tries)
                    num_tries = 0
                    done = True
                else:
                    num_tries += 1

            if num_tries >= max_trials:
                n = j - 1
                done = True

        if num_tries >= max_trials:
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    return x, y, try_count


def gen_super_stations():
    """Generation 85 super-stations by rotation"""
    # =========================================================================
    sll = -20
    num_super_stations = 85
    num_stations_per_super_station = 6
    num_tries = 10000  # per grid
    max_tries_per_station = 5
    diameter = 30.0  # m
    antenna_diameter = 1.5
    num_ant_station = 95
    ss_diameter = 90.0
    st_diameter = diameter
    angles = numpy.arange(num_stations_per_super_station - 1) * \
             (360.0 / float(num_stations_per_super_station - 1))
    angles += 90.0
    r0 = diameter + 1.0
    sx = r0 * numpy.cos(numpy.radians(angles))
    sy = r0 * numpy.sin(numpy.radians(angles))
    sx = numpy.insert(sx, 0, 0.0)
    sy = numpy.insert(sy, 0, 0.0)
    ss_model_dir = 'v4d_r_90m_%iant_ss_taylor-20.tm' % num_ant_station
    if os.path.isdir(ss_model_dir):
        shutil.rmtree(ss_model_dir)
    os.makedirs(ss_model_dir)
    st_model_dir = 'v4d_r_90m_%iant_st_taylor-20.tm' % num_ant_station
    if os.path.isdir(st_model_dir):
        shutil.rmtree(st_model_dir)
    os.makedirs(st_model_dir)
    ss_angles = -360.0 * numpy.random.random(num_super_stations) + 360.0

    # =========================================================================

    ss_ant_x = numpy.zeros((num_stations_per_super_station, num_ant_station))
    ss_ant_y = numpy.zeros_like(ss_ant_x)
    st_ant_x = numpy.zeros((num_stations_per_super_station, num_ant_station))
    st_ant_y = numpy.zeros_like(st_ant_x)
    ss_enu = numpy.zeros((num_ant_station * num_stations_per_super_station, 2))
    st_enu = numpy.zeros((num_ant_station, 2))

    # =========================================================================
    circle = pyplot.Circle((0.0, 0.0), ss_diameter / 2.0,
                           color='r', linestyle='--',
                           fill=False, alpha=0.3, lw=2.0)

    fig1 = pyplot.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.grid()
    ax1.set_xlim(-60, 60)
    ax1.set_ylim(-60, 60)
    line1, = ax1.plot([], [], 'k+')
    label1 = ax1.text(0.02, 0.98, '', ha='left', va='top', style='italic',
                      color='k', transform=ax1.transAxes, fontsize='x-small')
    circle = pyplot.Circle((0.0, 0.0), ss_diameter / 2.0,
                           color='r', linestyle='--',
                           fill=False, alpha=0.3, lw=2.0)
    ax1.add_artist(circle)

    fig2 = pyplot.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.grid()
    ax2.set_xlim(-60, 60)
    ax2.set_ylim(-60, 60)
    circle = pyplot.Circle((0.0, 0.0), ss_diameter / 2.0,
                           color='r', linestyle='--',
                           fill=False, alpha=0.3, lw=2.0)
    ax2.add_artist(circle)

    fig3 = pyplot.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(111, aspect='equal')
    ax3.set_xlabel('East [m]')
    ax3.set_ylabel('North [m]')
    ax3.grid()
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(-20, 20)
    line3, = ax3.plot([], [], 'k+')
    label3 = ax3.text(0.02, 0.98, '', ha='left', va='top', style='italic',
                      color='k', transform=ax3.transAxes, fontsize='x-small')
    circle = pyplot.Circle((0.0, 0.0), st_diameter / 2.0,
                           color='r', linestyle='--',
                           fill=False, alpha=0.3, lw=2.0)
    ax3.add_artist(circle)

    fig4 = pyplot.figure(figsize=(8, 8))
    ax4 = fig4.add_subplot(111, aspect='equal')
    ax4.set_xlabel('East [m]')
    ax4.set_ylabel('North [m]')
    ax4.grid()
    ax4.set_xlim(-20, 20)
    ax4.set_ylim(-20, 20)
    circle = pyplot.Circle((0.0, 0.0), st_diameter / 2.0,
                           color='r', linestyle='--',
                           fill=False, alpha=0.3, lw=2.0)
    ax4.add_artist(circle)
    # =========================================================================

    for i in range(num_super_stations):
        print('== super station %i == : ' % i, end='')
        for j in range(num_stations_per_super_station):
            print('%i' % j, end='')
            trial = 0
            while trial < max_tries_per_station:
                print('.', end='')
                ax, ay, mc, _, _, _ = gridgen_taylor(num_ant_station, diameter,
                                                     antenna_diameter, sll,
                                                     num_tries)
                if ax.shape[0] == num_ant_station:
                    ss_ant_x[j, :] = ax + sx[j]
                    ss_ant_y[j, :] = ay + sy[j]
                    st_ant_x[j, :] = ax
                    st_ant_y[j, :] = ay
                    break
                else:
                    trial += 1
                    continue
            if trial == max_tries_per_station:
                print()
                print('Error, Failed to find enough antennas for station '
                      '%i/%i' % (ax.shape[0], num_ant_station))
                return
        print()
        # Rotate super-station
        ss_ant_x, ss_ant_y = rotate_coords(ss_ant_x, ss_ant_y, ss_angles[i])

        # Write station and super-station folders
        station_dir = 'station%03i' % i
        os.makedirs(join(ss_model_dir, station_dir))
        ss_enu[:, 0] = ss_ant_x.flatten()
        ss_enu[:, 1] = ss_ant_y.flatten()
        station_file = join(ss_model_dir, station_dir, 'layout.txt')
        numpy.savetxt(station_file, ss_enu, fmt='% -16.12f % -16.12f')
        line1.set_data(ss_enu[:, 0], ss_enu[:, 1])
        label1.set_text('super station %03i' % i)
        fig1.savefig(join(ss_model_dir, 'station_%03i.png' % i))
        ax2.plot(ss_enu[:, 0], ss_enu[:, 1], 'k+', alpha=0.1)

        # Write station folders
        for j in range(num_stations_per_super_station):
            station_id = i * num_stations_per_super_station + j
            station_dir = 'station%03i' % station_id
            os.makedirs(join(st_model_dir, station_dir))
            st_enu[:, 0] = st_ant_x[j, :].flatten()
            st_enu[:, 1] = st_ant_y[j, :].flatten()
            station_file = join(st_model_dir, station_dir, 'layout.txt')
            numpy.savetxt(station_file, st_enu, fmt='% -16.12f % -16.12f')
            # TODO-BM plot station and add to station superposition
            line3.set_data(st_enu[:, 0], st_enu[:, 1])
            label3.set_text('station %03i' % station_id)
            fig3.savefig(join(st_model_dir, 'station_%03i.png' % station_id))
            ax4.plot(st_enu[:, 0], st_enu[:, 1], 'k+', alpha=0.1)

    fig2.savefig(join(ss_model_dir, 'all_stations.png'))
    fig4.savefig(join(st_model_dir, 'all_stations.png'))

    ss_layout = numpy.zeros((num_super_stations, 3))
    numpy.savetxt(join(ss_model_dir, 'layout.txt'), ss_layout,
                  fmt='%3.1f %3.1f %3.1f')
    total_stations = num_super_stations * num_stations_per_super_station
    st_layout = numpy.zeros((total_stations, 3))
    numpy.savetxt(join(st_model_dir, 'layout.txt'), st_layout,
                  fmt='%3.1f %3.1f %3.1f')


if __name__ == '__main__':
    gen_super_stations()
