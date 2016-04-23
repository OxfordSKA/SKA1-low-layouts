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


def rotate_coords(x, y, angle):
    """Rotate array of x, y coordinates counter clockwise by angle, in deg."""
    xr = x * numpy.cos(radians(angle)) - y * numpy.sin(radians(angle))
    yr = x * numpy.sin(radians(angle)) + y * numpy.cos(radians(angle))
    return xr, yr


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
    num_super_stations = 85
    num_stations_per_super_station = 6
    max_tries_per_station = 5
    diameter_gridgen = 40.0  # m
    diameter = 35.0  # m
    antenna_diameter = 1.5
    num_ant_station_gridgen = 300
    num_ant_station = 256
    ss_diameter = 100.0
    st_diameter = diameter
    angles = numpy.arange(num_stations_per_super_station - 1) * \
             (360.0 / float(num_stations_per_super_station - 1))
    angles += 90.0
    r0 = diameter
    sx = r0 * numpy.cos(numpy.radians(angles))
    sy = r0 * numpy.sin(numpy.radians(angles))
    sx = numpy.insert(sx, 0, 0.0)
    sy = numpy.insert(sy, 0, 0.0)
    ss_model_dir = 'v4d_r_90m_180ant_ss_uniform.tm'
    if os.path.isdir(ss_model_dir):
        shutil.rmtree(ss_model_dir)
    os.makedirs(ss_model_dir)
    st_model_dir = 'v4d_r_90m_180ant_st_uniform.tm'
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
                ax, ay, _ = gridgen(num_ant_station_gridgen, diameter_gridgen,
                                    antenna_diameter, max_trials=10000)
                if ax.shape[0] == num_ant_station_gridgen:
                    ar = (ax**2 + ay**2)**0.5
                    # Sort by radius
                    sort_idx = ar.argsort()
                    ax = ax[sort_idx]
                    ay = ay[sort_idx]
                    ax = ax[:num_ant_station]
                    ay = ay[:num_ant_station]
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
                      '%i/%i' % (ax.shape[0], num_ant_station_gridgen))
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
            # Plot station and add to station superposition
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
