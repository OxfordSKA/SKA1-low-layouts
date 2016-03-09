# -*- coding: utf-8 -*-
"""Script to generate v5a station and super-station coordinates.

Changes:
    01/03/2016: Initial version.
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
import layout_utils


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



def rotate_coords(x, y, angle):
    """Rotate array of x, y coordinates counter clockwise by angle, in deg."""
    xr = x * numpy.cos(radians(angle)) - y * numpy.sin(radians(angle))
    yr = x * numpy.sin(radians(angle)) + y * numpy.cos(radians(angle))
    return xr, yr


def generate_baseline_uvw(x, y, z, ra_rad, dec_rad, num_times, num_baselines,
                          mjd_start, dt_s):
    """Generate baseline coordinates from ecef station coordinates."""
    num_coords = num_times * num_baselines
    uu = numpy.zeros(num_coords, dtype='f8')
    vv = numpy.zeros(num_coords, dtype='f8')
    ww = numpy.zeros(num_coords, dtype='f8')
    for i in range(num_times):
        t = i * dt_s + dt_s / 2.0
        mjd = mjd_start + (t / 86400.0)
        i0 = i * num_baselines
        i1 = i0 + num_baselines
        uu_, vv_, ww_ = evaluate_baseline_uvw(x, y, z, ra_rad, dec_rad, mjd)
        uu[i0:i1] = uu_
        vv[i0:i1] = vv_
        ww[i0:i1] = ww_
    return uu, vv, ww


def main():
    # ==========================================================================
    # Telescope element radii
    st_radius = 35.0 / 2.0  # Station-radius
    ss_radius = 3.0 * st_radius

    # Core ring super-stations
    core_ring_radius = 370.0  # metres
    num_stations_core_ring = 34 * 6
    num_tries = 5

    # Core arms
    num_arms = 3
    core_arm_count = 5  # Number of super-stations per core arm
    a = 300.0
    b = 0.513
    delta_theta = 37.0
    arm_offsets = [35.0, 155.0, 275.0]
    num_super_stations_arms = num_arms * core_arm_count
    ss_arm_petal_angle = -360.0 * random(num_super_stations_arms) + 360.0

    # Outer arms (same outer 3 * 12 = 36 stations as v4a)
    outer_arm_count = 12  # Number of super-stations per outer arm
    num_super_stations_outer = num_arms * outer_arm_count
    v4a_ss_enu_file = 'v7ska1lowN1v2rev3R.enu.94x4.fixed.txt'
    ss_petal_angle_outer = -360.0 * random(num_super_stations_outer) + 360.0

    # Stations
    num_stations_per_ss = 6

    out_dir = 'v5a_layout'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # ==========================================================================

    # == Super-stations

    # Generate core spiral arm super-stations
    v5a_ss_x_arms = numpy.zeros(num_super_stations_arms)
    v5a_ss_y_arms = numpy.zeros(num_super_stations_arms)
    for i in range(num_arms):
        t = numpy.arange(1, core_arm_count + 1) * delta_theta
        t = numpy.radians(t)
        x = a * numpy.exp(b * t) * numpy.cos(t + numpy.radians(arm_offsets[i]))
        y = a * numpy.exp(b * t) * numpy.sin(t + numpy.radians(arm_offsets[i]))
        i0 = i * core_arm_count
        i1 = i0 + core_arm_count
        v5a_ss_x_arms[i0:i1] = x
        v5a_ss_y_arms[i0:i1] = y

    # Load super-station outer spiral arms from the v4a config
    v4a_ss_enu = numpy.loadtxt(v4a_ss_enu_file)
    v4a_ss_enu = v4a_ss_enu[:, 1:]
    v4a_ss_r = (v4a_ss_enu[:, 0]**2 + v4a_ss_enu[:, 1]**2)**0.5
    sort_idx = numpy.argsort(v4a_ss_r)
    v4a_ss_enu = v4a_ss_enu[sort_idx[::-1], :]
    v5a_ss_x_outer = v4a_ss_enu[:num_super_stations_outer, 0]
    v5a_ss_y_outer = v4a_ss_enu[:num_super_stations_outer, 1]

    # == Stations
    # Generate core ring stations
    for t in range(num_tries):
        print('.')
        x, y, _ = gridgen(num_stations_core_ring, core_ring_radius * 2.0,
                          st_radius*2.0)
        if x.shape[0] == num_stations_core_ring:
            break
    if not x.shape[0] == num_stations_core_ring:
        raise RuntimeError('Failed to generate enough stations in the inner '
                           'core. %i / %i generated'
                           % (x.shape, num_stations_core_ring))
    v5a_st_x_rings = x
    v5a_st_y_rings = y


    # Generate core spiral arm stations
    v5a_st_x_arms = numpy.zeros((num_super_stations_arms,
                                 num_stations_per_ss))
    v5a_st_y_arms = numpy.zeros_like(v5a_st_x_arms)
    for i in range(num_super_stations_arms):
        angles = 360.0 / (num_stations_per_ss - 1) * \
                 numpy.arange(num_stations_per_ss - 1)
        x = (st_radius * 2.0) * numpy.cos(numpy.radians(angles))
        y = (st_radius * 2.0) * numpy.sin(numpy.radians(angles))
        x, y = rotate_coords(x, y, ss_arm_petal_angle[i])
        v5a_st_x_arms[i, 1:] = x
        v5a_st_y_arms[i, 1:] = y
        v5a_st_x_arms[i, :] += v5a_ss_x_arms[i]
        v5a_st_y_arms[i, :] += v5a_ss_y_arms[i]
    v5a_st_x_arms = v5a_st_x_arms.flatten()
    v5a_st_y_arms = v5a_st_y_arms.flatten()

    # Generate outer arm stations
    v5a_st_x_outer = numpy.zeros((num_super_stations_outer, num_stations_per_ss))
    v5a_st_y_outer = numpy.zeros((num_super_stations_outer, num_stations_per_ss))
    for i in range(num_super_stations_outer):
        angles = 360.0 / (num_stations_per_ss - 1) * \
                 numpy.arange(num_stations_per_ss - 1)
        x = (st_radius * 2.0) * numpy.cos(numpy.radians(angles))
        y = (st_radius * 2.0) * numpy.sin(numpy.radians(angles))
        x, y = rotate_coords(x, y, ss_petal_angle_outer[i])
        v5a_st_x_outer[i, 1:] = x
        v5a_st_y_outer[i, 1:] = y
        v5a_st_x_outer[i, :] += v5a_ss_x_outer[i]
        v5a_st_y_outer[i, :] += v5a_ss_y_outer[i]
    v5a_st_x_outer = v5a_st_x_outer.flatten()
    v5a_st_y_outer = v5a_st_y_outer.flatten()

    # Concatenate coords.
    v5a_st_x = numpy.hstack((v5a_st_x_rings, v5a_st_x_arms, v5a_st_x_outer))
    v5a_st_y = numpy.hstack((v5a_st_y_rings, v5a_st_y_arms, v5a_st_y_outer))
    # v5a_st_x = numpy.hstack((v5a_st_x_rings, v5a_st_x_arms))
    # v5a_st_y = numpy.hstack((v5a_st_y_rings, v5a_st_y_arms))
    # v5a_st_x = v5a_st_x_rings
    # v5a_st_y = v5a_st_y_rings

    # === Generate layouts ==============================
    num_stations = v5a_st_x.shape[0]
    v5a_st_enu = numpy.zeros((num_stations, 3))
    v5a_st_enu[:, 0] = v5a_st_x
    v5a_st_enu[:, 1] = v5a_st_y
    numpy.savetxt(join(out_dir, 'v5a_stations_enu.txt'), v5a_st_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    # ==== Plotting ===========================================================
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')

    circle = pyplot.Circle((0.0, 0.0),
                           core_ring_radius, color='k',
                           fill=False, alpha=0.5, linewidth=1.0)
    ax.add_artist(circle)

    arm_colors = ['y', 'g', 'r']
    for i in range(num_super_stations_arms):
        q = int(i / core_arm_count)
        circle = pyplot.Circle((v5a_ss_x_arms[i], v5a_ss_y_arms[i]),
                               ss_radius, color=arm_colors[q],
                               fill=True, alpha=0.5, linewidth=0.0)
        ax.add_artist(circle)

    for q in range(num_arms):
        i0 = q * outer_arm_count
        i1 = i0 + outer_arm_count
        for i in range(i0, i1):
            circle = pyplot.Circle((v5a_ss_x_outer[i], v5a_ss_y_outer[i]),
                                   ss_radius, color='c', fill=True, alpha=0.5)
            ax.add_artist(circle)

    # Plot station positions
    for i in range(v5a_st_x.shape[0]):
        circle = pyplot.Circle((v5a_st_x[i], v5a_st_y[i]),
                               st_radius, color='k', linewidth=1.0,
                               fill=True, alpha=0.2)
        ax.add_artist(circle)

    # circle = pyplot.Circle((0.0, 0.0), 1700.0, color='r', linestyle='--',
    #                        linewidth=1.0, fill=False, alpha=0.5)
    # ax.add_artist(circle)

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1.0)
    ax.set_ylabel('North [m]')
    ax.set_xlabel('East [m]')
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    pyplot.savefig(join(out_dir, 'v5a_station_layout_zoom_1.5km.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'v5a_station_layout_zoom_3.0km.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'v5a_station_layout_zoom_5.0km.png'))
    ax.set_xlim(-50000, 50000)
    ax.set_ylim(-50000, 50000)
    pyplot.savefig(join(out_dir, 'v5a_station_layout_50.0km.png'))
    pyplot.close(fig)

    if uvwsim_found:
        x = v5a_st_x
        y = v5a_st_y
        num_stations = x.shape[0]
        z = numpy.zeros_like(x)
        lon = radians(116.63128900)
        lat = radians(-26.69702400)
        alt = 0.0
        ra = radians(68.698903779331502)
        dec = radians(-26.568851215532160)
        mjd_start = 57443.4375000000
        dt_s = 0.0
        num_times = 1
        num_baselines = num_stations * (num_stations - 1) / 2
        x, y, z = convert_enu_to_ecef(x, y, z, lon, lat, alt)
        uu, vv, ww = generate_baseline_uvw(x, y, z, ra, dec, num_times,
                                           num_baselines, mjd_start,
                                           dt_s)

        layout_utils.plot_hist(uu, vv, join(out_dir, 'v5a_hist.png'),
                               'v5a snapshot-uv')
        layout_utils.plot_uv_dist(uu, vv,
                                  join(out_dir, 'v5a_snapshot_uv_zenith'))

if __name__ == '__main__':
    main()
