# -*- coding: utf-8 -*-
"""Script to generate v4d station and super-station coordinates.

Based on v4d description in the document:
    'Objectives for a Proposed SKA1-Low Station-Configuration Workshop'
    P. Dewdney, J. Wagg, R. Braun, M.-G. Labate
    Feb 1st 2016
which was emailed prior to the SKA1-Low Station-Configuration Workshop
held on the 25-26th Feb 2016

Changes:
    29/02/2016: Initial version.
"""
from __future__ import print_function

import os
from math import radians
from os.path import join

import matplotlib.pyplot as pyplot
import numpy
from numpy.random import random


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
    ss_radius = st_radius * 3.0  # Super-station radius

    # Core ring super-stations
    num_rings = 4
    ring_counts = [1, 5, 11, 17]
    ring_radii = [0.0, 100.0, 190.0, 290.0]  # metres
    num_super_stations_rings = numpy.array(ring_counts).sum()
    ring_start_angle = -360.0 * random(num_rings) + 360.0
    ss_ring_petal_angle = -360.0 * random(num_super_stations_rings) + 360.0

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

    out_dir = 'v4d.tm'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # ==========================================================================

    # == Super-stations

    # Generate core ring super-stations
    v4d_ss_x_rings = numpy.zeros(num_super_stations_rings)
    v4d_ss_y_rings = numpy.zeros(num_super_stations_rings)
    idx = 0
    for i, hist in enumerate(ring_counts):
        angles = numpy.arange(hist) * (360.0 / hist)
        angles += ring_start_angle[i]
        x = ring_radii[i] * numpy.cos(numpy.radians(angles))
        y = ring_radii[i] * numpy.sin(numpy.radians(angles))
        v4d_ss_x_rings[idx:idx+hist] = x
        v4d_ss_y_rings[idx:idx+hist] = y
        idx += hist

    # Generate core spiral arm super-stations
    v4d_ss_x_arms = numpy.zeros(num_super_stations_arms)
    v4d_ss_y_arms = numpy.zeros(num_super_stations_arms)
    for i in range(num_arms):
        t = numpy.arange(1, core_arm_count + 1) * delta_theta
        t = numpy.radians(t)
        x = a * numpy.exp(b * t) * numpy.cos(t + numpy.radians(arm_offsets[i]))
        y = a * numpy.exp(b * t) * numpy.sin(t + numpy.radians(arm_offsets[i]))
        i0 = i * core_arm_count
        i1 = i0 + core_arm_count
        v4d_ss_x_arms[i0:i1] = x
        v4d_ss_y_arms[i0:i1] = y

    # Load super-station outer spiral arms from the v4a config
    v4a_ss_enu = numpy.loadtxt(v4a_ss_enu_file)
    v4a_ss_enu = v4a_ss_enu[:, 1:]
    v4a_ss_r = (v4a_ss_enu[:, 0]**2 + v4a_ss_enu[:, 1]**2)**0.5
    sort_idx = numpy.argsort(v4a_ss_r)
    v4a_ss_enu = v4a_ss_enu[sort_idx[::-1], :]
    v4d_ss_x_outer = v4a_ss_enu[:num_super_stations_outer, 0]
    v4d_ss_y_outer = v4a_ss_enu[:num_super_stations_outer, 1]

    # == Stations

    # Generate core ring stations
    v4d_st_x_rings = numpy.zeros((num_super_stations_rings,
                                  num_stations_per_ss))
    v4d_st_y_rings = numpy.zeros_like(v4d_st_x_rings)
    for i in range(num_super_stations_rings):
        angles = 360.0 / (num_stations_per_ss - 1) * \
                 numpy.arange(num_stations_per_ss - 1)
        x = (st_radius * 2.0) * numpy.cos(numpy.radians(angles))
        y = (st_radius * 2.0) * numpy.sin(numpy.radians(angles))
        x, y = rotate_coords(x, y, ss_ring_petal_angle[i])
        v4d_st_x_rings[i, 1:] = x
        v4d_st_y_rings[i, 1:] = y
        v4d_st_x_rings[i, :] += v4d_ss_x_rings[i]
        v4d_st_y_rings[i, :] += v4d_ss_y_rings[i]
    v4d_st_x_rings = v4d_st_x_rings.flatten()
    v4d_st_y_rings = v4d_st_y_rings.flatten()

    # Generate core spiral arm stations
    v4d_st_x_arms = numpy.zeros((num_super_stations_arms,
                                 num_stations_per_ss))
    v4d_st_y_arms = numpy.zeros_like(v4d_st_x_arms)
    for i in range(num_super_stations_arms):
        angles = 360.0 / (num_stations_per_ss - 1) * \
                 numpy.arange(num_stations_per_ss - 1)
        x = (st_radius * 2.0) * numpy.cos(numpy.radians(angles))
        y = (st_radius * 2.0) * numpy.sin(numpy.radians(angles))
        x, y = rotate_coords(x, y, ss_arm_petal_angle[i])
        v4d_st_x_arms[i, 1:] = x
        v4d_st_y_arms[i, 1:] = y
        v4d_st_x_arms[i, :] += v4d_ss_x_arms[i]
        v4d_st_y_arms[i, :] += v4d_ss_y_arms[i]
    v4d_st_x_arms = v4d_st_x_arms.flatten()
    v4d_st_y_arms = v4d_st_y_arms.flatten()

    # Generate outer arm stations
    v4d_st_x_outer = numpy.zeros((num_super_stations_outer, num_stations_per_ss))
    v4d_st_y_outer = numpy.zeros((num_super_stations_outer, num_stations_per_ss))
    for i in range(num_super_stations_outer):
        angles = 360.0 / (num_stations_per_ss - 1) * \
                 numpy.arange(num_stations_per_ss - 1)
        x = (st_radius * 2.0) * numpy.cos(numpy.radians(angles))
        y = (st_radius * 2.0) * numpy.sin(numpy.radians(angles))
        x, y = rotate_coords(x, y, ss_petal_angle_outer[i])
        v4d_st_x_outer[i, 1:] = x
        v4d_st_y_outer[i, 1:] = y
        v4d_st_x_outer[i, :] += v4d_ss_x_outer[i]
        v4d_st_y_outer[i, :] += v4d_ss_y_outer[i]
    v4d_st_x_outer = v4d_st_x_outer.flatten()
    v4d_st_y_outer = v4d_st_y_outer.flatten()

    # Concatenate coords.
    v4d_ss_x = numpy.hstack((v4d_ss_x_rings, v4d_ss_x_arms, v4d_ss_x_outer))
    v4d_ss_y = numpy.hstack((v4d_ss_y_rings, v4d_ss_y_arms, v4d_ss_y_outer))
    v4d_st_x = numpy.hstack((v4d_st_x_rings, v4d_st_x_arms, v4d_st_x_outer))
    v4d_st_y = numpy.hstack((v4d_st_y_rings, v4d_st_y_arms, v4d_st_y_outer))

    # === Generate layouts ==============================
    num_stations = v4d_st_x.shape[0]
    v4d_st_enu = numpy.zeros((num_stations, 3))
    v4d_st_enu[:, 0] = v4d_st_x
    v4d_st_enu[:, 1] = v4d_st_y
    numpy.savetxt(join(out_dir, 'layout_enu_stations.txt'), v4d_st_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    num_super_stations = v4d_ss_x.shape[0]
    v4d_ss_enu = numpy.zeros((num_super_stations, 3))
    v4d_ss_enu[:, 0] = v4d_ss_x
    v4d_ss_enu[:, 1] = v4d_ss_y
    numpy.savetxt(join(out_dir, 'layout_enu_super_stations.txt'), v4d_ss_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    # ==== Plotting ===========================================================
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(num_super_stations_rings):
        circle = pyplot.Circle((v4d_ss_x_rings[i], v4d_ss_y_rings[i]),
                               ss_radius, color='b', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)

    arm_colors = ['y', 'g', 'r']
    for i in range(num_super_stations_arms):
        q = int(i / core_arm_count)
        circle = pyplot.Circle((v4d_ss_x_arms[i], v4d_ss_y_arms[i]),
                               ss_radius, color=arm_colors[q],
                               fill=True, alpha=0.5, linewidth=0.0)
        ax.add_artist(circle)

    for q in range(num_arms):
        i0 = q * outer_arm_count
        i1 = i0 + outer_arm_count
        for i in range(i0, i1):
            circle = pyplot.Circle((v4d_ss_x_outer[i], v4d_ss_y_outer[i]),
                                   ss_radius, color='c', fill=True, alpha=0.5)
            ax.add_artist(circle)

    # Plot station positions
    for i in range(v4d_st_x.shape[0]):
        circle = pyplot.Circle((v4d_st_x[i], v4d_st_y[i]),
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
    pyplot.savefig(join(out_dir, 'v4d_station_layout_zoom_1.5km.png'))
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    pyplot.savefig(join(out_dir, 'v4d_station_layout_zoom_2.0km.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'v4d_station_layout_zoom_3.0km.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'v4d_station_layout_zoom_5.0km.png'))
    ax.set_xlim(-50000, 50000)
    ax.set_ylim(-50000, 50000)
    pyplot.savefig(join(out_dir, 'v4d_station_layout_50.0km.png'))
    pyplot.close(fig)


if __name__ == '__main__':
    main()
