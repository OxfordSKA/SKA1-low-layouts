# -*- coding: utf-8 -*-
"""Python script for generating SKA LFAA V4A layout files.

Based on super-mongo scripts by Robert Braun.

email: benjamin.mort@oerc.ox.ac.uk
Versions:
    [18/02/2016]: Creation
    [18/02/2016]: Added function to generate OSKAR telescope models.
    [19/02/2016]: Minor improvements to documentation.
                  Added option to align sub-station latices.
"""

from __future__ import print_function
import numpy
import matplotlib.pyplot as pyplot
from math import pi, cos, sin, radians, sqrt, degrees, ceil
import os
from os.path import join
import shutil
from gridgen_no_taper import gridgen_no_taper


def rotate_coords(x, y, angle):
    """Rotate array of x, y coordinates counter clockwise by angle, in deg."""
    xr = x * cos(radians(angle)) - y * sin(radians(angle))
    yr = x * sin(radians(angle)) + y * cos(radians(angle))
    return xr, yr


def generate_lattice(n, inc, lattice_angle):
    """Generate a zero centred hexagonal lattice, rotated with the specified
    lattice angle

    Args:
        n (int): lattice size (n by n)
        inc (float): lattice separation in the x-dimension
        lattice_angle (float): Lattice rotation angle, in degrees.

    Returns:
        Lattice coordinates.
    """
    xc = numpy.zeros((n, n))
    yc = numpy.zeros((n, n))
    for iy in range(n):
        for ix in range(n):
            xc[iy, ix] = (ix + 0.5 * iy) * inc
            yc[iy, ix] = (iy * inc * sqrt(3.0) / 2.0)
    xc -= xc[n/2, n/2]
    yc -= yc[n/2, n/2]
    xc, yc = rotate_coords(xc, yc, lattice_angle)
    return xc.flatten(), yc.flatten()


def select_antennas(x, y, r_max, num_antennas, angle):
    """Select a number of antennas from points x, y, inside a pentagonal
    area defined by radius r_max, and angle"""
    num_samples = 1000
    inc_x = ((2.0 * pi) / 5.0) * num_samples
    theta = numpy.arctan2(y, x) + 2.0 * pi - radians(angle)
    theta_x = theta * num_samples
    theta_m5 = (theta_x % inc_x) / num_samples
    rad_p = r_max * (cos(pi / 5.0) / numpy.cos(theta_m5 - pi / 5.0))
    rc = (x**2 + y**2)**0.5
    s_good = rc / rad_p
    sort_idx = numpy.argsort(s_good)
    s_good = s_good[sort_idx]
    num_ok = numpy.argmax(s_good > 1.0) - 1
    assert num_antennas <= num_ok, '%i / %i' % (num_ok, num_antennas)
    if num_antennas < 0:
        num_antennas = num_ok
    x1 = x[sort_idx]
    y1 = y[sort_idx]
    x2 = x1[:num_antennas]
    y2 = y1[:num_antennas]
    return x2, y2


def generate_super_station(super_station_angle, lattice_angles,
                           align_sub_station_lattices=False):
    """Generate a v4a super-station layout with a specified super-station
       rotation angle and lattice angles

    Args:
        super_station_angle (float): super station rotation angle, in degrees
        lattice_angles (array_like): lattice angles in degrees wrt east,
                                     in degrees. Array of dimensions
                                     (num_stations = 6, num_sub_stations = 6)

    Returns:
        Arrays of x and y antenna coordinates with dimensions
         (station = 6, sub-station = 6, antenna = 48)
        Arrays of x and y centres of sub-substations with dimensions
         (station = 6, sub-station = 6)
    """
    num_stations = 6
    num_sub_stations = 6
    num_antennas = 24
    # align_sub_station_lattices = True
    lattice_angles = numpy.asarray(lattice_angles)
    # Work out some geometry for the sub-station and station pentagon
    sub_station_radius_m = 7.0
    # Sub-station circular radius
    r_max = sub_station_radius_m
    # Sub-station pentagon side length
    side = 2.0 * r_max * sin(radians(36.0))
    # Shortest distance to edge of sub-station pentagon
    r_min = r_max * cos(radians(36.0))
    # Sub-station separation
    p_sep = 2.0 * r_min + 0.5
    # Station separation
    s_sep = 2.0 * (r_max + side * cos(radians(18.0))) + 1.5
    # Lattice spacing / antenna diameter
    lattice_x_inc = 1.5
    lattice_y_inc = lattice_x_inc * sqrt(3.0) / 2.0
    lattice_size = int(ceil((3.0 * r_max) / lattice_x_inc))

    # Positions and orientation of sub-stations
    sub_station_angle = numpy.arange(num_sub_stations - 1) * \
                        (360.0 / (num_sub_stations - 1)) - 90.0
    sub_station_angle = numpy.insert(sub_station_angle, 0, 90.0)
    sub_station_angle = numpy.radians(sub_station_angle)
    x0 = numpy.zeros(num_sub_stations)
    y0 = numpy.zeros(num_sub_stations)
    for i in range(1, num_sub_stations):
        x0[i] = p_sep * cos(sub_station_angle[i])
        y0[i] = p_sep * sin(sub_station_angle[i])

    # Positions and orientation of stations.
    station_angles = numpy.arange(num_stations - 1) * \
                     (360.0 / (num_stations - 1)) + 90.0
    station_angles = numpy.insert(station_angles, 0, -90.0)
    station_angles = numpy.radians(station_angles)
    sx0 = numpy.zeros(num_stations)
    sy0 = numpy.zeros(num_stations)
    for i in range(1, num_stations):
        sx0[i] = s_sep * cos(station_angles[i])
        sy0[i] = s_sep * sin(station_angles[i])

    ant_x = numpy.zeros((num_stations, num_sub_stations, num_antennas))
    ant_y = numpy.zeros_like(ant_x)
    centre_x = numpy.zeros((num_stations, num_sub_stations))
    centre_y = numpy.zeros((num_stations, num_sub_stations))

    for j in range(num_stations):
        for i in range(num_sub_stations):

            # Obtain the coordinates of the sub-station centre
            cx, cy = rotate_coords(x0[i], y0[i],
                                   degrees(station_angles[j]) + 90.0)
            cx += sx0[j]
            cy += sy0[j]
            cx, cy = rotate_coords(cx, cy, super_station_angle)

            # Generate hexagonal lattice
            # xc, yc = generate_lattice(lattice_size, lattice_x_inc,
            #                           lattice_angles[j, i])

            # Adjust lattice center to be a multiple of the lattice spacing.
            # If enabled results in slightly different sub-station
            if align_sub_station_lattices:
                xc -= cx - round(cx / lattice_x_inc) * lattice_x_inc
                yc -= cy - round(cy / lattice_y_inc) * lattice_y_inc

            # TODO-BM change to fit as many points as possible after x trials...
            xc, yc, _ = gridgen_no_taper(500, 20, 1.5, 5000)
            print(j, i, len(xc))

            # Select points inside pentagon
            angle = degrees(sub_station_angle[i] + station_angles[j])
            angle += super_station_angle + 18.0
            xc, yc = select_antennas(xc, yc, r_max, num_antennas, angle)

            centre_x[j, i] = cx
            centre_y[j, i] = cy
            ant_x[j, i, :] = xc + centre_x[j, i]
            ant_y[j, i, :] = yc + centre_y[j, i]

    return ant_x, ant_y, centre_x, centre_y


def plot_super_station():
    num_stations = 6
    num_sub_stations = 6

    super_station_angle = 20.0
    # lattice_angles = numpy.zeros((num_stations, num_sub_stations))
    lattice_angles = numpy.random.randint(0, 72, (num_stations, num_sub_stations))
    x, y, cx, cy = generate_super_station(super_station_angle, lattice_angles)
    # x, y have dimension (num_stations, num_sub_stations, num_antennas)

    x = x.flatten()
    y = y.flatten()

    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(x, y, 'k+')
    for i in range(x.shape[0]):
        circle = pyplot.Circle((x[i], y[i]), 0.75, color='k', fill=True,
                               alpha=0.3)
        ax.add_artist(circle)
    ax.plot(cx, cy, 'rx', ms=10, mew=1.5)
    ax.grid()
    lim = max(max(numpy.abs(ax.get_xlim())), max(numpy.abs(ax.get_ylim())))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    pyplot.show()


def plot_v4a_telescope():
    enu = numpy.loadtxt('../layouts/v7ska1lowN1v2arev3R.enu.564x4.txt')
    num_stations = enu.shape[0]
    num_super_stations = num_stations / 6
    east = enu[:, 1]
    north = enu[:, 2]

    dx = east[1::6] - east[::6]
    dy = north[1::6] - north[::6]
    angle = numpy.degrees(numpy.arctan2(dy, dx)) + 54.0

    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(num_super_stations):
        ax.arrow(east[i*6], north[i*6], dx[i], dy[i], fc='g', ec='g',
                 head_width=5.0, head_length=5.0)
        # lattice_angles = numpy.random.randint(0, 72, (6, 6))
        lattice_angles = -360.0 * numpy.random.random((6, 6)) + 360.0
        x, y, _, _ = generate_super_station(angle[i], lattice_angles)
        x += east[i*6]
        y += north[i*6]
        ax.plot(x.flatten(), y.flatten(), 'k+', alpha=0.2)
    ax.plot(east, north, 'k+')
    ax.plot(east[::6], north[::6], 'rx')
    ax.plot(east[1::6], north[1::6], 'bx')

    ax.grid()
    lim = max(max(numpy.abs(ax.get_xlim())), max(numpy.abs(ax.get_ylim())))
    # lim = 360
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    pyplot.savefig('v4a.png', dpi=300)
    # pyplot.savefig('core_zoom.eps')
    pyplot.show()


def generate_v4a_telescope_model():
    # FIXME-BM: also load super station layout to create top level layout
    enu = numpy.loadtxt('../layouts/v7ska1lowN1v2arev3R.enu.564x4.txt')
    num_stations = enu.shape[0]
    num_super_stations = num_stations / 6
    num_antennas = 24
    east = enu[:, 1]
    north = enu[:, 2]
    up = enu[:, 3]

    dx = east[1::6] - east[::6]
    dy = north[1::6] - north[::6]
    angle = numpy.degrees(numpy.arctan2(dy, dx)) + 54.0

    model_dir = 'v4a_super_station_random_positions.tm'
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    station_layout_enu = numpy.zeros((num_stations, 3))
    station_layout_enu[:, 0] = east
    station_layout_enu[:, 1] = north
    station_layout_enu[:, 2] = up
    station_file = join(model_dir, 'layout.txt')
    numpy.savetxt(station_file, station_layout_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    layout_enu = numpy.zeros((36*num_antennas, 2))
    for i in range(num_super_stations):
        print('=== Super station %i ===' % (i + 1))
        lattice_angles = -360.0 * numpy.random.random((6, 6)) + 360.0
        # lattice_angles = numpy.zeros((6, 6))
        x, y, cx, cy = generate_super_station(angle[i], lattice_angles,
                                              False)
        station_dir = join(model_dir, 'station%03i' % i)
        os.makedirs(station_dir)
        layout_enu[:, 0] = x.flatten()
        layout_enu[:, 1] = y.flatten()
        station_file = join(station_dir, 'layout.txt')
        numpy.savetxt(station_file, layout_enu, fmt='% -16.12f % -16.12f')

    print('plotting...')
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(num_stations / 6):
        layout_file = join(model_dir, 'station%03i' % i, 'layout.txt')
        layout = numpy.loadtxt(layout_file)
        ax.plot(layout[:, 0], layout[:, 1], 'k+', alpha=0.1)
    ax.grid()
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    pyplot.savefig(join(model_dir, 'all_stations.png'))
    # pyplot.show()

    for i in range(num_stations / 6):
        fig = pyplot.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect='equal')
        layout_file = join(model_dir, 'station%03i' % i, 'layout.txt')
        layout = numpy.loadtxt(layout_file)
        ax.plot(layout[:, 0], layout[:, 1], 'k+', alpha=1.0)
        ax.grid()
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        pyplot.savefig(join(model_dir, 'station_%03i.png' % i))
        pyplot.close()

        # pyplot.show()


if __name__ == '__main__':
    # plot_super_station()
    # plot_v4a_telescope()
    # FIXME-BM: Generate different model folders for each config
    # eg (random / aligned / rotated)
    generate_v4a_telescope_model()
