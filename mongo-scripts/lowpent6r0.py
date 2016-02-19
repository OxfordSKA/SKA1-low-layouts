# -*- coding: utf-8 -*-
"""
Script to generate ska1-low v4a super-station antenna layouts.
This is a port of super mongo script(s) written by Robert Braun.

benjamin.mort@oerc.ox.ac.uk [last update: 8/12/2015]
"""

import numpy
import matplotlib.pyplot as pyplot
from math import pi, cos, sin, radians, sqrt, atan2, degrees
from numpy.random import random


def generate_lattice(nx, ny, rmax, sqinc):
    xc = numpy.zeros((nx, ny))
    yc = numpy.zeros((nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            xc[ix, iy] = -1.5 * rmax + (ix + 0.5 * iy) * sqinc
            yc[ix, iy] = -rmax + (iy * sqinc * sqrt(3.0) / 2.0)
    return xc.flatten(), yc.flatten()


def rotate_values(x, y, angle):
    """Rotation by angle, +angle == counterclockwise rotation"""
    xr = x * cos(radians(angle)) - y * sin(radians(angle))
    yr = x * sin(radians(angle)) + y * cos(radians(angle))
    return xr, yr


def rotate_lattice(xc, yc, angle_deg):
    xcp = xc * cos(radians(angle_deg)) + yc * sin(radians(angle_deg))
    ycp = yc * cos(radians(angle_deg)) - xc * sin(radians(angle_deg))
    return xcp, ycp


def select_antennas(xcp, ycp, rmax, incx):
    rc = (xcp**2 + ycp**2)**0.5
    thetac = numpy.arctan2(ycp, xcp) + 2.0 * pi
    thetax = thetac * 1000.0
    thetam5 = (thetax % incx) / 1000.0
    radp = rmax * cos(pi / 5.0) / numpy.cos(thetam5 - pi / 5.0)
    sgood = rc / radp
    sind = numpy.argsort(sgood)
    xcp = xcp[sind]
    ycp = ycp[sind]
    return xcp[:48], ycp[:48]


def generate_ska_low_v4a_super_station(super_station_angle,
                                       lattice_orientation,
                                       sub_station_orientation,
                                       station_orientation):
    dalp = super_station_angle
    alpg = lattice_orientation
    angles = sub_station_orientation
    sangles = station_orientation

    var = numpy.arange(0.0, 0.999, 0.001)
    theta = var * 2.0 * pi
    incx = 2.0 * pi / 5.0 * 1000.
    thetax = theta * 1000.0
    thetam5 = (thetax % incx) / 1000.0

    # Work out some geometry for the sub-station pentagon
    rmax = 7.0  # maximum sub station radius, in m
    sqinc = 2.0 * rmax / 9.33
    side = 2.0 * rmax * cos(radians(54.0))
    rmin = rmax * sin(radians(54.0))
    psep = 2.0 * rmin + 0.5
    ssep = 2.0 * (rmax + side * cos(radians(18.0))) + 1.5

    # Positions and orientation of sub-stations
    x0 = numpy.zeros(6)
    y0 = numpy.zeros(6)
    alp0 = numpy.zeros(6)
    for i in range(6):
        alp0[i] = radians(angles[i]) - radians(dalp)
        x0[i] = psep * cos(-alp0[i])
        y0[i] = psep * sin(-alp0[i])
    x0[0] = 0.0
    y0[0] = 0.0

    # Positions and orientation of stations.
    salp0 = numpy.zeros(6)
    sx0 = numpy.zeros(6)
    sy0 = numpy.zeros(6)
    for i in range(6):
        salp0[i] = radians(sangles[i]) - radians(dalp)
        sx0[i] = ssep * cos(-salp0[i])
        sy0[i] = ssep * sin(-salp0[i])
    sx0[0] = 0.0
    sy0[0] = 0.0

    sbedist = numpy.zeros((6, 6, 48), dtype='f8')
    sbndist = numpy.zeros((6, 6, 48), dtype='f8')

    # Dimensions for hexagonal lattice generation.
    nx = 15
    ny = 15

    # Loop over stations
    for j in range(6):

        # Generate hexagonal lattice
        xc, yc = generate_lattice(nx, ny, rmax, sqinc)

        # Rotate the lattice by the lattice angle (for the station).
        # -ve angle as clockwise rotation.
        xcp, ycp = rotate_values(xc, yc, -alpg[j])

        # Select 48 antennas from the lattice for the sub-station.
        xcp, ycp = select_antennas(xcp, ycp, rmax, incx)

        # Loop over sub-stations
        for i in range(6):
            if j == 0:
                if i == 3:
                    # Clockwise rotation by alp0[i], negate y values.
                    x1, y1 = rotate_values(xcp, -ycp, degrees(-alp0[i]))
                    x1 += x0[i]
                    y1 += y0[i]
                else:
                    # Clockwise rotation by alp0[i]
                    x1, y1 = rotate_values(xcp, ycp, degrees(-alp0[i]))
                    x1 += x0[i]
                    y1 += y0[i]
            else:
                if i == 3:
                    # Counter-clockwise rotation by alp0[i]
                    x1, y1 = rotate_values(xcp, ycp, degrees(alp0[i]))
                    x1 += x0[i]
                    y1 -= y0[i]
                elif i == 1 or i == 2 or i == 4 or i == 5:
                    # Counter-clockwise rotation by alp0[i], negate y values.
                    x1, y1 = rotate_values(xcp, -ycp, degrees(alp0[i]))
                    x1 += x0[i]
                    y1 -= y0[i]
                elif i == 0:
                    # Counter-clockwise rotation by alp0[i], negate y values.
                    x1, y1 = rotate_values(xcp, -ycp, degrees(alp0[i]))
                    x1 += x0[i]
                    y1 += y0[i]

            sbedist[j, i, :] = x1 + sx0[j]
            sbndist[j, i, :] = y1 + sy0[j]

    return sbedist, sbndist, sx0, sy0

if __name__ == "__main__":
    dalp = 0.  # Rotation angle of the entire station
    alpg = [12.0, 20.0, 24.0, 34.0, 42.0, 57.0]  # lattice rotation angles
    alp0 = [-90.0, -54.0, 18.0, 90.0, 162.0, 234.0]  # Sub-station angles
    salp0 = [0.0, -18.0, 54.0, 126.0, 198.0, 270.0]  # Station angles

    sbedist, sbndist, sx0, sy0 = \
        generate_ska_low_v4a_super_station(dalp, alpg, alp0, salp0)

    sb6r0 = numpy.loadtxt('../ant_files/sbfile6r0.ant')

    plot_num_stations = 6

    x = sbedist.flatten()
    y = sbndist.flatten()
    x = x[:48*6*plot_num_stations]
    y = y[:48*6*plot_num_stations]
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(x, y, 'b+')
    r_ant = 1.5 / 2.0
    for i in range(x.shape[0]):
        circle = pyplot.Circle((x[i], y[i]), r_ant, color='k',
                               fill=True, alpha=0.3)
        ax.add_artist(circle)

    r_station = 29.0 / 2.0
    for i in range(plot_num_stations):
        circle = pyplot.Circle((sx0[i], sy0[i]), r_station, color='k', linestyle='--',
                               fill=False, alpha=0.5)
        ax.add_artist(circle)

    circle = pyplot.Circle((0.0, 0.0), 70.0/2.0, color='k', linestyle='--',
                           fill=False, alpha=0.5)
    ax.add_artist(circle)

    ax.plot(sb6r0[:48*6*plot_num_stations, 0],
            sb6r0[:48*6*plot_num_stations, 1], 'x',
            markeredgecolor='r', markerfacecolor='None')
    ax.set_title('6r0: 1.5m antenna footprints')

    lim = max(max(numpy.abs(ax.get_xlim())),
              max(numpy.abs(ax.get_ylim())))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    pyplot.show()

