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
            xc[ix, iy] = -1.5 * rmax + (ix + 0.5 * iy)
            yc[ix, iy] = -rmax + (iy * sqinc * sqrt(3.0) / 2.0)
    return xc.flatten(), yc.flatten()


def rotate_lattice(xc, yc, angle_deg):
    xcp = xc * cos(radians(angle_deg)) - yc * sin(radians(angle_deg))
    ycp = xc * sin(radians(angle_deg)) + yc * cos(radians(angle_deg))
    return xcp, ycp


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
    for j in range(1):

        # Generate hexagonal lattice
        xc, yc = generate_lattice(nx, ny, rmax, sqinc)

        # Rotate the lattice by the lattice angle.
        xcp, ycp = rotate_lattice(xc, yc, -alpg)
        rc = (xcp**2 + ycp**2)**0.5

        thetac = numpy.arctan2(ycp, xcp) + 2.0 * pi
        thetax = thetac * 1000.0
        thetam5 = (thetax % incx) / 1000.0
        radp = rmax * cos(pi / 5.0) / numpy.cos(thetam5 - pi / 5.0)
        sgood = rc / radp
        sind = numpy.argsort(sgood)
        sgood = sgood[sind]
        xcp = xcp[sind]
        ycp = ycp[sind]
        xcp = xcp[:48]
        ycp = ycp[:48]
        sgood = sgood[:48]

        # if j == 0 and i == 0:
        #     fig = pyplot.figure(figsize=(10.0, 10.0))
        #     ax = fig.add_subplot(111, aspect='equal')
        #     ax.plot(xc.flatten(), yc.flatten(), 'r+')
        #     # ax.plot(xcp.flatten(), ycp.flatten(), 'b+')
        #     ax.scatter(xcp, ycp, s=20, c=sgood, lw=0)
        #     lim = max(max(ax.get_xlim()), max(ax.get_ylim()))
        #     ax.set_xlim(-lim, lim)
        #     ax.set_ylim(-lim, lim)
        #     ax.plot(ax.get_xlim(), [0, 0], 'k--', alpha=0.5, lw=0.5)
        #     ax.plot([0, 0], ax.get_ylim(), 'k--', alpha=0.5, lw=0.5)
        #     pyplot.show()

        # Loop over sub-stations
        for i in range(6):
            x1 = xcp * cos(alp0[i]) + ycp * sin(alp0[i])
            y1 = ycp * cos(alp0[i]) - xcp * sin(alp0[i])
            x1 += x0[i]
            y1 += y0[i]
            # x2 = x1 * cos(salp0[j]) - y1 * sin(salp0[j])
            # y2 = x1 * sin(salp0[j]) + y1 * cos(salp0[j])
            # x2 += sx0[j]
            # y2 += sy0[j]
            # x3 = x2 * cos(dalp) - y2 * sin(dalp)
            # y3 = x2 * sin(dalp) + y2 * cos(dalp)
            sbedist[j, i, :] = x1
            sbndist[j, i, :] = y1

    return sbedist, sbndist

if __name__ == "__main__":
    dalp = 0.  # Rotation angle of the entire station
    alpg = 95.0  # lattice rotation angles
    alp0 = [-90.0, -54.0, 18.0, 90.0, 162.0, 234.0]  # Sub-station angles
    salp0 = [0.0, -18.0, 54.0, 126.0, 198.0, 270.0]  # Station angles

    sbedist, sbndist = generate_ska_low_v4a_super_station(dalp, alpg, alp0, salp0)

    sb6r0 = numpy.loadtxt('../ant_files/sbfile6r0.ant')

    x = sbedist.flatten()
    y = sbndist.flatten()
    x = x[:48*6]
    y = y[:48*6]
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(x, y, 'k+')
    r_ant = 1.5 / 2.0
    for i in range(x.shape[0]):
        circle = pyplot.Circle((x[i], y[i]), r_ant, color='b',
                               fill=False, alpha=0.5)
        ax.add_artist(circle)
    r_station = 29.0 / 2.0
    circle = pyplot.Circle((0, 0), r_station, color='k', linestyle='--',
                           fill=False, alpha=0.5)
    ax.add_artist(circle)
    lim = max(max(numpy.abs(ax.get_xlim())),
              max(numpy.abs(ax.get_ylim())))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.plot(sb6r0[:48*6, 0], sb6r0[:48*6, 1], 'gx')
    pyplot.show()



