# -*- coding: utf-8 -*-
"""
Script to generate ska1-low v4a super-station antenna layouts.
This is a port of super mongo script(s) written by Robert Braun.

benjamin.mort@oerc.ox.ac.uk [last update: 8/12/2015]
"""

import numpy
import matplotlib.pyplot as plt
from math import pi, cos, sin, radians, sqrt, atan2, degrees
from numpy.random import random

def generate_ska_low_v4a_super_station(super_station_angle,
                                       lattice_orientation,
                                       sub_station_orientation,
                                       station_orientation,
                                       fract_jitter):
    dalp = super_station_angle
    alpg = lattice_orientation
    angles = sub_station_orientation
    sangles = station_orientation

    var = numpy.arange(0., 0.999, 0.001)
    theta = var * 2. * pi
    incx = 2. * pi / 5. * 1000.
    thetax = theta * 1000.
    thetam5 = (thetax % incx) / 1000.0

    # Work out some geometry for the sub-station pentagon
    rmax = 7.  # maximum sub station radius, in m
    rmin = rmax * sin(radians(54.))
    sqinc = 2. * rmax / 9.33
    side = 2. * rmax * cos(radians(54.))
    psep = 2. * rmin + 0.5
    ssep = 2. * (rmax + side * cos(radians(18))) + 1.5
    jitter = fract_jitter * sqinc

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

    max_ant = 100
    sbedist = numpy.zeros((6, 6, max_ant), dtype='f8')
    sbndist = numpy.zeros((6, 6, max_ant), dtype='f8')

    # Dimentions for hexagonal lattice generation.
    nx = 15
    ny = 15

    # Loop over stations
    for j in range(6):

        # Loop over sub-stations
        for i in range(6):

            # 1. Generate hexagonal lattice and restrict to 'best' 48 antennas
            lx = numpy.zeros((nx, ny))
            ly = numpy.zeros((nx, ny))
            for ix in range(nx):
                for iy in range(ny):
                    lx[ix, iy] = -1.5 * rmax + (ix + 0.5 * iy)
                    ly[ix, iy] = -rmax + (iy * sqinc * sqrt(3.) / 2.)

            if j == 0 and i == 0:
                fig = plt.figure(figsize=(10.0, 10.0))
                ax = fig.add_subplot(111, aspect='equal')
                ax.plot(lx, ly, '+')
                plt.show()



            # 2. Rotate and translate the lattice into position
            pass

            #     # Generate a hexagonal lattice of antennas of nx by ny
            #     lx = numpy.zeros((nx, ny))
            #     ly = numpy.zeros((nx, ny))
            #     # TODO-BM: vectorise this
            #     for ix in range(nx):
            #         for iy in range(ny):
            #             lx[ix, iy] = (ix + 0.5 * iy) * sqinc
            #             ly[ix, iy] = iy * sqinc * (sqrt(3.) / 2.)
            #             lx[ix, iy] += jitter * (2. * random() - 0.5)
            #             ly[ix, iy] += jitter * (2. * random() - 0.5)
            #     lx -= lx.max() / 2.
            #     ly -= ly.max() / 2.
            #     # Rotate the grid with the given lattice angle.
            #     lxp = lx * cos(radians(alpg[j])) - ly * sin(radians(alpg[j]))
            #     lyp = lx * sin(radians(alpg[j])) + ly * cos(radians(alpg[j]))
            #
            #     # Remove points in the lattice outside the sub-station.
            #     lr = numpy.sqrt(lxp**2 + lyp**2)
            #     ltheta = numpy.arctan2(lyp, lxp) + radians(54.0)
            #     lthetax = ltheta * 1000.
            #     lthetam5 = numpy.mod(lthetax, incx) / 1000.
            #     lradp = rmax * cos(pi / 5.) / numpy.cos(lthetam5 - pi / 5)
            #     pr = 0.89
            #     xcp = lxp[lr < pr * lradp]
            #     ycp = lyp[lr < pr * lradp]
            #     num_ant = xcp.shape[0]
            #     print 'num ant', num_ant
            #     # ax.plot(xcp, ycp, 'rx')
            #
            #     # FIXME-BM this dosnt quite work as the lattice now rotates
            #     # relative to the sub-stations
            #     for i in range(6):
            #         x1 = xcp * cos(alp0[i]) - ycp * sin(alp0[i])
            #         y1 = xcp * sin(alp0[i]) + ycp * cos(alp0[i])
            #         x1 += x0[i]
            #         y1 += y0[i]
            #         x2 = x1 * cos(salp0[j]) - y1 * sin(salp0[j])
            #         y2 = x1 * sin(salp0[j]) + y1 * cos(salp0[j])
            #         x2 += sx0[j]
            #         y2 += sy0[j]
            #         x3 = x2 * cos(dalp) - y2 * sin(dalp)
            #         y3 = x2 * sin(dalp) + y2 * cos(dalp)
            #         sbedist[j, i, 0:num_ant] = x3
            #         sbndist[j, i, 0:num_ant] = y3
            #
            # return sbedist[:, :, 0:num_ant], sbndist[:, :, 0:num_ant]

if __name__ == "__main__":
    # dalp_ = 0.  # Rotation angle of the entire station
    # alpg_ = [12., 20., 24., 34., 42., 57.]  # lattice rotation angles
    # alp0_ = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
    # salp0_ = [0, -18., 54., 126., 198., 270.]  # Station orientations.
    # fract_jitter = 0.0
    # fig = plt.figure(figsize=(10., 10.))
    # ax = fig.add_subplot(111, aspect='equal')
    # generate_ska_low_v4a_super_station(dalp, alpg, angles, sangles,
    #                                    fract_jitter, ax)
    # sb6r0 = numpy.loadtxt('../sbfile7r0.ant')
    # # ax.plot(sb6r0[:, 0], sb6r0[:, 1], 'rx', label='reference')
    # plt.legend()
    # plt.savefig('test_sbfile7r0.png')
    # plt.show()
    pass
