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
import sys

def generate_ska_low_v4a_super_station(super_station_angle,
                                       lattice_orientation,
                                       sub_station_orientation,
                                       station_orientation,
                                       fract_jitter,
                                       ax):
    """Function to generate a ska1 low v4a super station configuration file.

    Args:
        super_station_angle (float):
            Orientation angle of the super-station, in degrees.
        lattice_orientation (array_like, length: 6):
            Orientation angles of the hexagonal latices used to construct
            sub-stations, in degrees.
        sub_station_orientation (array_like, length: 6):
            Orientation angles of sub-stations within a station, in degrees.
        station_orientation (array_like, length: 6):
            Orientation angles of stations within the super-station, in degrees.
        fract_jitter (float):
            Fractional (wrt antenna separation) shift of antenna positions
            in the lattice. For a regular layout use a value of 0.0.
        ant_file_name (string):
            File name with which to save the super-station antenna layout.
        make_plot (bool, defualt=True):
            If true, make a png plot of the super-station.
    """
    dalp = super_station_angle
    alpg = lattice_orientation
    angles = sub_station_orientation
    sangles = station_orientation

    print(dalp)
    print(alpg)
    print(angles)
    print(sangles)

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
    area = (side**2 / 4.) * sqrt(5. * (5. + 2. * sqrt(5.)))
    sqarea = (2. * rmax)**2
    retarea = sqarea / area  # retarea is the relative number of locations to
    # populate on a square grid.
    ssep = 2. * (rmax + side * cos(radians(18))) + 1.5  # station separation??!
    jitter = fract_jitter * sqinc

    # position and orientation of sub-stations
    x0 = numpy.zeros(6)
    y0 = numpy.zeros(6)
    alp0 = numpy.zeros(6)
    for i in range(6):
        alp0[i] = radians(angles[i])
        x0[i] = psep * cos(-alp0[i])
        y0[i] = psep * sin(-alp0[i])
    x0[0] = 0.0
    y0[0] = 0.0

    # Positions and orientation of stations.
    salp0 = numpy.zeros(6)
    sx0 = numpy.zeros(6)
    sy0 = numpy.zeros(6)
    for i in range(6):
        salp0[i] = radians(sangles[i])
        sx0[i] = ssep * cos(-salp0[i])
        sy0[i] = ssep * sin(-salp0[i])
    sx0[0] = 0.0
    sy0[0] = 0.0
    print ' '

    # == PLOT OUTLINES ========================================================
    radp = rmax * cos(pi / 5.) / numpy.cos(thetam5 - pi / 5.)
    xd = radp * numpy.cos(theta)
    yd = radp * numpy.sin(theta)
    # ax.plot(xd, yd, '-', color='r', label2='unroated sub-station')
    for j in range(6):
        for i in range(6):
            # rotate sub-station about origin
            xdp = xd * cos(alp0[i]) + yd * sin(alp0[i])
            ydp = yd * cos(alp0[i]) - xd * sin(alp0[i])
            # translate sub-station
            xdp += x0[i]
            ydp += y0[i]
            ax.plot(xdp, ydp, '-', color='g')
            # rotate sub-station about station centre.
            xdp1 = xdp * cos(salp0[j]) - ydp * sin(salp0[j])
            ydp1 = xdp * sin(salp0[j]) + ydp * cos(salp0[j])
            # translate sub-station to station position
            xdp1 += sx0[j]
            ydp1 += sy0[j]
            if j == 0:
                ax.plot(xdp1, ydp1, '-', color='y', lw=3.0)
            # # rotate entire super-station
            # xdp2 = xdp1 * cos(dalp) + ydp1 * sin(dalp)
            # ydp2 = xdp1 * sin(dalp) - ydp1 * cos(dalp)
            # ax.plot(xdp2, ydp2, ':', color='k')
    # ==========================================================================
    #
    # max_ant = 100
    # sbedist = numpy.zeros((6, 6, max_ant), dtype='f8')
    # sbndist = numpy.zeros((6, 6, max_ant), dtype='f8')
    #
    # # Dimensions for hexagonal lattice generation.
    # nx = 14
    # ny = 14
    #
    # # Loop over stations
    # for j in range(6):
    #
    #     # Generate a hexagonal lattice of antennas of nx by ny
    #     lx = numpy.zeros((nx, ny))
    #     ly = numpy.zeros((nx, ny))
    #
    #     for ix in range(nx):
    #         for iy in range(ny):
    #             lx[ix, iy] = (ix + 0.5 * iy) * sqinc
    #             ly[ix, iy] = iy * sqinc * (sqrt(3.) / 2.)
    #             lx[ix, iy] += jitter * (2. * random() - 0.5)
    #             ly[ix, iy] += jitter * (2. * random() - 0.5)
    #     lx -= lx.max() / 2.
    #     ly -= ly.max() / 2.
    #
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
    #     # FIXME-BM this dosn't quite work as the lattice now rotates
    #     # relative to the sub-stations
    #     for i in range(6):
    #         x1 = xcp * cos(alp0[i]) + ycp * sin(alp0[i])
    #         y1 = xcp * sin(alp0[i]) - ycp * cos(alp0[i])
    #         x1 += x0[i]
    #         y1 += y0[i]
    #         x2 = x1 * cos(salp0[j]) + y1 * sin(salp0[j])
    #         y2 = x1 * sin(salp0[j]) - y1 * cos(salp0[j])
    #         x2 += sx0[j]
    #         y2 += sy0[j]
    #         x3 = x2 * cos(dalp) - y2 * sin(dalp)
    #         y3 = x2 * sin(dalp) + y2 * cos(dalp)
    #         sbedist[j, i, 0:num_ant] = x3
    #         sbndist[j, i, 0:num_ant] = y3

    # sbedist = sbedist[:, :, 0:num_ant]
    # sbndist = sbndist[:, :, 0:num_ant]
    # ax.plot(sbedist.flatten(), sbndist.flatten(), '+', color='b',
    #         label2='generated')



if __name__ == "__main__":

    # scripts = ('lowpent6.mon', 'lowpent7.mon', 'lowpent6r0.mon',
    #            'lowpent7r0.mon')
    # for script in scripts:
    #     if script == 'lowpent6.mon':
    #         dalp = 0.  # Angle of the entire station
    #         alpg = [95., 95., 95., 95., 95., 95.]  # lattice rotation angles
    #         angles = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
    #         sangles = [0, -18., 54., 126., 198., 270.]  # Station orientations.
    #         fract_jitter = 0.15
    #
    #     elif script == 'lowpent7.mon':
    #         dalp = 10.  # Rotation angle of the entire station
    #         alpg = [95., 95., 95., 95., 95., 95.]  # lattice rotation angles
    #         angles = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
    #         sangles = [0, -18., 54., 126., 198., 270.]  # Station orientations.
    #         fract_jitter = 0.15
    #
    #     elif script == 'lowpent6r0.mon':
    #         dalp = 0.  # Rotation angle of the entire station
    #         alpg = [12., 20., 24., 34., 42., 57.]  # lattice rotation angles
    #         angles = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
    #         sangles = [0, -18., 54., 126., 198., 270.]  # Station orientations.
    #         fract_jitter = 0.0
    #
    #     elif script == 'lowpent7r0.mon':
    #         dalp = 10.  # Rotation angle of the entire station
    #         alpg = [12., 20., 24., 34., 42., 57.]  # lattice rotation angles
    #         angles = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
    #         sangles = [0, -18., 54., 126., 198., 270.]  # Station orientations.
    #         fract_jitter = 0.0
    #
    #     else:
    #         print 'ERROR: Unknown parameter set.'
    #         sys.exit(1)
    #
    #     generate_ska_low_v4a_super_station(dalp, alpg, angles, sangles,
    #                                        fract_jitter,
    #                                        '%s.ant' % script,
    #                                        make_plot=True)

    dalp = 0.0
    alpg = [12.0, 20.0, 24.0, 34.0, 42.0, 57.0]
    angles = [-90.0, -54.0, 18.0, 90.0, 162.0, 234.0]
    sangles = [0.0, -18.0, 54.0, 126.0, 198.0, 270.0]
    fract_jitter = 0.0

    fig = plt.figure(figsize=(10., 10.))
    ax = fig.add_subplot(111, aspect='equal')
    generate_ska_low_v4a_super_station(dalp, alpg, angles, sangles,
                                       fract_jitter, ax)
    sb6r0 = numpy.loadtxt('../ant_files/sbfile6r0.ant')
    ax.plot(sb6r0[:, 0], sb6r0[:, 1], 'k.', label='reference',
            alpha=0.3)
    # plt.legend()
    # ax.grid(True)
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')

    # plt.savefig('test_sbfile6r0_v3.png')
    plt.show()




    # dalp = 10.  # Rotation angle of the entire station
    # alpg = [12., 20., 24., 34., 42., 57.]  # lattice rotation angles
    # angles = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
    # sangles = [0, -18., 54., 126., 198., 270.]  # Station orientations.
    # fract_jitter = 0.0
    # fig = plt.figure(figsize=(10., 10.))
    # ax = fig.add_subplot(111, aspect='equal')
    # generate_ska_low_v4a_super_station(dalp, alpg, angles, sangles,
    #                                    fract_jitter, ax)
    # sb6r0 = numpy.loadtxt('../ant_files/sbfile7r0.ant')
    # # ax.plot(sb6r0[:, 0], sb6r0[:, 1], 'rx', label2='reference')
    # plt.legend()
    # plt.savefig('test_sbfile7r0.png')
    # plt.show()
