# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
from math import pi, cos, sin, radians, sqrt, atan2, degrees, radians
from numpy.random import random
import matplotlib.pyplot as pyplot


def main():
    # inc = 360 / 5.0
    # alp0 = numpy.arange(5) * inc + inc / 2.0 - 90.0

    # ============================================================
    # Super-station angle
    dalp = 0.0

    # Lattice orientation
    alpg = [12.0, 20.0, 24.0, 34.0, 42.0, 57.0]

    # Sub-station orientations
    ss_ang = [-90.0, -54.0, 18.0, 90.0, 162.0, 234.0]

    # Station orientations
    s_ang = [0.0, -18.0, 54.0, 126.0, 198.0, 270.0]
    # ============================================================

    nside = 5
    var = numpy.arange(0., 0.999, 0.001)
    theta = var * 2. * pi
    incx = 2. * pi / 5. * 1000.
    thetax = theta * 1000.
    thetam5 = (thetax % incx) / 1000.0

    # Work out some geometry for the sub-station pentagon
    rmax = 7.0  # maximum sub station radius, in m
    sqinc = 2.0 * rmax / 9.33
    jitter = 0.0 * sqinc
    side = 2.0 * rmax * cos(radians(54.))
    rmin = rmax * sin(radians(54.))
    psep = 2.0 * rmin + 0.5
    area = (side**2 / 4.0) * sqrt(5.0 * (5.0 + 2.0 * sqrt(5.0)))
    sqarea = (2.0 * rmax)**2
    # Relative number of locations to populate on a square grid.
    retarea = sqarea / area
    # Station separation
    ssep = 2.0 * (rmax + side * cos(radians(18.0))) + 1.5

    # Positions and orientation of sub-stations
    x0 = numpy.zeros(6)
    y0 = numpy.zeros(6)
    alp0 = numpy.zeros(6)
    for i in range(6):
        alp0[i] = radians(ss_ang[i]) - radians(dalp)
        x0[i] = psep * cos(-alp0[i])
        y0[i] = psep * cos(-alp0[i])
    x0[0] = 0.0
    y0[0] = 0.0

    # Position and orientation of stations.
    sx0 = numpy.zeros(6)
    sy0 = numpy.zeros(6)
    salp0 = numpy.zeros(6)
    for i in range(6):
        salp0[i] = radians(s_ang[i]) - radians(dalp)
        sx0[i] = ssep * cos(-salp0[i])
        sy0[i] = ssep * cos(-salp0[i])
    x0[0] = 0.0
    y0[0] = 0.0

    max_ant = 100
    sbedist = numpy.zeros((6, 6, max_ant), dtype='f8')
    sbndist = numpy.zeros((6, 6, max_ant), dtype='f8')

    # Dimensions for hexagonal lattice generation.
    nx = 15
    ny = 15

    # # Loop over stations in super-station
    # for j in range(6):
    #     nx = 15
    #     ny = 12
    #     iant = 0   # Sub-station index
    #
    #     # Generate a hexagonal lattice of antennas of nx by ny
    #     for ix in range(nx):
    #         for iy in range(ny):
    #
    #             # Generate hexagonal lattice.
    #             xc = -1.5 * rmax + ((ix + 0.5 * iy) * sqinc) + jitter * (2. * random() - 0.5)
    #             yc = -rmax + (iy * sqinc * sqrt(3.) / 2.) + jitter * (2. * random() - 0.5)
    #
    #             # Rotate the lattice by alpg
    #             xcp = xc * cos(radians(alpg[j])) + yc * sin(radians(alpg[j]))
    #             ycp = yc * cos(radians(alpg[j])) - xc * sin(radians(alpg[j]))
    #
    #             rc = sqrt(xcp**2 + ycp**2)
    #             thetac = atan2(ycp, xcp) + 2. * pi
    #             thetax = thetac * 1000.
    #             thetam5 = (thetax % incx) / 1000.
    #             radp = rmax * cos(pi / 5.) / cos(thetam5 - pi/5.)
    #
    #             # FIXME-BM get complete set of rotations from the scripts.
    #             # For lattice positions that fit within the sub-station,
    #             # shift by x0, y0 and rotate by alp0
    #             if rc < 0.9 * radp:
    #                 # loop over sub-stations
    #                 for i in range(6):
    #                     if j == 0:
    #                         if i == 3:
    #                             sbedist[j, i, iant] = xcp * cos(alp0[i]) - ycp * sin(alp0[i]) + x0[i]
    #                             sbndist[j, i, iant] = -ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i]
    #                         else:
    #                             sbedist[j, i, iant] = xcp * cos(alp0[i]) + ycp * sin(alp0[i]) + x0[i]
    #                             sbndist[j, i, iant] = ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i]
    #                     else:
    #                         if i == 0:
    #                             sbedist[j, i, iant] = xcp * cos(alp0[i]) + ycp * sin(alp0[i]) + x0[i]
    #                             sbndist[j, i, iant] = -(ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i])
    #                         elif i == 1 or i == 2 or i == 4 or i == 5:
    #                             sbedist[j, i, iant] = xcp * cos(alp0[i]) + ycp * sin(alp0[i]) + x0[i]
    #                             sbndist[j, i, iant] = -(ycp * cos(alp0[i]) - ycp * sin(alp0[i]) + y0[i])
    #                         elif i == 3:
    #                             sbedist[j, i, iant] = xcp * cos(alp0[i]) - ycp * sin(alp0[i]) + x0[i]
    #                             sbndist[j, i, iant] = -(-ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i])
    #
    #                     sbedist[j, i, iant] += sx0[j]
    #                     sbndist[j, i, iant] += sy0[j]
    #
    #                 iant += 1
    #     print('=> Station %i has %i antennas per sub-station' % (j, iant))
    #
    # return sbedist[:, :, 0:iant], sbndist[:, :, 0:iant]


if __name__ == '__main__':
    main()
    # print(x.shape)
    # fig = pyplot.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.plot(x.flatten(), y.flatten(), '+')
    # pyplot.show()
