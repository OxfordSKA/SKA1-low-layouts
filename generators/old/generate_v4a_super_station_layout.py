# -*- coding: utf-8 -*-
import numpy
from math import pi, cos, sin, radians, sqrt, atan2
from numpy.random import random


def generate(super_station_angle, lattice_orientation, sub_station_orientation,
             station_orientation, fract_jitter):
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
    """
    dalp = radians(super_station_angle)
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
        alp0[i] = radians(angles[i]) - dalp
        x0[i] = psep * cos(-alp0[i])
        y0[i] = psep * sin(-alp0[i])
    x0[0] = 0.0
    y0[0] = 0.0

    # Positions and orientation of stations.
    salp0 = numpy.zeros(6)
    sx0 = numpy.zeros(6)
    sy0 = numpy.zeros(6)
    for i in range(6):
        salp0[i] = radians(sangles[i]) - dalp
        sx0[i] = ssep * cos(-salp0[i])
        sy0[i] = ssep * sin(-salp0[i])
    sx0[0] = 0.0
    sy0[0] = 0.0

    max_ant = 100
    sbedist = numpy.zeros((6, 6, max_ant), dtype='f8')
    sbndist = numpy.zeros((6, 6, max_ant), dtype='f8')

    # Loop over stations in super-station
    for j in range(6):
        nx = 15
        ny = 12
        iant = 0   # Sub-station index

        # Generate a hexagonal lattice of antennas of nx by ny
        for ix in range(nx):
            for iy in range(ny):

                # Generate hexagonal lattice.
                xc = -1.5 * rmax + ((ix + 0.5 * iy) * sqinc) + jitter * (2. * random() - 0.5)
                yc = -rmax + (iy * sqinc * sqrt(3.) / 2.) + jitter * (2. * random() - 0.5)

                # Rotate the lattice by alpg
                xcp = xc * cos(radians(alpg[j])) + yc * sin(radians(alpg[j]))
                ycp = yc * cos(radians(alpg[j])) - xc * sin(radians(alpg[j]))

                rc = sqrt(xcp**2 + ycp**2)
                thetac = atan2(ycp, xcp) + 2. * pi
                thetax = thetac * 1000.
                thetam5 = (thetax % incx) / 1000.
                radp = rmax * cos(pi / 5.) / cos(thetam5 - pi/5.)

                # FIXME-BM get complete set of rotations from the scripts.
                # For lattice positions that fit within the sub-station,
                # shift by x0, y0 and rotate by alp0
                if rc < 0.9 * radp:
                    # loop over sub-stations
                    for i in range(6):
                        if j == 0:
                            if i == 3:
                                sbedist[j, i, iant] = xcp * cos(alp0[i]) - ycp * sin(alp0[i]) + x0[i]
                                sbndist[j, i, iant] = -ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i]
                            else:
                                sbedist[j, i, iant] = xcp * cos(alp0[i]) + ycp * sin(alp0[i]) + x0[i]
                                sbndist[j, i, iant] = ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i]
                        else:
                            if i == 0:
                                sbedist[j, i, iant] = xcp * cos(alp0[i]) + ycp * sin(alp0[i]) + x0[i]
                                sbndist[j, i, iant] = -(ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i])
                            elif i == 1 or i == 2 or i == 4 or i == 5:
                                sbedist[j, i, iant] = xcp * cos(alp0[i]) + ycp * sin(alp0[i]) + x0[i]
                                sbndist[j, i, iant] = -(ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i])
                            elif i == 3:
                                sbedist[j, i, iant] = xcp * cos(alp0[i]) - ycp * sin(alp0[i]) + x0[i]
                                sbndist[j, i, iant] = -(-ycp * cos(alp0[i]) - xcp * sin(alp0[i]) + y0[i])


                        sbedist[j, i, iant] += sx0[j]
                        sbndist[j, i, iant] += sy0[j]

                    iant += 1
        print '=> Station %i has %i antennas per sub-station' % (j, iant)

    return sbedist[:, :, 0:iant], sbndist[:, :, 0:iant]
