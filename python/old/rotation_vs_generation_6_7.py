# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as pyplot
from math import pi, cos, sin, radians, sqrt, atan2, degrees


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


def generate_super_station_6(dalp, alpg, angles, sangles):
    # ??
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


def generate_super_station_7(dalp, alpg, angles, sangles):
    # ??
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
                    x1, y1 = rotate_values(xcp, -ycp, degrees(-alp0[i]))
                    x1 += x0[i]
                    y1 += y0[i]
                else:
                    x1, y1 = rotate_values(xcp, ycp, degrees(-alp0[i]))
                    x1 += x0[i]
                    y1 += y0[i]
            else:
                if i == 3:
                    x1, y1 = rotate_values(-xcp, ycp, degrees(-alp0[i]))
                    x1 -= x0[i]
                    y1 -= y0[i]
                else:
                    x1, y1 = rotate_values(-xcp, -ycp, degrees(-alp0[i]))
                    x1 -= x0[i]
                    y1 -= y0[i]

            sbedist[j, i, :] = x1 + sx0[j]
            sbndist[j, i, :] = y1 + sy0[j]

    return sbedist, sbndist, sx0, sy0


if __name__ == '__main__':
    dalp = 0.  # Rotation angle of the entire station
    alpg = [12.0, 20.0, 24.0, 34.0, 42.0, 57.0]  # lattice rotation angles
    alp0 = [-90.0, -54.0, 18.0, 90.0, 162.0, 234.0]  # Sub-station angles
    salp0 = [0.0, -18.0, 54.0, 126.0, 198.0, 270.0]  # Station angles
    x6, y6, sx6, sy6 = generate_super_station_6(dalp, alpg, alp0, salp0)
    x6 = x6.flatten()
    y6 = y6.flatten()
    sb6 = numpy.loadtxt('../ant_files/sbfile6r0.ant')

    dalp = 10.  # Rotation angle of the entire station
    alpg = [12.0, 20.0, 24.0, 34.0, 42.0, 57.0]  # lattice rotation angles
    alp0 = [-90.0, -54.0, 18.0, 90.0, 162.0, 234.0]  # Sub-station angles
    salp0 = [0.0, -18.0, 54.0, 126.0, 198.0, 270.0]  # Station angles
    x7, y7, sx7, sy7 = generate_super_station_7(dalp, alpg, alp0, salp0)
    x7 = x7.flatten()
    y7 = y7.flatten()
    sb7 = numpy.loadtxt('../ant_files/sbfile7r0.ant')

    d_ant = 1.5
    r_ant = d_ant / 2.0

    x7r, y7r = rotate_values(x6, y6, 10.0)
    x7r += 100.0
    x7 += 100.0

    fig = pyplot.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(x6, y6, 'b+')
    ax.plot(sb6[:, 0], sb6[:, 1], 'yx')
    # for i in range(x0.shape[0]):
    #     circle = pyplot.Circle((x0[i], y0[i]), r_ant, color='k', fill=True,
    #                            alpha=0.3)
    #     ax.add_artist(circle)

    ax.plot(x7, y7, 'b+', label='generated ss7')
    ax.plot(sb7[:, 0] + 100.0, sb7[:, 1], 'yx', label='ant file ss7')
    ax.plot(x7r, y7r, 'rx', label='rot ss7')
    ax.legend()

    pyplot.show()
