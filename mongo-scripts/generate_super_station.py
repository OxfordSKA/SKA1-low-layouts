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


def generate_lattice_2(nx, ny, sq_inc):
    xc = numpy.zeros((nx, ny))
    yc = numpy.zeros((nx, ny))
    for iy in range(ny):
        for ix in range(nx):
            xc[iy, ix] = (ix + 0.5 * iy) * sq_inc
            yc[iy, ix] = (iy * sq_inc * sqrt(3.0) / 2.0)
    xc -= xc[ny/2, nx/2]
    yc -= yc[ny/2, nx/2]
    return xc.flatten(), yc.flatten()


def generate_lattice_3(nx, ny, sq_inc,lattice_angle):
    xc = numpy.zeros((nx, ny))
    yc = numpy.zeros((nx, ny))
    for iy in range(ny):
        for ix in range(nx):
            xc[iy, ix] = (ix + 0.5 * iy) * sq_inc
            yc[iy, ix] = (iy * sq_inc * sqrt(3.0) / 2.0)
    xc -= xc[ny/2, nx/2]
    yc -= yc[ny/2, nx/2]
    xc, yc = rotate_values(xc, yc, lattice_angle)
    return xc.flatten(), yc.flatten()


def select_antennas_2(x, y, r_max, angle):
    num_samples = 100
    inc_x = ((2.0 * pi) / 5.0) * num_samples
    theta = numpy.arctan2(y, x) + 2.0 * pi - radians(angle)
    theta_x = theta * num_samples
    theta_m5 = (theta_x % inc_x) / num_samples
    rad_p = r_max * (cos(pi / 5.0) / numpy.cos(theta_m5 - pi / 5.0))
    rc = (x**2 + y**2)**0.5
    sgood = rc / rad_p
    sind = numpy.argsort(sgood)
    sgood = sgood[sind]
    n_ok = numpy.argmax(sgood > 1.0) - 1
    rc_1 = rc[sind]
    x1 = x[sind]
    y1 = y[sind]
    x2 = x1[:48]
    y2 = y1[:48]
    xe = rad_p * numpy.cos(theta + radians(angle))
    ye = rad_p * numpy.sin(theta + radians(angle))
    return x2, y2, xe, ye



def rotate_values(x, y, angle):
    """Rotation by angle, +angle == counterclockwise rotation"""
    xr = x * cos(radians(angle)) - y * sin(radians(angle))
    yr = x * sin(radians(angle)) + y * cos(radians(angle))
    return xr, yr


def select_antennas(xcp, ycp, rmax, incx):
    num_points = 1000
    thetac = numpy.arctan2(ycp, xcp) + 2.0 * pi
    thetax = thetac * num_points
    thetam5 = (thetax % incx) / float(num_points)
    radp = rmax * cos(pi / 5.0) / numpy.cos(thetam5 - pi / 5.0)
    rc = (xcp**2 + ycp**2)**0.5
    sgood = rc / radp
    sind = numpy.argsort(sgood)
    sgood = sgood[sind]
    n_ok = numpy.argmax(sgood > 1.0) - 1
    xcp = xcp[sind]
    ycp = ycp[sind]
    return xcp[:48], ycp[:48]


def generate_super_station(sub_station_radius_m=7.0, super_station_angle=-10.0):
    num_sub_stations = 6
    num_stations = 6
    num_antennas = 48

    # Work out some geometry for the sub-station pentagon
    # FIXME-BM remove all arbitrary values here
    r_max = sub_station_radius_m
    sqinc = 2.0 * r_max / 9.33
    side = 2.0 * r_max * cos(radians(54.0))
    rmin = r_max * sin(radians(54.0))
    psep = 2.0 * rmin + 0.5
    ssep = 2.0 * (r_max + side * cos(radians(18.0))) + 1.5

    # Positions and orientation of sub-stations
    sub_station_angle = numpy.arange(num_sub_stations - 1) * \
                        (360.0 / (num_sub_stations - 1)) - 90.0
    sub_station_angle = numpy.insert(sub_station_angle, 0, 90.0)
    sub_station_angle = numpy.radians(sub_station_angle)
    x0 = numpy.zeros(num_sub_stations)
    y0 = numpy.zeros(num_sub_stations)
    for i in range(1, num_sub_stations):
        x0[i] = psep * cos(sub_station_angle[i])
        y0[i] = psep * sin(sub_station_angle[i])
    #     ax.text(x0[i], y0[i], '%i' % i, color='r')
    # ax.text(0.0, 0.0, '%i' % 0, color='r')
    # ax.plot(x0, y0, 'r+', ms=10.0, lw=2.0)

    # # Positions and orientation of stations.
    station_angles = numpy.arange(num_stations - 1) * \
        (360.0 / (num_stations - 1)) + 90.0
    station_angles = numpy.insert(station_angles, 0, -90.0)
    station_angles = numpy.radians(station_angles)
    sx0 = numpy.zeros(num_stations)
    sy0 = numpy.zeros(num_stations)
    for i in range(1, num_stations):
        sx0[i] = ssep * cos(station_angles[i])
        sy0[i] = ssep * sin(station_angles[i])
    #     ax.text(sx0[i], sy0[i] + 1.0, '%i' % i, color='b')
    # ax.plot(sx0, sy0, 'b+', ms=10.0, lw=2.0)

    ant_x = numpy.zeros((num_sub_stations, num_stations, num_antennas))
    ant_y = numpy.zeros_like(ant_x)

    colors = ['r', 'g', 'b', 'y', 'c', 'm']

    # Variables for plotting lattice outline
    num_samples = 100
    theta = numpy.linspace(0.0, 1.0, num_samples)
    theta *= 2.0 * pi
    inc_x = ((2.0 * pi) / 5.0) * num_samples
    theta_x = theta * num_samples
    theta_m5 = (theta_x % inc_x) / float(num_samples)
    radp = r_max * cos(pi / 5.0) / numpy.cos(theta_m5 - pi / 5.0)
    xd = radp * numpy.cos(theta)
    yd = radp * numpy.sin(theta)

    fig = pyplot.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    print(numpy.degrees(sub_station_angle))
    print(numpy.degrees(station_angles))

    # Loop over stations
    for j in range(6):

        for i in range(num_sub_stations):

            # Lattice outline
            xpd1, ypd1 = rotate_values(xd, yd, degrees(sub_station_angle[i]))
            xpd1 += x0[i]
            ypd1 += y0[i]
            xpd2, ypd2 = rotate_values(xpd1, ypd1,
                                       degrees(station_angles[j]) + 90.0)
            xpd2 += sx0[j]
            ypd2 += sy0[j]
            xpd3, ypd3 = rotate_values(xpd2, ypd2, super_station_angle)
            ax.plot(xpd3, ypd3, linestyle='-', color=colors[j])

            # sub_station_centre
            cx1 = x0[i]
            cy1 = y0[i]
            cx2, cy2 = rotate_values(cx1, cy1,
                                     degrees(station_angles[j]) + 90.0)
            cx2 += sx0[j]
            cy2 += sy0[j]
            cx3, cy3 = rotate_values(cx2, cy2, super_station_angle)
            ax.plot(cx3, cy3, '+', ms=15, color=colors[j])
            ax.text(cx3, cy3 + 1.0, '%i,%i' % (j, i), fontsize='x-small')

            # Generate hexagonal lattice
            xc, yc = generate_lattice_3(12, 12, sqinc,
                                        super_station_angle +
                                        numpy.random.randint(0, 72))

            angle = degrees(sub_station_angle[i] +
                            station_angles[j])
            xc1, yc1, xe1, ye1 = select_antennas_2(xc, yc, r_max,
                                                   super_station_angle
                                                   + 18.0 + angle)

            xc2 = xc1 + cx3
            yc2 = yc1 + cy3
            xe2 = xe1 + cx3
            ye2 = ye1 + cy3

            ax.plot(xc2, yc2, 'ko')


    lim = max(max(numpy.abs(ax.get_xlim())), max(numpy.abs(ax.get_ylim())))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    pyplot.show()


    # return sbedist, sbndist, sx0, sy0



if __name__ == '__main__':

    generate_super_station()

    # fig = pyplot.figure(figsize=(16, 8))
    # ax = fig.add_subplot(111, aspect='equal')
    # pyplot.show()
