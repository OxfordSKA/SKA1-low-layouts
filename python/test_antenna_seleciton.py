# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as pyplot
from math import pi, cos, sin, radians, sqrt, atan2, degrees
from gridgen_no_taper import gridgen_no_taper

def rotate_values(x, y, angle):
    """Rotation by angle, +angle == counterclockwise rotation"""
    xr = x * cos(radians(angle)) - y * sin(radians(angle))
    yr = x * sin(radians(angle)) + y * cos(radians(angle))
    return xr, yr


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


def select_antennas(x, y, angle=18.0):
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
    x2 = x1[:n_ok]
    y2 = y1[:n_ok]

    xe = rad_p * numpy.cos(theta + radians(angle))
    ye = rad_p * numpy.sin(theta + radians(angle))
    return x2, y2, xe, ye


if __name__ == '__main__':
    sub_station_radius_m = 7.0
    r_max = sub_station_radius_m
    sqinc = 2.0 * r_max / 9.33
    side = 2.0 * r_max * cos(radians(54.0))
    rmin = r_max * sin(radians(54.0))
    psep = 2.0 * rmin + 0.5
    ssep = 2.0 * (r_max + side * cos(radians(18.0))) + 1.5

    # Generate the lattice
    # x, y = generate_lattice_2(30, 30, sqinc/2.0)
    x, y, _ = gridgen_no_taper(90, 20, 1.5, 10000)

    xs, ys, xe, ye = select_antennas(x, y, angle=-90.0)

    fig1 = pyplot.figure(figsize=(16, 8))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.plot(xe, ye, 'bs', ms=1.0)
    ax1.plot(x, y, '+')
    ax1.plot(xs, ys, 'o')
    ax1.grid()
    pyplot.show()
