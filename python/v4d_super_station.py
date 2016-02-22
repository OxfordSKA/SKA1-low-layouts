"""Module to generate super-stations for trail v4d spec. layouts"""
# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib.pyplot as pyplot
import numpy
from numpy.random import rand


def gridgen(num_points, diameter, min_dist, max_trials=1000):
    def grid_position(x, y, scale, grid_size):
        jx = int(round(x * scale)) + grid_size / 2
        jy = int(round(y * scale)) + grid_size / 2
        return jx, jy

    def get_trail_position(r):
        x = -r + 2.0 * r * rand()
        y = -r + 2.0 * r * rand()
        return x, y

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / min_dist)))
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 1

    r = diameter / 2.0  # Radius
    r_sq = r**2  # Radius, squared
    min_dist_sq = min_dist**2  # minimum distance, squared
    r_ant = min_dist / 2.0

    # Pre-allocate coordinate arrays
    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    # Grid meta-data
    grid_i_start = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_i_end = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_count = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_i_next = numpy.zeros(num_points, dtype='i8')

    n = num_points
    n_req = num_points
    num_tries = 0
    try_count = list()
    for j in range(n_req):

        done = False
        while not done:

            # Generate a trail position
            xt, yt = get_trail_position(r)
            rt = (xt**2 + yt**2)**0.5

            # Check if the point is inside the diameter.
            if rt + r_ant > r:
                num_tries += 1

            # Check if min distance is met.
            else:
                jx, jy = grid_position(xt, yt, scale, grid_size)
                y0 = max(0, jy - check_width)
                y1 = min(grid_size, jy + check_width + 1)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size, jx + check_width + 1)
                d_min = diameter  # Set initial min to diameter.
                for ky in range(y0, y1):
                    for kx in range(x0, x1):
                        if grid_count[kx, ky] > 0:
                            kh1 = grid_i_start[kx, ky]
                            for kh in range(grid_count[kx, ky]):
                                dx = xt - x[kh1]
                                dy = yt - y[kh1]
                                d_min = min((dx**2 + dy**2)**0.5, d_min)
                                kh1 = grid_i_next[kh1]

                if d_min >= min_dist:
                    x[j] = xt
                    y[j] = yt
                    if grid_count[jx, jy] == 0:
                        grid_i_start[jx, jy] = j
                    else:
                        grid_i_next[grid_i_end[jx, jy]] = j
                    grid_i_end[jx, jy] = j
                    grid_count[jx, jy] += 1
                    try_count.append(num_tries)
                    num_tries = 0
                    done = True
                else:
                    num_tries += 1

            if num_tries >= max_trials:
                n = j - 1
                done = True

        if num_tries >= max_trials:
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    return x, y, try_count


def gen_super_stations():
    """Generation 85 super-stations by rotation"""
    num_stations = 6
    diameter = 30.0  # m
    antenna_diameter = 1.5
    num_ant_station = 180
    angles = numpy.arange(num_stations - 1) * (360.0 / float(num_stations - 1))
    angles += 90.0
    r0 = diameter + 1.0
    sx = r0 * numpy.cos(numpy.radians(angles))
    sy = r0 * numpy.sin(numpy.radians(angles))
    sx = numpy.append(sx, 0.0)
    sy = numpy.append(sy, 0.0)

    ant_x, ant_y, tries = gridgen(num_ant_station, diameter, antenna_diameter,
                                  max_trials=10000)
    print('Number of antennas generated = %i' % ant_x.shape[0])
    if ant_x.shape[0] != num_ant_station:
        print('Error, not enough antennas generated... %i / %i' %
              (ant_x.shape[0], num_ant_station))
        return

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    # Station centres
    ax.plot(sx, sy, '+', ms=20, mew=2.0)
    # Station diameters
    for i in range(num_stations):
        circle = pyplot.Circle((sx[i], sy[i]), diameter / 2.0,
                               color='k', linestyle='-',
                               fill=False, alpha=0.5)
        ax.add_artist(circle)
        ant_x, ant_y, tries = gridgen(num_ant_station, diameter,
                                      antenna_diameter, max_trials=10000)
        if ant_x.shape[0] != num_ant_station:
            continue
        ant_x += sx[i]
        ant_y += sy[i]
        for j in range(ant_x.shape[0]):
            circle = pyplot.Circle((ant_x[j], ant_y[j]), antenna_diameter / 2.0,
                                   color='r', linestyle='-',
                                   fill=True, alpha=0.3)
            ax.add_artist(circle)

    # Super-station diameter
    circle = pyplot.Circle((0.0, 0.0), 81.0 / 2.0,
                           color='k', linestyle='-',
                           fill=False, alpha=0.5)
    ax.add_artist(circle)

    circle = pyplot.Circle((0.0, 0.0), 66.0 / 2.0,
                           color='r', linestyle='-',
                           fill=False, alpha=0.5)
    ax.add_artist(circle)

    lim = 3.0 * diameter / 2.0 + 5.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    pyplot.show()


if __name__ == '__main__':
    gen_super_stations()
