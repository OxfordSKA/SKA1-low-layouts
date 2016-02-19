# -*- coding: utf-8 -*-

import numpy
from numpy.random import rand, seed
from math import floor, ceil
import matplotlib.pyplot as pyplot
from taper_function import taylor


def grid_position(x, y, scale, grid_size):
    jx = int(round(x * scale)) + grid_size / 2
    jy = int(round(y * scale)) + grid_size / 2
    return jx, jy


def get_trail_position(r):
    x = -r + 2.0 * r * rand()
    y = -r + 2.0 * r * rand()
    return x, y


def gridgen_no_taper(num_points, diameter, min_dist, n_miss_max=1000):
    """Generate uniform random positions within a specified diameter which
    are no closer than a specified minimum distance.

    Uses and algorithm where the area is split into a grid sectors
    so that when checking for minimum distance, only nearby points need to be
    considered.
    """

    # seed(2)

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / min_dist)))
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = int(ceil(grid_cell / min_dist))
    print('- Grid size: %i' % grid_size)
    print('- Min dist: %f' % min_dist)
    print('- Grid cell: %f' % grid_cell)
    print('- check width: %i' % check_width)

    r = diameter / 2.0  # Radius
    r_sq = r**2  # Radius, squared
    min_dist_sq = min_dist**2  # minimum distance, squared

    # Pre-allocate coordinate arrays
    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    # Grid meta-data
    grid_istart = numpy.zeros((grid_size, grid_size), dtype='i8')  # First index in the grid
    grid_iend = numpy.zeros((grid_size, grid_size), dtype='i8')  # Last index in the grid
    grid_count = numpy.zeros((grid_size, grid_size), dtype='i8')  # Points in grid cell.
    grid_next = numpy.zeros(num_points, dtype='i8')   # Next coordinate index.

    n = num_points
    n_req = num_points
    num_miss = 0
    max_num_miss = 0
    for j in range(n_req):

        done = False
        while not done:

            # Generate a trail position
            xt, yt = get_trail_position(r)

            # Check if the point is inside the diameter.
            if (xt**2 + yt**2)**0.5 + (min_dist / 2.0) > r:
                num_miss += 1

            # Check if min distance is met.
            else:
                jx, jy = grid_position(xt, yt, scale, grid_size)
                y0 = max(0, jy - check_width)
                y1 = min(grid_size - 1, jy + check_width)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size - 1, jx + check_width)
                dmin = diameter  # Set initial min to diameter.
                for ky in range(y0, y1 + 1):
                    for kx in range(x0, x1 + 1):
                        if grid_count[kx, ky] > 0:
                            kh1 = grid_istart[kx, ky]
                            for kh in range(grid_count[kx, ky]):
                                dx = xt - x[kh1]
                                dy = yt - y[kh1]
                                dmin = min((dx**2 + dy**2)**0.5, dmin)
                                kh1 = grid_next[kh1]

                if dmin >= min_dist:
                    x[j] = xt
                    y[j] = yt
                    if grid_count[jx, jy] == 0:
                        grid_istart[jx, jy] = j
                    else:
                        grid_next[grid_iend[jx, jy]] = j
                    grid_iend[jx, jy] = j
                    grid_count[jx, jy] += 1
                    max_num_miss = max(max_num_miss, num_miss)
                    num_miss = 0
                    done = True
                else:
                    num_miss += 1

            if num_miss >= n_miss_max:
                n = j - 1
                done = True

        if num_miss >= n_miss_max:
            max_num_miss = max(max_num_miss, num_miss)
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    print('- Found %i / %i points [max. misses: %i / %i]' %
          (n, n_req, max_num_miss, n_miss_max))

    return x, y


def main():
    n = 256
    d = 35
    d_min = 1.55
    x, y = gridgen_no_taper(n, d, d_min, n_miss_max=100000)
    num_points = len(x)

    weight = taylor(x, y, sll=-28)

    print('sum weights = %.5f / %i' % (numpy.sum(weight), num_points))

    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot((x**2 + y**2)**0.5, weight, '+')
    pyplot.show()

    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(x, y, s=30, c=weight, lw=0)
    pyplot.show()

    if num_points != n:
        print('did not find enough antennas!')
    else:

        # TODO-BM plot with circles of minimum spacing
        fig = pyplot.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, aspect='equal')
        ax.plot(x, y, '.', color='k', ms=3.0)
        circle = pyplot.Circle((0, 0), d / 2.0, color='k',
                               linestyle='--', fill=False)
        ax.add_artist(circle)
        for i in range(num_points):
            dx = x - x[i]
            dy = y - y[i]
            dx = dx[dx != 0]
            dy = dy[dy != 0]
            dist = (dx**2 + dy**2)**0.5
            # print(i, numpy.min(dist), numpy.min(dist) < d_min)
            xp = x[i]
            yp = y[i]
            if numpy.min(dist) >= d_min:
                color = 'b'
            else:
                print('antenna-%-3i d_min = %.3f' % (i, numpy.min(dist)))
                color = 'r'
            circle = pyplot.Circle((xp, yp), d_min / 2.0, color=color,
                                   fill=False)
            ax.add_artist(circle)
            extent = d_min / 2**0.5
            xp = xp
            yp = yp - extent / 2.0 * 2**0.5
            angle = 45.0
            # xp = xp - extent / 2.0
            # yp = yp - extent / 2.0
            # angle = 0.0
            rect = pyplot.Rectangle((xp, yp),
                                    width=extent, height=extent,
                                    angle=angle, color=color, linestyle='-',
                                    fill=True, alpha=0.3)
            ax.add_artist(rect)
        ax.set_title('%i / %i' % (len(x), n))
        ax.set_xlim(-(d / 2.0 + d_min / 2.0), d / 2.0 + d_min / 2.0)
        ax.set_ylim(-(d / 2.0 + d_min / 2.0), d / 2.0 + d_min / 2.0)
        pyplot.show()


if __name__ == '__main__':
    main()
