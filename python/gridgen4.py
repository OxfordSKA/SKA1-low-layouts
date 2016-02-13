# -*- coding: utf-8 -*-

import numpy
from numpy.random import rand
from math import floor
import matplotlib.pyplot as pyplot


def grid_position(x, y, scale, r):
    jx = int(floor(x + r) * scale)
    jy = int(floor(y + r) * scale)
    return jx, jy


def get_trail_position(r):
    x = -r + 2.0 * r * rand()
    y = -r + 2.0 * r * rand()
    return x, y


def gridgen4(num_points, diameter, min_dist, n_miss_max=10000):
    """Generate uniform random positions within a specified diameter which
    are no closer than a specified minimum distance.

    Uses and algorithm where the area is split into a grid sectors
    so that when checking for minimum distance, only nearby points need to be
    considered.
    """

    # Grid size and scaling onto the grid
    grid_size = min(100, int(floor(float(diameter) / min_dist)))
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    print('- Grid size: %i' % grid_size)
    print('- Grid cell: %f' % grid_cell)

    r = diameter / 2.0  # Radius
    r_sq = r**2  # Radius, squared
    min_dist_sq = min_dist**2  # minimum distance, squared

    # Pre-allocate coordinate arrays
    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    # Grid meta-data
    next = numpy.zeros(num_points, dtype='i8')   # Next coordinate index.
    h1 = -numpy.ones((grid_size, grid_size), dtype='i8')  # First index in the grid
    h2 = -numpy.ones((grid_size, grid_size), dtype='i8')  # Last index in the grid
    grid_count = numpy.zeros((grid_size, grid_size), dtype='i8')  # Points in grid cell.

    n = num_points
    n_req = num_points
    num_miss = 0
    for j in range(n_req):

        # First time no need to check the minimum distance req, just needs
        # to be inside the diameter.
        if j == 0:
            done = False
            while not done:
                x[j], y[j] = get_trail_position(r)
                done = (x[j]**2 + y[j]**2) <= r_sq
            jx, jy = grid_position(x[j], y[j], scale, r)
            grid_count[jx, jy] += 1
            h1[jx, jy] = 0
            h2[jx, jy] = 0

        # All other points have to be inside the diameter and match the
        # minimum separation requirements.
        else:
            done = False
            while not done:
                xt, yt = get_trail_position(r)

                # Check if the point is inside the diameter
                if (xt**2 + yt**2) > r_sq:
                    num_miss += 1
                else:
                    # Scale onto grid.
                    jx, jy = grid_position(xt, yt, scale, r)
                    # Find minimum distance to other points
                    y0 = max(0, jy - 1)
                    y1 = min(grid_size - 1, jy + 1)
                    x0 = max(0, jx - 1)
                    x1 = min(grid_size - 1, jx + 1)
                    dmin_sq = diameter
                    for ky in range(y0, y1 + 1):
                        for kx in range(x0, x1 + 1):
                            if grid_count[kx, ky] > 0:
                                kh1 = h1[kx, ky]
                                for kh in range(grid_count[kx, ky]):
                                    dx = xt - x[kh1]
                                    dy = yt - y[kh1]
                                    dist_sq = dx**2 + dy**2
                                    dmin_sq = min(dist_sq, dmin_sq)
                                    kh1 = next[kh1]

                    # Check if the minimum distance requirement is met.
                    if dmin_sq >= min_dist_sq:
                        x[j] = xt
                        y[j] = yt
                        if h1[jx, jy] == -1:
                            h1[jx, jy] = j
                        else:
                            next[h2[jx, jy]] = j
                        h2[jx, jy] = j
                        grid_count[jx, jy] += 1
                        num_miss = 0
                        done = True
                    else:
                        num_miss += 1

                if num_miss >= n_miss_max:
                    n = j - 1
                    done = True

        if num_miss >= n_miss_max:
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    return x, y

if __name__ == '__main__':
    x, y = gridgen4(256, 35.0, 1.5, n_miss_max=10000)

    # TODO-BM plot with circles of minimum spacing
    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(x, y, '.')
    ax.set_title('%i / %i' % (len(x), 256))
    pyplot.show()
