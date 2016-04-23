# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy
from numpy.random import rand, seed
from math import ceil, log, exp, floor
import matplotlib.pyplot as pyplot


def grid_position(x, y, scale, r):
    jx = int(floor((x + r) * scale))
    jy = int(floor((y + r) * scale))
    return jx, jy


def grid_position_2(x, y, scale, grid_size):
    jx = int(round(x * scale)) + grid_size / 2
    jy = int(round(y * scale)) + grid_size / 2
    return jx, jy


def get_trail_position(r):
    x = -r + 2.0 * r * rand()
    y = -r + 2.0 * r * rand()
    return x, y


def norm_pdf(x, sigma):
    return exp(-(x**2) / (2.0*sigma**2))


def gridgen6(edge_density, num_points, diameter, min_dist, n_miss_max=1000):
    """Generate uniform random positions within a specified diameter which
    are no closer than a specified minimum distance.

    Uses and algorithm where the area is split into a grid sectors
    so that when checking for minimum distance, only nearby points need to be
    considered.
    """

    # Fix seed to study closest match fails (with fixed seed can
    # print problematic indices)
    # seed(2)

    r = diameter / 2.0  # Radius
    p = 1.0 / edge_density
    max_dist = p * min_dist
    sigma = r / log(p)**0.5
    scale_max = 1.0 / norm_pdf(diameter / 2.0, sigma)
    edge_dist = (1.0 / norm_pdf(20, sigma)) * min_dist
    print('- Edge dist:', edge_dist)
    print('- Area scaling: %f' % (edge_dist**2 / min_dist**2))

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / max_dist)))
    grid_size += grid_size%2
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 2
    print('- Station d: %f' % diameter)
    print('- Grid size: %i' % grid_size)
    print('- Min dist: %f' % min_dist)
    print('- Max dist: %f' % max_dist)
    print('- Sigma: %f' % sigma)
    print('- Grid cell: %f' % grid_cell)
    print('- check width: %i' % check_width)

    # Pre-allocate coordinate arrays
    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    # Grid meta-data
    # First index in the grid
    grid_i_start = numpy.zeros((grid_size, grid_size), dtype='i8')
    # Last index in the grid
    grid_i_end = numpy.zeros((grid_size, grid_size), dtype='i8')
    # Points in grid cell.
    grid_count = numpy.zeros((grid_size, grid_size), dtype='i8')
    # Next coordinate index.
    grid_next = numpy.zeros(num_points, dtype='i8')

    n = num_points
    n_req = num_points
    num_miss = 0
    max_num_miss = 0
    for j in range(n_req):

        done = False
        while not done:

            # Generate a trail position
            xt, yt = get_trail_position(r)
            rt = (xt**2 + yt**2)**0.5
            ant_r = min_dist / (2.0 * norm_pdf(rt, sigma))

            # Check if the point is inside the diameter.
            if rt + ant_r > r:
                num_miss += 1

            # Check if min distance is met.
            else:
                jx, jy = grid_position(xt, yt, scale, r)
                y0 = max(0, jy - check_width)
                y1 = min(grid_size, jy + check_width + 1)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size, jx + check_width + 1)
                dmin = diameter  # Set initial min to diameter.
                for ky in range(y0, y1):
                    for kx in range(x0, x1):
                        if grid_count[kx, ky] > 0:
                            i_other = grid_i_start[kx, ky]
                            for num_other in range(grid_count[kx, ky]):
                                dx = xt - x[i_other]
                                dy = yt - y[i_other]
                                dr = (dx**2 + dy**2)**0.5
                                r_other = (x[i_other]**2 + y[i_other]**2)**0.5
                                ant_r_other = min_dist / (2.0 * norm_pdf(r_other, sigma))

                                if dr - ant_r_other <= dmin:
                                    dmin = dr - ant_r_other
                                i_other = grid_next[i_other]

                scaled_min_dist_3 = (min_dist / 2.0) / norm_pdf(rt, sigma)

                if dmin >= scaled_min_dist_3:
                    x[j] = xt
                    y[j] = yt
                    if grid_count[jx, jy] == 0:
                        grid_i_start[jx, jy] = j
                    else:
                        grid_next[grid_i_end[jx, jy]] = j
                    grid_i_end[jx, jy] = j
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

    return x, y, sigma

if __name__ == '__main__':
    n = 200
    d = 40
    d_min = 1.2
    edge_density = 0.1  # w.r.t. centre.
    x, y, sigma = gridgen6(edge_density, n, d, d_min, n_miss_max=100000)
    num_points = len(x)

    if num_points != n:
        print('did not find enough antennas!')
    else:
        # TODO-BM plot with circles of minimum spacing
        fig = pyplot.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0.0, hspace=0.0)

        ax = fig.add_subplot(111, aspect='equal')
        ax.plot(x, y, '.', color='k', ms=3.0)
        circle = pyplot.Circle((0, 0), d / 2.0, color='k',
                               linestyle='--', fill=False)
        ax.add_artist(circle)
        for i in range(num_points):
            xp = x[i]
            yp = y[i]
            rp = (xp**2 + yp**2)**0.5  # Radius of this point

            dx = x - x[i]
            dy = y - y[i]
            dist = (dx**2 + dy**2)**0.5
            i_min = numpy.where(dist == dist[dist != 0].min())[0][0]
            min_dist = dist[i_min]
            ro = (x[i_min]**2 + y[i_min]**2)**0.5  # Radius of closest point

            # Min dist radius for this point + that of closest defines
            # defines if antennas overlap.
            r_ant_this = d_min / (2.0 * norm_pdf(rp, sigma))
            r_ant_closest = d_min / (2.0 * norm_pdf(ro, sigma))

            ox = x[i_min] - xp
            oy = y[i_min] - yp
            ax.arrow(xp, yp, ox, oy, head_width=0.1, head_length=0.05,
                     fc='g', ec='g')

            ax.text(xp, yp, '%i' % i, fontsize='x-small')
            if min_dist >= r_ant_this + r_ant_closest:
                color = 'b'
            else:
                print(i, min_dist, r_ant_this, r_ant_closest,
                      r_ant_this + r_ant_closest)
                color = 'r'

            circle = pyplot.Circle((xp, yp), r_ant_this, color=color,
                                   fill=False, alpha=0.1)
            ax.add_artist(circle)
            circle = pyplot.Circle((xp, yp), (d_min / 2.0), color=color,
                                   fill=True, alpha=0.3)
            ax.add_artist(circle)
        ax.set_title('%i / %i' % (len(x), n))
        ax.set_xlim(-(d / 2.0 + d_min / 2.0), d / 2.0 + d_min / 2.0)
        ax.set_ylim(-(d / 2.0 + d_min / 2.0), d / 2.0 + d_min / 2.0)
        pyplot.show()
