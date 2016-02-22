# -*- coding: utf-8 -*-

import numpy
from numpy.random import rand, seed
from math import ceil
import matplotlib.pyplot as pyplot
import time


def grid_position(x, y, scale, grid_size):
    jx = int(round(x * scale)) + grid_size / 2
    jy = int(round(y * scale)) + grid_size / 2
    return jx, jy


def get_trail_position(r):
    x = -r + 2.0 * r * rand()
    y = -r + 2.0 * r * rand()
    return x, y


def gridgen_no_taper(num_points, diameter, min_dist, max_trials=1000):
    """Generate uniform random positions within a specified diameter which
    are no closer than a specified minimum distance.

    Uses and algorithm where the area is split into a grid sectors
    so that when checking for minimum distance, only nearby points need to be
    considered.
    """
    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / min_dist)))
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = int(ceil(grid_cell / min_dist))

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
                y1 = min(grid_size - 1, jy + check_width)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size - 1, jx + check_width)
                d_min = diameter  # Set initial min to diameter.
                for ky in range(y0, y1 + 1):
                    for kx in range(x0, x1 + 1):
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


def main():
    n = 256
    d = 35
    d_min = 1.5
    x, y = gridgen_no_taper(n, d, d_min, max_trials=100000)
    num_points = len(x)

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
            yp -= extent / 2.0 * 2 ** 0.5
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


def plot_station(x, y, min_sep, diameter, title, file_name=None):
    num_points = len(x)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(x, y, '+', color='k', ms=1.0)
    circle = pyplot.Circle((0, 0), diameter / 2.0, color='k',
                           linestyle='--', fill=False)
    ax.add_artist(circle)

    for i in range(num_points):
        dx = x - x[i]
        dy = y - y[i]
        dx = dx[dx != 0]
        dy = dy[dy != 0]
        dist = (dx**2 + dy**2)**0.5
        xp = x[i]
        yp = y[i]
        color = 'r'
        if dist.min() >= min_sep:
            color = 'b'
        circle = pyplot.Circle((xp, yp), min_sep / 2.0, color=color,
                               fill=False)
        ax.add_artist(circle)
        extent = min_sep / 2**0.5
        xp = xp
        yp -= extent / 2.0 * 2 ** 0.5
        angle = 45.0
        # xp = xp - extent / 2.0
        # yp = yp - extent / 2.0
        # angle = 0.0
        rect = pyplot.Rectangle((xp, yp),
                                width=extent, height=extent,
                                angle=angle, color=color, linestyle='-',
                                fill=True, alpha=0.3)
        ax.add_artist(rect)

    ax.set_title(title)
    ax.set_xlabel('x [metres]')
    ax.set_ylabel('y [metres]')
    lim1 = -(diameter / 2.0 + min_sep / 2.0)
    lim2 = diameter / 2.0 + min_sep / 2.0
    ax.set_xlim(lim1, lim2)
    ax.set_ylim(lim1, lim2)

    if file_name:
        pyplot.savefig(file_name)
    else:
        pyplot.show()


def get_spacings(x, y):
    num_points = len(x)
    min_dist = numpy.zeros(num_points, dtype='f8')
    for j in range(num_points):
        dx = x - x[j]
        dy = y - y[j]
        dx = dx[dx != 0]
        dy = dy[dy != 0]
        dist = (dx**2 + dy**2)**0.5
        min_dist[j] = dist.min()
    return min_dist


def main2():
    num_antennas = 256 * 6
    diameter = 85.0  # m
    antenna_footprint = 1.5  # m
    max_trials = 100000
    for i in range(1):
        t1 = time.time()
        x, y, trial_count = gridgen_no_taper(num_antennas, diameter,
                                             antenna_footprint, max_trials)
        num_points = len(x)
        print('[%03i] found %i / %i points, max trials = %i, '
              'time taken = %.2f s' % (i, num_points, num_antennas,
                                       numpy.sort(trial_count)[-1],
                                       time.time() - t1))
        if not num_points == num_antennas:
            continue
        min_dist = get_spacings(x, y)
        plot_station(x, y, antenna_footprint, diameter,
                     'station-%i, mean sep. = %.2f m' % (i, min_dist.mean()),
                     'station-%03i.png' % i)


def main3():
    inc = (360.0/5.0)
    # ang = numpy.arange(5) * inc + inc / 2.0 - 90.0
    ang = numpy.arange(5) * inc

    num_antennas = 256 * 6
    diameter = 85.0  # m
    radius = diameter / 2.0
    antenna_footprint = 1.5  # m
    max_trials = 100000
    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    x, y, trial_count = gridgen_no_taper(num_antennas, diameter,
                                         antenna_footprint, max_trials)
    num_points = len(x)
    if not num_points == num_antennas:
        return

    r = (x**2 + y**2)**0.5
    r0 = numpy.sort(r)[256]
    # r_outer = (diameter - r0) / 2.0
    r_outer = diameter / 3.0

    inner_idx = numpy.where(r < r0)[0]
    x0 = x[inner_idx]
    y0 = y[inner_idx]

    outer_idx = numpy.where(r >= r0)[0]
    x = x[outer_idx]
    y = y[outer_idx]

    print(num_points, inner_idx.shape[0] + outer_idx.shape[0])

    circle = pyplot.Circle((0, 0), r0, color='c', linestyle='--',
                           fill=False, alpha=1.0, lw=3.0)
    ax.add_artist(circle)

    sx = r_outer * numpy.cos(numpy.radians(ang))
    sy = r_outer * numpy.sin(numpy.radians(ang))
    # sx = numpy.append(sx, 0.0)
    # sy = numpy.append(sy, 0.0)
    ax.plot(sx, sy, 'r+', ms=30.0, alpha=1.0)
    circle = pyplot.Circle((0, 0), diameter / 2.0, color='k', linestyle='--',
                           fill=False, alpha=1.0)
    ax.add_artist(circle)
    for i in range(sx.shape[0]):
        circle = pyplot.Circle((sx[i], sy[i]), diameter / 6.0, color='k',
                               linestyle='--', fill=False, alpha=1.0)
        ax.add_artist(circle)

    ax.plot([0.0], [0.0], 'r+', ms=30.0, alpha=1.0)
    circle = pyplot.Circle((0.0, 0.0), diameter / 6.0, color='k',
                           linestyle='--', fill=False, alpha=1.0)
    ax.add_artist(circle)

    ax.plot(x, y, '+', markeredgecolor='k', ms=1.0)
    for i in range(x.shape[0]):
        circle = pyplot.Circle((x[i], y[i]), antenna_footprint / 2.0,
                               color='b', linestyle='--', fill=True, alpha=0.2)
        ax.add_artist(circle)

    # Find the centre nearest to each antenna
    color = ['r', 'g', 'b', 'y', 'm', 'k']
    station_count = numpy.zeros(6, dtype='i8')
    for i in range(x.shape[0]):
        dx = x[i] - sx
        dy = y[i] - sy
        dr = (dx**2 + dy**2)**0.5
        station_idx = numpy.where(dr == min(dr))[0][0]
        station_count[station_idx] += 1
        circle = pyplot.Circle((x[i], y[i]), antenna_footprint / 2.0,
                               color=color[station_idx], linestyle='--',
                               fill=True, alpha=0.3)
        ax.add_artist(circle)

    ax.plot(x0, y0, '+', markeredgecolor='k', ms=1.0)
    for i in range(x0.shape[0]):
        circle = pyplot.Circle((x0[i], y0[i]), antenna_footprint / 2.0,
                               color=color[5], linestyle='--',
                               fill=True, alpha=0.3)
        ax.add_artist(circle)

    for i in range(5):
        print(i, color[i], station_count[i])
    print(5, color[5], len(x0))
    pyplot.show()

if __name__ == '__main__':
    main3()
    # TODO-BM main4() split area into pentagons?
