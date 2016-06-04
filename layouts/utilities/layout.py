# -*- coding: utf-8 -*-

import math

import numpy


def rotate_coords(x, y, angle):
    """ Rotate coordinates counter clockwise by angle, in degrees.
    Args:
        x (array like): array of x coordinates.
        y (array like): array of y coordinates.
        angle (float): Rotation angle, in degrees.

    Returns:
        (x, y) tuple of rotated coordinates

    """
    theta = math.radians(angle)
    xr = x * numpy.cos(theta) - y * numpy.sin(theta)
    yr = x * numpy.sin(theta) + y * numpy.cos(theta)
    return xr, yr


def log_spiral_1(r0, b, delta_theta_deg, n):
    """Computes coordinates on a log spiral.

    Args:
        r0 (float): minimum radius
        b (float): Spiral constant.
        delta_theta_deg (float): angle between points, in degrees.
        n (int): Number of points.

    Returns:
        tuple: (x, y) coordinates
    """
    t = numpy.arange(n) * math.radians(delta_theta_deg)
    tmp = r0 * numpy.exp(b * t)
    x = tmp * numpy.cos(t)
    y = tmp * numpy.sin(t)
    return x, y


def log_spiral_2(r0, r1, b, n):
    """Computes coordinates on a log spiral.

    Args:
        r0 (float): minimum radius
        r1 (float): maximum radius
        b (float): Spiral constant.
        n (int): Number of points.

    Returns:
        tuple: (x, y) coordinates
    """
    if b == 0.0:
        x = numpy.exp(numpy.linspace(math.log(r0), math.log(r1), n))
        y = numpy.zeros(n)
    else:
        t_max = math.log(r1 / r0) * (1.0 / b)
        t = numpy.linspace(0, t_max, n)
        tmp = r0 * numpy.exp(b * t)
        x = tmp * numpy.cos(t)
        y = tmp * numpy.sin(t)
    return x, y


def log_spiral_clusters(r0, r1, b, n, n_cluster, r_cluster, min_sep):
    """Computes coordinates on a log spiral.

    Args:
        r0 (float): minimum radius
        r1 (float): maximum radius
        b (float): Spiral constant.
        n (int): Number of points.
        n_cluster (int): Number of points per cluster
        r_cluster (double): Radius of the cluster.
        min_sep (double): minimum separation of points in each cluster.

    Returns:
        tuple: (x, y) coordinates
    """
    if b == 0.0:
        x = numpy.exp(numpy.linspace(math.log(r0), math.log(r1), n))
        y = numpy.zeros(n)
    else:
        t_max = math.log(r1 / r0) * (1.0 / b)
        t = numpy.linspace(0, t_max, n)
        tmp = r0 * numpy.exp(b * t)
        x = tmp * numpy.cos(t)
        y = tmp * numpy.sin(t)
    x_all = numpy.zeros((n, n_cluster))
    y_all = numpy.zeros((n, n_cluster))
    for k in range(n):
        xc, yc, _ = rand_uniform_2d(n_cluster, r_cluster, min_sep, max_tries=10000)
        if not xc.shape[0] == n_cluster:
            raise RuntimeError('Failed to generate cluster [%i] %i/%i stations '
                               'generated.' % (k, xc.shape[0], n_cluster))
        x_all[k, :] = xc + x[k]
        y_all[k, :] = yc + y[k]
    x_all = x_all.flatten()
    y_all = y_all.flatten()
    return x_all, y_all, x, y


def rand_uniform_2d(num_points, r1, min_sep, r0=0.0, max_tries=1000):
    """

    Args:
        num_points:
        r0 (float): minimum radius
        r1 (float):  maximum radius
        min_sep (float): Minimum separation of points. (diameter around each
                         point)
        max_tries:

    Returns:

    """
    def grid_position(x, y, scale, r):
        ix = int(math.floor(x + r) * scale)
        iy = int(math.floor(y + r) * scale)
        return ix, iy

    def get_trial_position(r):
        return tuple(numpy.random.rand(2) * 2.0 * r - r)

    grid_size = min(100, int(round(float(r1 * 2.0) / min_sep)))
    grid_cell = float(r1 * 2.0) / grid_size  # Grid sector size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 1  ## ???

    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    grid = {
        'start': numpy.zeros((grid_size, grid_size), dtype='i8'),
        'end': numpy.zeros((grid_size, grid_size), dtype='i8'),
        'count': numpy.zeros((grid_size, grid_size), dtype='i8'),
        'next': numpy.zeros(num_points, dtype='i8')
    }

    n = num_points
    num_tries = 0
    try_count = list()
    for j in range(num_points):
        done = False
        while not done:
            xt, yt = get_trial_position(r1)
            rt = (xt**2 + yt**2)**0.5
            # Point is inside area defined by: r0 < r < r1
            if rt + min_sep / 2.0 > r1 or rt - min_sep / 2.0 < r0:
                num_tries += 1
            else:
                jx, jy = grid_position(xt, yt, scale, r1)
                y0 = max(0, jy - check_width)
                y1 = min(grid_size, jy + check_width + 1)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size, jx + check_width + 1)

                # Find minimum spacing between trial and other points.
                d_min = r1 * 2.0
                for ky in range(y0, y1):
                    for kx in range(x0, x1):
                        if grid['count'][ky, kx] > 0:
                            kh1 = grid['start'][ky, kx]
                            for kh in range(grid['count'][ky, kx]):
                                dx = xt - x[kh1]
                                dy = yt - y[kh1]
                                d_min = min((dx**2 + dy**2)**0.5, d_min)
                                kh1 = grid['next'][kh1]

                if d_min >= min_sep:
                    x[j] = xt
                    y[j] = yt
                    if grid['count'][jy, jx] == 0:
                        grid['start'][jy, jx] = j
                    else:
                        grid['next'][grid['end'][jy, jx]] = j
                    grid['end'][jy, jx] = j
                    grid['count'][jy, jx] += 1
                    try_count.append(num_tries)
                    num_tries = 0
                    done = True
                else:
                    num_tries += 1

            if num_tries >= max_tries:
                n = j - 1
                done = True

        if num_tries >= max_tries:
            break

    if n < num_points:
        x = x[0:n]
        y = y[0:n]

    return x, y, try_count
