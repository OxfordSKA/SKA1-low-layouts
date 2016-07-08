# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import math
from math import ceil
import time
import numpy
import sys


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


def log_spiral_clusters(r0, r1, b, n, n_cluster, r_cluster, min_sep,
                        cluster_timeout=2.0, tries_per_cluster=5, seed=None):
    """Computes coordinates on a log spiral.

    Args:
        r0 (float): minimum radius
        r1 (float): maximum radius
        b (float): Spiral constant.
        n (int): Number of points.
        n_cluster (int): Number of points per cluster
        r_cluster (double): Radius of the cluster.
        min_sep (double): minimum separation of points in each cluster.
        cluster_timeout (Optional[double]): timeout per cluster, in seconds
        tries_per_cluster (Optional[int]): number of seeds to try
        seed (Optional[int]): Random number seed.

    Returns:
        tuple: (x, y) coordinates
    """
    sys.stdout.flush()
    seed = numpy.random.randint(1, 1e5) if not seed else seed
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
    max_iter, max_time_taken, max_tries, max_total_tries = 0, 0.0, 0, 0
    for k in range(n):
        for t in range(tries_per_cluster):
            print('cluster %i try %i' % (k, t))
            sys.stdout.flush()
            numpy.random.seed(seed + t)
            xc, yc, tries, time_taken, total_tries = \
                rand_uniform_2d(n_cluster, r_cluster, min_sep,
                                timeout=cluster_timeout)
            max_iter = max(max_iter, tries)
            max_time_taken = max(max_time_taken, time_taken)
            max_total_tries = max(max_total_tries, total_tries)
            if xc.shape[0] == n_cluster:
                max_tries = max(max_tries, t)
                break
        if not xc.shape[0] == n_cluster:
            raise RuntimeError('Failed to generate cluster [%i]. '
                               '%i/%i stations generated. '
                               '(max tries for 1 point = %i, '
                               'max total tries = %i)'
                               % (k, xc.shape[0], n_cluster, max_iter,
                                  max_total_tries))
        x_all[k, :], y_all[k, :] = xc + x[k], yc + y[k]
    print('Finished [seed = %i, max_tries = %i,  max iter = %i, total iter = %i %.3f s]'
          % (seed + t, t, max_iter, max_total_tries, time_taken))
    sys.stdout.flush()
    return x_all.flatten(), y_all.flatten(), x, y


def grid_position(x, y, scale, r):
    ix = int(round(x + r) * scale)
    iy = int(round(y + r) * scale)
    return ix, iy


def get_trial_position(r):
    return tuple(numpy.random.rand(2) * 2.0 * r - r)


def rand_uniform_2d(n, r_max, min_sep, timeout, r_min=0.0, seed=None):
    """
    Generate 2d random points with a minimum separation within a radius
    range specified by r_max and r_min.

    Args:
        n (int): Number of points to generate.
        r_max (float):  Maximum radius.
        min_sep (float): Minimum separation of points.
        r_min (Optional[float]): Minimum radius.
        timeout (Optional[float]): timeout, in seconds.

    Returns:

    """
    if seed:
        numpy.random.seed(seed)
    grid_size = min(100, int(ceil(float(r_max * 2.0) / min_sep)))
    grid_cell = float(r_max * 2.0) / grid_size  # Grid sector size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.

    x, y = numpy.zeros(n), numpy.zeros(n)
    grid = {
        'start': numpy.zeros((grid_size, grid_size), dtype='i8'),
        'end': numpy.zeros((grid_size, grid_size), dtype='i8'),
        'count': numpy.zeros((grid_size, grid_size), dtype='i8'),
        'next': numpy.zeros(n, dtype='i8')
    }

    t0 = time.time()
    n_generated = n
    num_tries, max_tries, total_tries = 0, 0, 0
    for j in range(n):
        done = False
        while not done:
            xt, yt = get_trial_position(r_max)
            rt = (xt**2 + yt**2)**0.5
            # Point is inside area defined by: r_min < r < r_max
            if rt + (min_sep / 2.0) > r_max:
                num_tries += 1
            elif r_min and rt - (min_sep / 2.0) < r_min:
                num_tries += 1
            else:
                jx, jy = grid_position(xt, yt, scale, r_max)
                y0 = max(0, jy - 2)
                y1 = min(grid_size, jy + 3)
                x0 = max(0, jx - 2)
                x1 = min(grid_size, jx + 3)

                # Find minimum spacing between trial and other points.
                d_min = r_max * 2.0
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
                    x[j], y[j] = xt, yt
                    if grid['count'][jy, jx] == 0:
                        grid['start'][jy, jx] = j
                    else:
                        grid['next'][grid['end'][jy, jx]] = j
                    grid['end'][jy, jx] = j
                    grid['count'][jy, jx] += 1
                    max_tries = max(max_tries, num_tries)
                    total_tries += num_tries
                    num_tries = 0
                    done = True
                else:
                    num_tries += 1

            if (time.time() - t0) >= timeout:
                max_tries = max(max_tries, num_tries)
                total_tries += num_tries
                n_generated = j - 1
                done = True

        if (time.time() - t0) >= timeout:
            max_tries = max(max_tries, num_tries)
            total_tries += num_tries
            break

    if n_generated < n:
        x, y = x[:n_generated], y[:n_generated]

    return x, y, {'max_tries': max_tries,
                  'total_tries': total_tries,
                  'time_taken': time.time() - t0
                  }


def rand_uniform_2d_trials(n, r_max, min_sep, trial_timeout, num_trials=5,
                           seed=None, r_min=0.0, verbose=False):
    seed = numpy.random.randint(1, 1e8) if not seed else seed
    max_generated = 0
    all_info = dict()
    t0 = time.time()
    for t in range(num_trials):
        numpy.random.seed(seed + t)
        x, y, info = rand_uniform_2d(n, r_max, min_sep, trial_timeout, r_min)
        all_info[t] = info
        all_info[t]['seed'] = seed + t
        all_info[t]['num_generated'] = x.shape[0]
        if x.shape[0] == n:
            all_info['attempt_id'] = t
            all_info['total_time'] = time.time() - t0
            all_info['final_seed'] = seed + t
            if verbose and t > 0:
                print('%i' % x.shape[0])
            return x, y, all_info
        else:
            max_generated = max(max_generated, x.shape[0])
            if verbose:
                print('%i-' % x.shape[0], end='')
    if verbose:
        print('%i' % x.shape[0])
        sys.stdout.flush()
        for key in all_info:
            print(key, ':', all_info[key])
            sys.stdout.flush()

    raise RuntimeError('Failed to generate enough points. '
                       'max generated: %i / %i' % (max_generated, n))


def rand_uniform_2d_trials_r_max(n, r_max, min_sep, trial_timeout,
                                 num_trials=5, seed=None,
                                 r_min=0.0, verbose=False):
    if r_max.shape[0] == 0:
        raise AssertionError('rmax must be an array of r max values')

    seed = numpy.random.randint(1, 1e8) if not seed else seed
    max_generated = 0
    all_info = dict()
    t0 = time.time()
    for ir, r in enumerate(r_max):
        all_info[r] = dict()
        if verbose:
            print('(%-3i/%3i) %8.3f' % (ir, r_max.shape[0], r), end=': ')
        for t in range(num_trials):
            numpy.random.seed(seed + t)
            x, y, info = rand_uniform_2d(n, r, min_sep, trial_timeout, r_min)
            all_info[r][t] = info
            all_info[r][t]['seed'] = seed + t
            all_info[r][t]['num_generated'] = x.shape[0]
            if x.shape[0] == n:
                all_info['attempt_id'] = t
                all_info['total_time'] = time.time() - t0
                all_info['final_seed'] = seed + t
                all_info['final_radius'] = r
                all_info['final_radius_id'] = ir
                all_info['r_max'] = r_max
                if verbose:
                    print('%i' % x.shape[0])
                return x, y, all_info
            else:
                max_generated = max(max_generated, x.shape[0])
                if verbose:
                    print('%i%s' % (x.shape[0], ',' if t < num_trials-1 else ''),
                          end='')
        if verbose:
            print(' ')
    if verbose:
        print('%i' % x.shape[0])
        sys.stdout.flush()
        for key in all_info:
            print(key, ':', all_info[key])
            sys.stdout.flush()

    raise RuntimeError('Failed to generate enough points. '
                       'max generated: %i / %i' % (max_generated, n))


def generate_clusters(num_clusters, n, r_max, min_sep, trail_timeout,
                      num_trials, seed=None, r_min = 0.0, verbose=False):
    x_all, y_all, info_all = list(), list(), list()
    for i in range(num_clusters):
        if verbose:
            print('Cluster-%i' % i, n, r_max, min_sep)
        x, y, info = rand_uniform_2d_trials(n, r_max, min_sep, trail_timeout,
                                            num_trials, seed, r_min, verbose)
        if x.shape[0] != n:
            raise RuntimeError('Failed to generate cluster %i' % i)
        x_all.append(x)
        y_all.append(y)
        info_all.append(info)
    return x_all, y_all, info_all
