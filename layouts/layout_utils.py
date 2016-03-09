# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy
import matplotlib.pyplot as pyplot
from numpy.random import rand
from math import radians, floor
import math
try:
    from pyuvwsim import (load_station_coords, convert_enu_to_ecef,
                          evaluate_baseline_uvw)
    uvwsim_found = True
except ImportError:
    print('pyuvwsim not found, skipping uvw co-ordiante generation.')
    print('see: https://github.com/SKA-ScienceDataProcessor/uvwsim, pyuvwsim.rst')
    uvwsim_found = False


def rotate_coords(x, y, angle):
    """Rotate array of x, y coordinates counter clockwise by angle, in deg."""
    xr = x * numpy.cos(radians(angle)) - y * numpy.sin(radians(angle))
    yr = x * numpy.sin(radians(angle)) + y * numpy.cos(radians(angle))
    return xr, yr


def taylor_win(n, nbar, sll):
    """
    http://www.dsprelated.com/showcode/6.php

    from http://mathforum.org/kb/message.jspa?messageID=925929:

    A Taylor window is very similar to Chebychev weights. While Chebychev
    weights provide the tighest beamwidth for a given side-lobe level, taylor
    weights provide the least taper loss for a given sidelobe level.

    'Antenna Theory: Analysis and Design' by Constantine Balanis, 2nd
    edition, 1997, pp. 358-368, or 'Modern Antenna Design' by Thomas
    Milligan, 1985, pp.141-148.
    """
    def calculate_fm(m, sp2, a, nbar):
        n = numpy.arange(1, nbar)
        p = numpy.hstack([numpy.arange(1, m, dtype='f8'),
                          numpy.arange(m+1, nbar, dtype='f8')])
        num = numpy.prod((1 - (m**2/sp2) / (a**2 + (n - 0.5)**2)))
        den = numpy.prod(1.0 - m**2 / p**2)
        fm = ((-1)**(m + 1) * num) / (2.0 * den)
        return fm
    a = numpy.arccosh(10.0**(-sll/20.0))/numpy.pi
    sp2 = nbar**2 / (a**2 + (nbar - 0.5)**2)
    w = numpy.ones(n)
    fm = numpy.zeros(nbar)
    summation = 0
    k = numpy.arange(n)
    xi = (k - 0.5 * n + 0.5) / n
    for m in range(1, nbar):
        fm[m] = calculate_fm(m, sp2, a, nbar)
        summation = fm[m] * numpy.cos(2.0 * numpy.pi * m * xi) + summation
    w += 2.0 * summation
    return w


def gridgen_taylor(num_points, diameter, min_dist, sll=-28, n_miss_max=1000):
    """Generate uniform random positions within a specified diameter which
    are no closer than a specified minimum distance.

    Uses and algorithm where the area is split into a grid sectors
    so that when checking for minimum distance, only nearby points need to be
    considered.
    """

    def get_trail_position(r):
        x = -r + 2.0 * r * rand()
        y = -r + 2.0 * r * rand()
        return x, y

    def grid_position(x, y, scale, r):
        jx = int(floor((x + r) * scale))
        jy = int(floor((y + r) * scale))
        return jx, jy

    # Fix seed to study closest match fails (with fixed seed can
    # print problematic indices)
    # seed(2)

    r = diameter / 2.0  # Radius

    # Initialise taylor taper.
    nbar = int(numpy.ceil(2.0 * (numpy.arccosh(10**(-sll / 20.0)) /
                                 numpy.pi)**2 + 0.5))
    n_taylor = 10000
    w_taylor = taylor_win(n_taylor + 1, nbar, sll)
    w_taylor /= w_taylor.max()
    w_taylor = w_taylor[n_taylor/2:]
    r_taylor = numpy.arange(w_taylor.shape[0]) * (diameter / (n_taylor + 1))
    n_taylor = w_taylor.shape[0]

    p = 1.0 / w_taylor[-1]
    max_dist = p * min_dist

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / max_dist)))
    grid_size += grid_size % 2
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 1

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
    miss_count = []
    j = 0
    space_remaining = True
    while space_remaining:
        done = False
        while not done:
            # Generate a trail position
            xt, yt = get_trail_position(r)
            rt = (xt**2 + yt**2)**0.5

            # Check if the point is inside the diameter.
            if rt + min_dist / 2.0 > r:
                num_miss += 1

            # Check if min distance is met.
            else:
                iw = int(round((rt / r) * n_taylor))
                ant_r = min_dist / (2.0 * w_taylor[iw])

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
                                iw = int(round(r_other / r * n_taylor))
                                ant_r_other = min_dist / (2.0 * w_taylor[iw])

                                if dr - ant_r_other <= dmin:
                                    dmin = dr - ant_r_other
                                i_other = grid_next[i_other]

                iw = int(round(rt / r * n_taylor))
                scaled_min_dist_3 = (min_dist / 2.0) / w_taylor[iw]

                if dmin >= scaled_min_dist_3:
                    x[j] = xt
                    y[j] = yt

                    if grid_count[jx, jy] == 0:
                        grid_i_start[jx, jy] = j
                    else:
                        grid_next[grid_i_end[jx, jy]] = j
                    grid_i_end[jx, jy] = j
                    grid_count[jx, jy] += 1
                    miss_count.append(num_miss)
                    max_num_miss = max(max_num_miss, num_miss)
                    num_miss = 0
                    done = True
                    j += 1
                else:
                    num_miss += 1

            if num_miss >= n_miss_max:
                n = j - 1
                done = True

        if num_miss >= n_miss_max or j >= num_points:
            max_num_miss = max(max_num_miss, num_miss)
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    return x, y, miss_count  # , w_taylor, r_taylor, n_taylor


def gridgen_taylor_padded(num_points, diameter, inner_diameter, min_dist,
                          sll=-28, n_miss_max=1000):
    """Generate uniform random positions within a specified diameter which
    are no closer than a specified minimum distance.

    Uses and algorithm where the area is split into a grid sectors
    so that when checking for minimum distance, only nearby points need to be
    considered.
    """

    def get_trail_position(r):
        x = -r + 2.0 * r * rand()
        y = -r + 2.0 * r * rand()
        return x, y

    def grid_position(x, y, scale, r):
        jx = int(floor((x + r) * scale))
        jy = int(floor((y + r) * scale))
        return jx, jy

    # Fix seed to study closest match fails (with fixed seed can
    # print problematic indices)
    # seed(2)

    r = diameter / 2.0  # Radius
    weights, r_weights = generate_padded_taylor_weights(diameter,
                                                        inner_diameter, sll)
    n_weights = weights.shape[0]

    p = 1.0 / weights[-1]
    max_dist = p * min_dist

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / max_dist)))
    grid_size += grid_size % 2
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 1

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
    miss_count = []
    j = 0
    space_remaining = True
    while space_remaining:
        done = False
        while not done:
            # Generate a trail position
            xt, yt = get_trail_position(r)
            rt = (xt**2 + yt**2)**0.5

            # Check if the point is inside the diameter.
            if rt + min_dist / 2.0 > r:
                num_miss += 1

            # Check if min distance is met.
            else:
                iw = int(round((rt / r) * n_weights))
                ant_r = min_dist / (2.0 * weights[iw])

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
                                iw = int(round(r_other / r * n_weights))
                                ant_r_other = min_dist / (2.0 * weights[iw])

                                if dr - ant_r_other <= dmin:
                                    dmin = dr - ant_r_other
                                i_other = grid_next[i_other]

                iw = int(round(rt / r * n_weights))
                scaled_min_dist_3 = (min_dist / 2.0) / weights[iw]

                if dmin >= scaled_min_dist_3:
                    x[j] = xt
                    y[j] = yt

                    if grid_count[jx, jy] == 0:
                        grid_i_start[jx, jy] = j
                    else:
                        grid_next[grid_i_end[jx, jy]] = j
                    grid_i_end[jx, jy] = j
                    grid_count[jx, jy] += 1
                    miss_count.append(num_miss)
                    max_num_miss = max(max_num_miss, num_miss)
                    num_miss = 0
                    done = True
                    j += 1
                else:
                    num_miss += 1

            if num_miss >= n_miss_max:
                n = j - 1
                done = True

        if num_miss >= n_miss_max or j >= num_points:
            max_num_miss = max(max_num_miss, num_miss)
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    return x, y, numpy.array(miss_count, dtype='i8'), weights, r_weights


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
    check_width = 2

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


def generate_baseline_uvw(x, y, z, ra_rad, dec_rad, num_times, num_baselines,
                          mjd_start, dt_s):
    """Generate baseline coordinates from ecef station coordinates."""
    num_coords = num_times * num_baselines
    uu = numpy.zeros(num_coords, dtype='f8')
    vv = numpy.zeros(num_coords, dtype='f8')
    ww = numpy.zeros(num_coords, dtype='f8')
    for i in range(num_times):
        t = i * dt_s + dt_s / 2.0
        mjd = mjd_start + (t / 86400.0)
        i0 = i * num_baselines
        i1 = i0 + num_baselines
        uu_, vv_, ww_ = evaluate_baseline_uvw(x, y, z, ra_rad, dec_rad, mjd)
        uu[i0:i1] = uu_
        vv[i0:i1] = vv_
        ww[i0:i1] = ww_
    return uu, vv, ww


def plot_hist(uu, vv, file_name, title):
    uv_dist = (uu**2 + vv**2)**0.5
    uv_dist_range = uv_dist.max() - uv_dist.min()
    bin_width_m = 10.0
    num_bins = numpy.ceil(uv_dist_range / bin_width_m)
    bin_inc = uv_dist_range / num_bins
    print('total coords:', uv_dist.shape[0])
    print('min:', uv_dist.min())
    print('max:', uv_dist.max())
    print('mean:', numpy.mean(uv_dist))
    print('median:', numpy.median(uv_dist))

    # bins = numpy.arange(num_bins) * bin_inc + uv_dist.min()
    bins = numpy.logspace(numpy.log10(uv_dist.min()),
                          numpy.log10(uv_dist.max()), 50)
    # bins = numpy.logspace(0, numpy.log10(uv_dist.max()), 20)
    # bins = numpy.logspace(0, numpy.ceil(numpy.log10(uv_dist.max())), 20)
    hist, bin_edges = numpy.histogram(uv_dist, bins=bins, density=False)
    hist = numpy.array(hist, dtype='f8')
    hist /= numpy.diff(bin_edges)
    x = bin_edges[:-1] + numpy.diff(bin_edges) / 2.0
    fig = pyplot.figure(figsize=(14, 6))
    ax = fig.add_subplot(121)
    ax.grid()
    ax.loglog(x, hist, '-')
    ax.set_xlim(10.0, 1.0e5)
    ax.set_xlabel('Radius (m)', fontsize='small')
    ax.set_ylabel('Visibility density', fontsize='small')
    ax.set_title(title)

    ax = fig.add_subplot(122)
    uv_dist_range = uv_dist.max() - uv_dist.min()
    bin_width_m = 10.0
    num_bins = numpy.ceil(uv_dist_range / bin_width_m)
    bin_inc = uv_dist_range / num_bins
    bins = numpy.arange(num_bins) * bin_inc + uv_dist.min()
    hist, bin_edges = numpy.histogram(uv_dist, bins=bins, density=True)
    hist = numpy.array(hist, dtype='f8')
    hist *= numpy.diff(bin_edges)
    # hist *= float(uv_dist.shape[0])
    cum_hist = numpy.cumsum(hist)
    x = bin_edges[:-1] + numpy.diff(bin_edges) / 2.0
    ax.semilogx(x, cum_hist, '-')
    ax.grid()
    median = numpy.median(uv_dist)
    # ax.set_xlim(20.0, 70.0e3)
    ax.set_xlim(10.0, 1.0e5)
    ax.set_ylim(0.0, 1.02)
    ax.plot([median, median], ax.get_ylim(), 'r--')
    ax.text(0.05, 0.95, 'median = %.2f km' % (median / 1.0e3),
            ha='left', va='center', style='italic', color='r',
            transform=ax.transAxes, fontsize='small')
    ax.text(0.05, 0.90, 'min = %.2f m' % (uv_dist.min()),
            ha='left', va='center', style='italic', color='r',
            transform=ax.transAxes, fontsize='small')
    ax.text(0.05, 0.85, 'max = %.2f km' % (uv_dist.max() / 1.0e3),
            ha='left', va='center', style='italic', color='r',
            transform=ax.transAxes, fontsize='small')
    ax.set_xlabel('Radius (m)', fontsize='small')
    ax.set_ylabel('Cumulative visibility number', fontsize='small')
    ax.set_title(title)
    pyplot.savefig(file_name)
    pyplot.close(fig)


def plot_uv_dist(uu, vv, station_radius_m, file_root):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    circle = pyplot.Circle((0.0, 0.0), station_radius_m, color='r',
                           fill=False, alpha=0.5, linewidth=1.0)
    ax.add_artist(circle)
    ax.plot(uu, vv, 'k.', alpha=0.1, ms=2.0)
    ax.plot(-uu, -vv, 'k.', alpha=0.1, ms=2.0)
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    ax.set_title(file_root)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    pyplot.savefig(file_root + '_00.2km.png')
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    pyplot.savefig(file_root + '_00.5km.png')
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    pyplot.savefig(file_root + '_01.0km.png')
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(file_root + '_03.0km.png')
    ax.set_xlim(-70000, 70000)
    ax.set_ylim(-70000, 70000)
    pyplot.savefig(file_root + '_70.0km.png')
    pyplot.close(fig)


def plot_uv_grid_image(uu, vv, cell_size, file_name):
    uu = numpy.hstack((uu, -uu))
    vv = numpy.hstack((vv, -vv))
    # uv_max = max(numpy.abs(uu).max(), numpy.abs(vv).max())
    uv_max = 1500.0
    print('uv_max:', uv_max)
    num_cells = int(math.ceil(uv_max / cell_size)) + 10
    num_cells += num_cells % 2
    num_cells *= 2
    print('num_cells:', num_cells)
    uv_image = numpy.zeros((num_cells, num_cells))
    centre = num_cells / 2
    for i in range(uu.shape[0]):
        if math.fabs(uu[i]) >= uv_max or math.fabs(vv[i]) >= uv_max:
            continue
        x = int(round(uu[i] / cell_size)) + centre
        y = int(round(vv[i] / cell_size)) + centre
        uv_image[y, x] += 1

    extent = [-centre * cell_size, centre * cell_size,
              -centre * cell_size, centre * cell_size]
    extent = numpy.array(extent) - cell_size / 2.0
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    im = ax.imshow(uv_image, interpolation='nearest', extent=extent)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    pyplot.savefig(file_name + '_0500m.png')
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    pyplot.savefig(file_name + '_1000m.png')
    # pyplot.show()


# def uvwsim(st_x, st_y, out_dir, telescope_name):
#     print('generating uv coords...')
#     x = v4d_st_x
#     y = v4d_st_y
#     num_stations = x.shape[0]
#     z = numpy.zeros_like(x)
#     lon = radians(116.63128900)
#     lat = radians(-26.69702400)
#     alt = 0.0
#     ra = radians(68.698903779331502)
#     dec = radians(-26.568851215532160)
#     mjd_mid = 57443.4375000000
#     obs_length = 6.0 * 3600.0  # seconds
#     num_times = int(obs_length / (3 * 60.0))
#     print(num_times)
#     dt_s = obs_length / float(num_times)
#     mjd_start = mjd_mid - obs_length / (3600.0 * 24.0)
#     num_baselines = num_stations * (num_stations - 1) / 2
#     x, y, z = convert_enu_to_ecef(x, y, z, lon, lat, alt)
#     uu, vv, ww = generate_baseline_uvw(x, y, z, ra, dec, num_times,
#                                        num_baselines, mjd_start,
#                                        dt_s)
#     layout_utils.plot_hist(uu, vv, join(out_dir, '%s_hist.png' % telescope_name),
#                            '%s snapshot-uv' % telescope_name)
#     layout_utils.plot_uv_dist(uu, vv,
#                               join(out_dir, '%s_snapshot_uv_zenith' % telescope_name))


def round_to_odd(x):
    return int(math.floor(x) // 2 * 2 + 1)


def round_to_even(x):
    return round(x / 2.0) * 2


def generate_padded_taylor_weights(diameter, inner_diameter, sll, n=10001):
    n_inner = round_to_even((inner_diameter / diameter) * n)
    n_taylor = n - n_inner
    nbar = int(numpy.ceil(2.0 * (numpy.arccosh(10**(-sll / 20.0)) /
                                 numpy.pi)**2 + 0.5))
    w_taylor = taylor_win(n_taylor, nbar, sll)
    w_taylor /= w_taylor.max()
    w_taylor = w_taylor[n_taylor/2:]
    weights = numpy.ones(n/2 + 1)
    weights[n_inner/2:] = w_taylor
    r_weights = numpy.linspace(0, diameter/2.0, weights.shape[0])
    return weights, r_weights


if __name__ == '__main__':
    sll = -28
    diameter = 35.0
    inner_diameter = 10.0
    n = 101
    w, r = generate_padded_taylor_weights(diameter, inner_diameter, sll, n)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(r, w, '-')
    ax.plot([diameter / 2.0, diameter / 2.0], ax.set_ylim())
    ax.plot([inner_diameter / 2.0, inner_diameter / 2.0], ax.set_ylim())
    ax.set_ylim(0, 1.1)
    ax.grid()
    pyplot.show()
