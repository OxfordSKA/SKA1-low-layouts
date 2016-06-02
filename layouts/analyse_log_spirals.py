# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy
import oskar
import pyuvwsim
from matplotlib import ticker
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# TODO(BM) single log-spiral rotated
# TODO(BM) tapered random generator
# TODO(BM) add beam convolution
# TODO(BM) Metrics (see apple notes, ALMA paper etc)


def uv_cell_size_to_fov(cell_size_uv_wavelengths, size):
    cell_size_lm = 1.0 / (size * cell_size_uv_wavelengths)
    lm_max = (size * math.sin(cell_size_lm)) / 2.0
    theta_max_rad = math.asin(lm_max)
    return math.degrees(theta_max_rad) * 2.0


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
        xc, yc, _ = gridgen(n_cluster, r_cluster, min_sep, max_tries=10000)
        if not xc.shape[0] == n_cluster:
            raise RuntimeError('Failed to generate cluster [%i] %i/%i stations '
                               'generated.' % (k, xc.shape[0], n_cluster))
        x_all[k, :] = xc + x[k]
        y_all[k, :] = yc + y[k]
    x_all = x_all.flatten()
    y_all = y_all.flatten()
    return x_all, y_all, x, y


def gridgen(num_points, r1, min_sep, r0=0.0, max_tries=1000):
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


def add_observation(obs_length_s, obs_list, duty_cycle_s=3.0*60.0):
    if obs_length_s == 0.0:
        num_times = 1
        interval_s = 0.0
    else:
        num_times = int(obs_length_s / duty_cycle_s)
        interval_s = obs_length_s / float(num_times)
    obs_list.append({'num_times': num_times,
                     'obs_length_s': obs_length_s,
                     'interval_s': interval_s})
    return obs_list


def generate_uv_coords(x_ecef, y_ecef, z_ecef, settings, obs):
    wavelength_m = 299792458.0 / settings['freq_hz']
    mjd_start = settings['mjd_mid'] - (obs['obs_length_s'] / 2.0) / 86400.0
    num_stations = x_ecef.shape[0]
    num_baselines = num_stations * (num_stations - 1) / 2
    num_coords = obs['num_times'] * num_baselines
    uu = numpy.zeros(num_coords, dtype='f4')
    vv = numpy.zeros_like(uu, dtype='f4')
    ww = numpy.zeros_like(uu, dtype='f4')
    for i in range(obs['num_times']):
        mjd = mjd_start + (i * obs['interval_s'] +
                           obs['interval_s'] / 2.0) / 86400.0
        uu_, vv_, ww_ = pyuvwsim.evaluate_baseline_uvw(x_ecef, y_ecef, z_ecef,
                                                       settings['ra'],
                                                       settings['dec'], mjd)
        uu[i * num_baselines: (i + 1) * num_baselines] = uu_ / wavelength_m
        vv[i * num_baselines: (i + 1) * num_baselines] = vv_ / wavelength_m
        ww[i * num_baselines: (i + 1) * num_baselines] = ww_ / wavelength_m
    return uu, vv, ww


def generate_uv_grid(uu, vv, ww, x, y, settings):
    wavelength_m = 299792458.0 / settings['freq_hz']
    r = (x**2 + y**2)**0.5
    uv_cell_size_m = (r.max() * 4.1) / settings['uv_grid_size']
    uv_cell_size_wavelengths = uv_cell_size_m / wavelength_m
    fov_deg = uv_cell_size_to_fov(uv_cell_size_wavelengths,
                                  settings['uv_grid_size'])
    im = oskar.imager.Imager('Single')
    im.set_fft_on_gpu(False)
    im.set_fov(fov_deg)
    im.set_size(settings['uv_grid_size'])
    im.set_grid_kernel('Pillbox', 1, 1)
    uv_grid = numpy.zeros((settings['uv_grid_size'], settings['uv_grid_size']),
                          dtype='c8')
    im.update_plane(uu, vv, ww, numpy.ones_like(uu, dtype='c8'),
                    numpy.ones_like(uu), uv_grid, 0.0)
    im.update_plane(-uu, -vv, -ww, numpy.ones_like(uu, dtype='c8'),
                    numpy.ones_like(uu), uv_grid, 0.0)
    return uv_grid, uv_cell_size_wavelengths, fov_deg


def generate_psf(uu, vv, ww, settings):
    tx = time.time()
    im = oskar.imager.Imager('Single')
    im.set_grid_kernel('Spheroidal', 3, 100)
    im.set_fft_on_gpu(False)
    psf = im.make_image(uu, vv, ww, numpy.ones_like(uu, dtype='c16'),
                        numpy.ones_like(uu), settings['psf_fov_deg'],
                        settings['psf_im_size'])
    print('psf (gen): %.2fs, ' % (time.time() - tx), end=' ')
    psf_cell_size = im.fov_to_cellsize(
        math.radians(settings['psf_fov_deg']), settings['psf_im_size'])
    l = numpy.arange(-settings['psf_im_size'] / 2,
                     settings['psf_im_size'] / 2) * psf_cell_size
    l, m = numpy.meshgrid(l, l[::-1])
    r_lm = (l ** 2 + m ** 2) ** 0.5
    r_lm = r_lm.flatten()
    sort_idx = numpy.argsort(r_lm)
    r_lm = r_lm[sort_idx]
    psf_1d = psf.flatten()
    psf_1d = psf_1d[sort_idx]
    # psf_1d = numpy.abs(psf_1d)
    return psf, psf_cell_size, psf_1d, r_lm


def bin_psf_1d(psf, psf_cell_size, r_lm, psf_az_1d, num_bins=100):
    bin_inc = (psf.shape[0] / 2) * psf_cell_size / float(num_bins)
    psf_bins = {'r': numpy.zeros(num_bins), 'mean': numpy.zeros(num_bins),
                'std': numpy.zeros(num_bins), 'min': numpy.zeros(num_bins),
                'max': numpy.zeros(num_bins),}

    # FIXME(BM) replace manual bins with numpy.digitize?
    # http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy

    for i in range(num_bins):
        r0 = i * bin_inc
        r1 = r0 + bin_inc
        idx = numpy.where(numpy.logical_and(r_lm > r0, r_lm <= r1))
        y_bin = psf_az_1d[idx]
        psf_bins['r'][i] = r0 + (r1 - r0) / 2.0
        psf_bins['mean'][i] = numpy.mean(y_bin)
        psf_bins['std'][i] = numpy.std(y_bin)
        psf_bins['min'][i] = numpy.min(y_bin)
        psf_bins['max'][i] = numpy.max(y_bin)
    return psf_bins


def analyse_layout(layout, settings):
    observations = list()
    add_observation(0.0, observations)  # snapshot
    add_observation(4.0 * 3600.0, observations)  # 4h observation

    # Combine components to a single set of station coordinates.
    x, y = list(), list()
    for coords in layout:
        x.extend(coords['x'])
        y.extend(coords['y'])
    x, y, z = numpy.array(x), numpy.array(y), numpy.zeros_like(x)
    ecef_x, ecef_y, ecef_z = pyuvwsim.convert_enu_to_ecef(x, y, z,
                                                          settings['lon'],
                                                          settings['lat'])
    # Generate results
    results = list()
    for i_obs, obs in enumerate(observations):
        t0 = time.time()
        print('- Analysing observation %i, no. times = %i ... ' %
              (i_obs, obs['num_times']), end=' ')

        # Generate uvw coordinates (in wavelengths).
        uu, vv, ww = generate_uv_coords(ecef_x, ecef_y, ecef_z, settings, obs)

        # Generate uv grid image
        t1 = time.time()
        uv_grid, uv_cell_size_wavelengths, fov_deg = \
            generate_uv_grid(uu, vv, ww, x, y, settings)
        print('uv_grid: %.2fs,' % (time.time() - t1), end=' ')

        # TODO(BM) convolve with beam (use gaussian or lookup table)

        # Make the PSF
        t1 = time.time()
        psf, psf_cell_size, psf_az_1d, r_lm = generate_psf(uu, vv, ww, settings)
        print('psf: %.2fs,' % (time.time() - t1), end=' ')

        # Bin the 1d psf
        t1 = time.time()
        psf_bins = bin_psf_1d(psf, psf_cell_size, r_lm, psf_az_1d, num_bins=200)
        print('bin_psf: %.2fs,' % (time.time() - t1), end=' ')

        # TODO(BM) Metrics UVGAP, PSFRMS etc

        results.append({
            'uu': uu,
            'vv': vv,
            'uv_grid': uv_grid,
            'uv_grid_fov': fov_deg,
            'uv_grid_cell_size_wavelengths': uv_cell_size_wavelengths,
            'psf': psf,
            'psf_r_lm': r_lm,
            'psf_az_1d': psf_az_1d,
            'psf_cell_size': psf_cell_size,
            'psf_bins': psf_bins,
            'obs': obs,
            'settings': settings
        })
        print('total: %.2f s' % (time.time() - t0))

    return results


def plot_layout(ax, layout, settings, r_min, r_max, fontsize):
    for coords in layout:
        x = coords['x'] / 1.0e3
        y = coords['y'] / 1.0e3
        for i in range(x.shape[0]):
            c = plt.Circle((x[i], y[i]), settings['station_r'] / 1.0e3,
                           fill=True, color=coords['color'], alpha=0.5)
            ax.add_artist(c)
        if 'type' in coords and coords['type'] == 'cluster':
            cx = coords['cluster_x'] / 1.0e3
            cy = coords['cluster_y'] / 1.0e3
            cluster_r = coords['cluster_diameter'] / (2.0 * 1.0e3)
            for j in range(cx.shape[0]):
                c = plt.Circle((cx[j], cy[j]), cluster_r,
                               fill=False, color='b', alpha=0.5)
                ax.add_artist(c)
    c = plt.Circle((0.0, 0.0), r_min / 1.0e3, fill=False, color='r',
                   linestyle=':')
    ax.add_artist(c)
    c = plt.Circle((0.0, 0.0), r_max / 1.0e3, fill=False, color='r',
                   linestyle=':')
    ax.add_artist(c)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax.locator_params(axis='both', nbins=5)
    ax.set_xlabel('East (kilometres)', fontsize=fontsize)
    ax.set_ylabel('North (kilometres)', fontsize=fontsize)
    ax.set_title('Station positions (n=%i)' % x.shape[0], fontsize=fontsize)
    ax.grid()


def plot_uv_scatter(ax, result, settings, r_min, r_max, fontsize):
    wavelength_m = 299792458.0 / settings['freq_hz']
    if result['obs']['num_times'] == 1:
        alpha = 0.2
    else:
        alpha = max(0.002, 0.2 / (result['obs']['num_times']))
    uu = result['uu'] * wavelength_m
    vv = result['vv'] * wavelength_m
    ax.plot(uu/ 1.0e3, vv / 1.0e3, '.', color='k', ms=2.0, alpha=alpha)
    ax.plot(-uu / 1.0e3, -vv / 1.0e3, '.', color='k', ms=2.0, alpha=alpha)
    c = plt.Circle((0.0, 0.0), r_max * 2.0 / 1.0e3, fill=False, color='r',
                   linestyle='-', linewidth=2, alpha=0.5)
    ax.add_artist(c)
    ax.set_xlim(r_max * 2.0 / 1.0e3, -r_max * 2.0 / 1.0e3)
    ax.set_ylim(-r_max * 2.0 / 1.0e3, r_max * 2.0 / 1.0e3)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax.set_xlabel('uu (kilometres)', fontsize=fontsize)
    ax.set_ylabel('vv (kilometres)', fontsize=fontsize)
    ax.set_title('Baseline coordinates (%.2f h, %i samples)' %
                 (result['obs']['obs_length_s'] / 3600.0,
                  result['obs']['num_times']), fontsize=fontsize)


def plot_psf(ax, result, settings, fontsize, vmin=-0.03, linthresh=0.1):
    c = settings['psf_im_size'] / 2
    dx = result['psf_cell_size']
    extent = [(c + 0.5) * dx, (-c + 0.5) * dx,
              (-c - 0.5) * dx, (c - 0.5) * dx]
    im_ = ax.imshow(result['psf'], interpolation='nearest',
                    cmap='inferno',
                    origin='lower', norm=SymLogNorm(vmin=vmin, vmax=1.0,
                                                    linthresh=linthresh,
                                                    linscale=1.0),
                    extent=extent)

    # Color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.03)
    cbar = ax.figure.colorbar(im_, cax=cax, format='%.1f')
    cbar.ax.tick_params(labelsize='smaller')
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    # cbar.set_label('PSF amplitude', fontsize=fontsize)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    # ax.set_xlabel('l direction cosine', fontsize=fontsize)
    # ax.set_ylabel('m direction cosine', fontsize=fontsize)
    # ax.tick_params(axis='both', which='both', bottom='off', left='off',
    #                top='off', labelbottom='off', labelleft='off')
    ax.tick_params(axis='both', which='major', labelsize='smaller')
    ax.tick_params(axis='both', which='minor', labelsize='smaller')
    # ax.set_xticks(numpy.linspace(extent[0], extent[1], 3))
    # ax.set_yticks(numpy.linspace(extent[0], extent[1], 3))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_title('PSF image (FoV %.1f degrees, %.2f h)' %
                 (settings['psf_fov_deg'],
                  result['obs']['obs_length_s'] / 3600.0), fontsize=fontsize)


def plot_psf_1d(ax, result, settings, fontsize, vmin=-0.03, linthresh=0.1):
    r_lm = result['psf_r_lm']
    psf_1d = result['psf_az_1d']
    ax.plot(r_lm, psf_1d, 'k.', ms=2.0, alpha=0.05)
    ax.set_ylim(vmin, linthresh)
    ax.set_xlim(0, (settings['psf_im_size'] / 2) * result['psf_cell_size'])
    ax.grid(True)
    ax.locator_params(axis='x', nbins=6)
    ax.set_xlabel('PSF radius', fontsize=fontsize)
    ax.set_ylabel('PSF amplitude', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plot_psf_1d_2(ax, result, settings, nbins, fontsize, vmin, linthresh):
    x = result['psf_r_lm']
    y = numpy.abs(result['psf_az_1d'])
    n, _ = numpy.histogram(x, bins=nbins)
    sy, _ = numpy.histogram(x, bins=nbins, weights=y)
    sy2, _ = numpy.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = numpy.sqrt(sy2 / n - mean**2)
    ax.plot(x, y, '.', markersize=1.5, alpha=0.1, color='k')
    x = (_[1:] + _[:-1]) / 2
    ax.plot(x, mean, '-', color='r', linewidth=1.0)
    # ax.plot(x, mean + std, '-', color='b')
    # ax.plot(x, mean - std, '-', color='b')
    # ax.fill_between(x, mean - std, mean + std, edgecolor='b', facecolor='b',
    #                 alpha=0.4)
    # ax.set_ylim(vmin, linthresh)
    ax.set_ylim(1e-3, 1.02)
    ax.set_xlim(0.0001, (settings['psf_im_size'] / 2) * result['psf_cell_size'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    ax.set_xlabel('PSF radius', fontsize=fontsize)
    ax.set_ylabel('PSF amplitude', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plot_psf_1d_bins(ax, result, settings, fontsize, vmin=-0.03, linthresh=0.1):
    num_antennas = 72 * result['obs']['num_times']  # FIXME(BM) get num antennas
    x = result['psf_bins']['r']
    y1 = result['psf_bins']['min']
    y2 = result['psf_bins']['max']
    ax.plot(x, y1, '-', color='k', linewidth=0.5)
    ax.plot(x, y2, '-', color='k', linewidth=0.5)
    ax.fill_between(x, y1, y2, edgecolor='k', facecolor='k', alpha=0.2)
    y = result['psf_bins']['mean']
    y_std = result['psf_bins']['std']
    ax.plot(x, y, '-', color='r')
    ax.fill_between(x, y - y_std, y + y_std, edgecolor='b', facecolor='b',
                    alpha=0.2)
    ax.plot(x, y + y_std, '-', color='b', linewidth=0.5)
    ax.plot(x, y - y_std, '-', color='b', linewidth=0.5)
    yy = 1.0 / num_antennas
    ax.plot(ax.get_xlim(), [yy, yy], 'g--')
    yy = -1.0 / float(num_antennas - 1)
    ax.plot(ax.get_xlim(), [yy, yy], 'g--')

    ax.set_ylim(vmin, linthresh)
    ax.set_xlim(0, (settings['psf_im_size'] / 2) * result['psf_cell_size'])
    # ax.set_yscale('symlog', linthresh=0.01)
    # ax.set_xscale('symlog', linthresh=0.2)
    # ax.set_xscale('log')
    # ax.locator_params(axis='x', nbins=6)  # Not possible with log axes

    ax.grid(True)
    ax.set_xlabel('PSF radius', fontsize=fontsize)
    ax.set_ylabel('PSF amplitude', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plot_psf_histogram(ax, result, settings, fontsize, y_max):
    n, _ = numpy.histogram(numpy.abs(result['psf'].flatten()),
                           bins=1000, density=True)
    x = (_[1:] + _[:-1])/2
    g = numpy.zeros_like(x)
    print(n.max(), result['obs']['num_times'])
    N = 72.0 * result['obs']['num_times']  # N = number of antennas?
    g = N * numpy.exp(-N * x)
    ax.plot(x, n, '-', color='b')
    ax.plot(x, g, '--', color='r')
    ax.set_xlim(0.01, y_max)
    if result['obs']['num_times'] == 1:
        ax.set_ylim(1.0e-3, N)
    ax.set_yscale('log', nonposy='clip')
    ax.grid(True)
    ax.set_xlabel('sidelobe magitude', fontsize=fontsize)
    ax.set_ylabel('number of samples', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plot_uv_hist(ax, result, settings, r_min, r_max, fontsize):
    wavelength_m = 299792458.0 / settings['freq_hz']
    uu = result['uu'] * wavelength_m
    vv = result['vv'] * wavelength_m
    uv_dist = (uu**2 + vv**2)**0.5
    uv_dist.sort()
    num_bins = 200
    # uv_dist_max = max(numpy.abs(uu.max()), numpy.abs(vv.max()))
    uv_dist_min = uv_dist.min()
    uv_dist_max = r_max * 2.0
    bin_width = uv_dist_max / float(num_bins)
    bins = numpy.arange(num_bins) * bin_width
    y, edges = numpy.histogram(uv_dist, bins=bins, density=True)
    dx = numpy.diff(edges)
    x = edges[:-1]
    # ax.bar(x, y, width=dx, color='0.6', fill=True, alpha=0.5, lw=1.5,
    #        edgecolor='k')
    ax.plot(x, y, 'k-', lw=1.0)
    ax.plot([r_max * 2.0, r_max * 2.0], ax.get_ylim(), '-', color='r', lw=2.0,
            alpha=0.5)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plot_uv_hist_2(ax, result, settings, r_min, r_max, fontsize):
    wavelength_m = 299792458.0 / settings['freq_hz']
    uu = result['uu'] * wavelength_m
    vv = result['vv'] * wavelength_m
    uv_dist = (uu**2 + vv**2)**0.5
    uv_dist.sort()
    num_bins = 200
    # uv_dist_max = max(numpy.abs(uu.max()), numpy.abs(vv.max()))
    uv_dist_min = uv_dist.min()
    uv_dist_max = r_max * 2.0
    bin_width = uv_dist_max / float(num_bins)
    # bins = numpy.arange(num_bins) * bin_width
    bins = numpy.logspace(math.log10(uv_dist_min), math.log10(uv_dist_max),
                          num_bins)
    y, edges = numpy.histogram(uv_dist, bins=bins, density=True)
    dx = numpy.diff(edges)
    x = edges[:-1]
    # ax.bar(x, y, width=dx, color='0.6', fill=True, alpha=0.5, lw=1.5,
    #        edgecolor='k')
    ax.plot([r_max * 2.0, r_max * 2.0], ax.get_ylim(), '-', color='r', lw=2.0,
            alpha=0.5)
    ax.plot(x, y, 'k-', lw=1.0)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_xscale('log', nonposx='clip')
    ax.set_yscale('log', nonposx='clip')
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plotting(layout, results, settings, save_root_name, r_min=500.0,
             r_max=1700.0):
    # FIXME(BM) generate multi page plot (or multiple plots)

    t0 = time.time()
    nx, ny = 3, 4
    #
    # 1   2  3
    # 4   5  6
    # 7   8  9
    # 10 11 12
    #

    vmin_0h = -0.03
    linthresh_0h = 0.2

    vmin_4h = -0.015
    linthresh_4h = 0.05

    fontsize_ = 'small'
    # fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig = plt.figure(figsize=(11.69, 16.53))  # A3

    fig.subplots_adjust(left=0.1, bottom=0.08, right=0.95, top=0.92,
                        hspace=0.2, wspace=0.2)

    # ------ Stations --------------
    plot_layout(fig.add_subplot(ny, nx, 1, aspect='equal'), layout,
                settings, r_min, r_max, fontsize_)

    # UV scatter
    plot_uv_scatter(fig.add_subplot(ny, nx, 2, aspect='equal'),
                    results[0], settings, r_min, r_max, fontsize_)
    plot_uv_scatter(fig.add_subplot(ny, nx, 3, aspect='equal'),
                    results[1], settings, r_min, r_max, fontsize_)

    nx, ny = 2, 4
    # 1   2
    # 3   4
    # 5   6
    # 7   8


    # UV histogram
    plot_uv_hist(fig.add_subplot(ny, nx, 3), results[0], settings, r_min, r_max,
                 fontsize_)
    plot_uv_hist(fig.add_subplot(ny, nx, 4), results[1], settings, r_min, r_max,
                 fontsize_)

    # plot_uv_hist_2(fig.add_subplot(ny, nx, 3), results[0], settings, r_min, r_max,
    #              fontsize_)
    # plot_uv_hist_2(fig.add_subplot(ny, nx, 4), results[1], settings, r_min, r_max,
    #              fontsize_)

    # TODO(BM) UV density histogram (see alma memo 390)

    # nbins = 500
    # plot_psf_1d_2(fig.add_subplot(ny, nx, 3), results[0], settings, nbins,
    #               fontsize_, vmin_0h, linthresh_0h)
    # plot_psf_1d_2(fig.add_subplot(ny, nx, 4), results[1], settings, nbins,
    #               fontsize_, vmin_0h, linthresh_0h)

    # TODO(BM) PSF peak average and RMS plot (see alma memo 390)


    # # 1D PSF
    # plot_psf_1d(fig.add_subplot(ny, nx, 11), results[0], settings, fontsize_,
    #             vmin, linthresh)
    # plot_psf_1d(fig.add_subplot(ny, nx, 12), results[1], settings, fontsize_,
    #             vmin, linthresh)

    # 1D PSF (bins)
    # http://stackoverflow.com/questions/15556930/turn-scatter-data-into-binned-data-with-errors-bars-equal-to-standard-deviation
    plot_psf_1d_bins(fig.add_subplot(ny, nx, 5), results[0], settings,
                     fontsize_, vmin_0h, linthresh_0h)
    plot_psf_1d_bins(fig.add_subplot(ny, nx, 6), results[1], settings,
                     fontsize_, vmin_4h, linthresh_4h)

    # 2D PSF
    # vmin_psf_im = -0.03
    # linthresh_psf_im = 0.05
    # plot_psf(fig.add_subplot(ny, nx, 7), results[0], settings, fontsize_,
    #          vmin_psf_im, linthresh_psf_im)
    # plot_psf(fig.add_subplot(ny, nx, 8), results[1], settings, fontsize_,
    #          vmin_psf_im, linthresh_psf_im)

    # PSF histogram
    plot_psf_histogram(fig.add_subplot(ny, nx, 7), results[0], settings,
                       fontsize_, y_max=0.2)
    plot_psf_histogram(fig.add_subplot(ny, nx, 8), results[1], settings,
                       fontsize_, y_max=0.2)

    plt.savefig(join(settings['results_dir'], '%s.png' % (save_root_name)))
    plt.close(fig)
    print('- Plotting took %.2f s' % (time.time() - t0))


def inner_arms(b=0.513, num_arms=3, n=24, r_min=500.0, r_max=1700.0,
               layout=None):
    """
    Generate inner spiral arms in ENU coordinates

    Args:
        b:
        num_arms:
        n:
        r_min:
        r_max:

    Returns:
    """
    if not layout:
        layout = list()
    for i in range(num_arms):
        x, y = log_spiral_2(r0=r_min, r1=r_max, b=b, n=n)
        x, y = rotate_coords(x, y, i * (360.0 / num_arms))
        layout.append({'x': x, 'y': y, 'color': 'k'})
    return layout


def inner_arms_clusters(b=0.513, num_arms=3, clusters_per_arm=4,
                        stations_per_cluster=6, cluster_diameter_m=100.0,
                        station_radius_m=22.5, r_min=500.0, r_max=1700.0,
                        layout=None):
    if not layout:
        layout = list()
    for i in range(num_arms):
        x, y, cx, cy = log_spiral_clusters(r0=r_min, r1=r_max, b=b,
                                           n=clusters_per_arm,
                                           n_cluster=stations_per_cluster,
                                           r_cluster=cluster_diameter_m / 2.0,
                                           min_sep=station_radius_m * 2.0)
        x, y = rotate_coords(x, y, i * (360.0 / num_arms))
        cx, cy = rotate_coords(cx, cy, i * (360.0 / num_arms))
        layout.append({'x': x, 'y': y, 'color': 'k', 'type': 'cluster',
                       'cluster_diameter': cluster_diameter_m,
                       'cluster_x': cx, 'cluster_y': cy})
    return layout


def inner_arms_rand_uniform(num_stations, station_radius_m,
                            r_min=500.0, r_max=1700.0, layout=None):
    if not layout:
        layout = list()
    x, y, _ = gridgen(num_points=num_stations, r1=r_max,
                      min_sep=station_radius_m*2.0, r0=r_min, max_tries=1000)
    if not x.shape[0] == num_stations:
        raise RuntimeError('Failed to generate enough stations.')
    layout.append({'x': x, 'y': y, 'color': 'r'})
    return layout


def main():
    settings = {
        'station_r': 45.0 / 2.0,
        'lon': math.radians(116.63128900),
        'lat': math.radians(-26.69702400),
        'ra': math.radians(68.698903779331502),
        'dec': math.radians(-26.568851215532160),
        'mjd_mid': 57443.4375000000,
        'freq_hz': 200e6,
        'uv_grid_size': 64,
        'psf_im_size': 2048 * 2,
        'psf_fov_deg': 60.0,
        'results_dir': 'results'
    }
    r_min = 500.0
    r_max = 1700.0

    # FIXME(BM) multiple PSF FoV's
    # FIXME(BM) think about PSF FoV vs frequency .... (cant just use lambda=1m!)

    # Single arms
    b, num_arms, n = 0.5, 3, 24
    layout = inner_arms(b, num_arms, n, r_min=500.0, r_max=1700.0)
    results = analyse_layout(layout, settings)
    plotting(layout, results, settings,
             'spiral_b%.1f_%02ix%02i' % (b, num_arms, n))

    return

    # Clusters
    b = 0.5
    num_arms, clusters_per_arm, stations_per_cluster = 3, 4, 6
    cluster_diameter_m, station_radius_m = 200, settings['station_r']
    layout = inner_arms_clusters(b, num_arms, clusters_per_arm,
                                 stations_per_cluster,
                                 cluster_diameter_m, station_radius_m,
                                 r_min=r_min, r_max=r_max)
    results = analyse_layout(layout, settings)
    plotting(layout, results, settings,
             'spiral_clusters_b%.1f_%02ix%02ix%02i' %
             (b, num_arms, clusters_per_arm, stations_per_cluster))

    # Random uniform
    num_stations = 72
    station_radius_m = settings['station_r']
    for iter in range(1):
        seed = numpy.random.randint(1, 20000)
        numpy.random.seed(seed)
        layout = inner_arms_rand_uniform(num_stations, station_radius_m, r_min,
                                         r_max)
        results = analyse_layout(layout, settings)
        plotting(layout, results, settings,
                 'random_uniform_%02i_%i' % (num_stations, iter))


if __name__ == '__main__':
    main()
