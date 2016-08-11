# -*- coding: utf-8 -*-

import math
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import oskar
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyuvwsim import (convert_enu_to_ecef,
                      evaluate_baseline_uvw,
                      evaluate_station_uvw)
from scipy import stats

from utilities.generators import inner_arms


# TODO(BM) move some of these functions to utilties.


def plot_layout(layout, settings, r_max, plot_name='layout.png'):
    """Plot the station layout"""
    x, y = layout['x'], layout['y']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(x.shape[0]):
        c = plt.Circle((x[i], y[i]), settings['station_r'], fill=True,
                       color='r', alpha=0.5)
        ax.add_artist(c)
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    plt.savefig(join(settings['results_dir'], plot_name))
    plt.close(fig)


def generate_baseline_coordinates(layout, settings):
    """Generate baseline coordinates"""
    x, y = layout['x'], layout['y']
    wavelength_m = 299792458.0 / settings['freq_hz']
    mjd_start = settings['mjd_mid'] - (settings['obs_length_s'] / 2.0) / 86400.0
    x, y, z = np.array(x), np.array(y), np.zeros_like(x)
    x_ecef, y_ecef, z_ecef = convert_enu_to_ecef(x, y, z, settings['lon'],
                                                 settings['lat'])
    x_c, y_c, z_c = convert_enu_to_ecef([0.0], [0.0], [0.0], settings['lon'],
                                        settings['lat'])
    x_ecef -= x_c
    y_ecef -= y_c
    z_ecef -= z_c
    num_stations = x_ecef.shape[0]
    num_baselines = num_stations * (num_stations - 1) / 2
    num_coords = settings['num_times'] * num_baselines
    uu = np.zeros(num_coords, dtype='f4')
    vv, ww = np.zeros_like(uu), np.zeros_like(uu)
    u = np.zeros(num_stations, dtype='f4')
    v, w = np.zeros_like(u), np.zeros_like(u)
    for i in range(settings['num_times']):
        mjd = mjd_start + (i * settings['interval_s'] +
                           settings['interval_s'] / 2.0) / 86400.0
        uu_, vv_, ww_ = evaluate_baseline_uvw(x_ecef, y_ecef, z_ecef,
                                              settings['ra'], settings['dec'],
                                              mjd)
        u, v, w = evaluate_station_uvw(x_ecef, y_ecef, z_ecef, settings['ra'],
                                       settings['dec'], mjd)
        uu[i * num_baselines: (i + 1) * num_baselines] = uu_ / wavelength_m
        vv[i * num_baselines: (i + 1) * num_baselines] = vv_ / wavelength_m
        ww[i * num_baselines: (i + 1) * num_baselines] = ww_ / wavelength_m
    return uu, vv, ww, u, w, v


def plot_baseline_distribution(uu, vv, settings, r_max,
                               plot_name='uv_dist.png'):
    wavelength_m = 299792458.0 / settings['freq_hz']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu * wavelength_m, vv * wavelength_m, 'k.', ms=3.0, alpha=0.5)
    ax.plot(-uu * wavelength_m, -vv * wavelength_m, 'k.', ms=3.0, alpha=0.5)
    ax.set_xlim(-r_max * 2.0, r_max * 2.0)
    ax.set_ylim(-r_max * 2.0, r_max * 2.0)
    plt.savefig(join(settings['results_dir'], plot_name))
    plt.close(fig)


def generate_psf(uu, vv, ww, settings):
    im = oskar.imager.Imager('Single')
    im.set_fft_on_gpu(False)
    psf = im.make_image(uu, vv, ww, np.ones_like(uu, dtype='c8'),
                        np.ones_like(uu, 'c8'), settings['psf_fov_deg'],
                        settings['psf_im_size'])
    return psf


def get_psf_coords(settings):
    lm_max = math.sin(0.5 * math.radians(settings['psf_fov_deg']))
    lm_inc = 2.0 * lm_max / settings['psf_im_size']
    l = np.arange(-settings['psf_im_size'] / 2, settings['psf_im_size'] / 2)
    l = l.astype(dtype='f8') * lm_inc
    l, m = np.meshgrid(-l, l)
    r = (l**2 + m**2)**0.5
    return l, m, r


def find_psf_peak(psf, l, m, r_lm, mask_r, settings):
    masked_psf = np.copy(psf)
    masked_psf[np.where(r_lm <= mask_r)] = 0.0
    iy, ix = np.unravel_index(np.argmax(masked_psf), masked_psf.shape)
    l_peak = l[iy, ix]
    m_peak = m[iy, ix]
    return l_peak, m_peak


def plot_psf(psf, settings, l_peak, m_peak, mask_r, plot_name='psf_2d.png'):

    centre = settings['psf_im_size'] / 2.0
    extent = np.array([centre + 0.5, -centre + 0.5,
                       -centre - 0.5, centre - 0.5])
    lm_max = math.sin(0.5 * math.radians(settings['psf_fov_deg']))
    lm_inc = 2.0 * lm_max / settings['psf_im_size']
    extent *= lm_inc

    r_peak = (l_peak**2 + m_peak**2)**0.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(l_peak, m_peak, 'gx', markersize=10.0, mew=2.0)
    c = plt.Circle((0.0, 0.0), r_peak, fill=False, color='g', alpha=1.0)
    ax.add_artist(c)
    c = plt.Circle((0.0, 0.0), mask_r, fill=True, color='w', alpha=0.5)
    ax.add_artist(c)
    im_ = ax.imshow(psf, interpolation='nearest', cmap='inferno',
                    origin='lower',
                    norm=SymLogNorm(linthresh=0.05, linscale=1.0,
                                    vmin=-0.2, vmax=1.0, clip=False),
                    extent=extent)

    # color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    cbar = ax.figure.colorbar(im_, cax=cax, format='%.1f')
    cbar.ax.tick_params(labelsize='smaller')

    ax.grid(True)
    plt.savefig(join(settings['results_dir'], plot_name))
    # plt.show()
    plt.close(fig)


def plot_psf_1d(psf, r_lm, r_peak, mask_r, settings, plot_name='psf_1d.png'):
    r_lm = r_lm.flatten()
    order = np.argsort(r_lm)
    r_lm = r_lm[order]
    psf_1d = psf.flatten()[order]

    num_bins = 500
    bin_mean, edges, number = \
        stats.binned_statistic(r_lm, psf_1d, statistic='mean', bins=num_bins)
    bin_x = (edges[1:] + edges[:-1]) / 2

    def bin_max(values):
        return values.max()

    bin_max, edges, number = \
        stats.binned_statistic(r_lm, psf_1d, statistic=bin_max, bins=num_bins)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # ax.plot(r_lm, psf_1d, 'k.', markersize=3.0, alpha=0.5)
    ax.plot(bin_x, bin_mean, 'b--', linewidth=1.0)
    ax.plot(bin_x, bin_max, 'b-', linewidth=1.0)
    ax.grid(True)
    # ax.set_yscale('symlog', linthresh=0.05)
    y_max = psf_1d[r_lm > mask_r].max()
    y_min = psf_1d[r_lm > mask_r].min()
    # ax.set_ylim(y_min, y_max * 1.1)
    ax.set_ylim(-0.1, y_max * 1.1)
    ax.plot([r_peak, r_peak], ax.get_ylim(), 'g-', alpha=0.5, linewidth=3.0)
    ax.plot([mask_r, mask_r], ax.get_ylim(), 'r-', alpha=0.5, linewidth=3.0)
    ax.plot(r_peak, y_max, 'x', color='g', markersize=10.0, alpha=0.8)
    plt.savefig(join(settings['results_dir'], plot_name))
    # plt.show()
    plt.close(fig)


def psf_peak_derivative(u, v, l_peak, m_peak, wavelength_m, drn_x, drn_y):
    i_ant = np.random.randint(0, u.shape[0])
    rn_x, rn_y = u[i_ant], v[i_ant]
    r_uu = u - rn_x
    r_vv = v - rn_y
    r_uu, r_vv = np.delete(r_uu, i_ant), np.delete(r_vv, i_ant)
    dp_u = np.sum(np.sin(2.0 * math.pi * (r_uu / wavelength_m) * l_peak))
    n_sq = float(u.shape[0]) ** 2
    dp_u *= (4.0 * math.pi * l_peak * drn_x) / n_sq
    dp_v = np.sum(np.sin(2.0 * math.pi * (r_vv / wavelength_m) * m_peak))
    dp_v *= (4.0 * math.pi * m_peak * drn_y) / n_sq
    return i_ant, dp_u, dp_v


def main():
    settings = {
        'station_r': 45.0 / 2.0,
        'lon': math.radians(116.63128900),
        'lat': math.radians(-26.69702400),
        'ra': math.radians(68.698903779331502),
        'dec': math.radians(-26.568851215532160),
        'mjd_mid': 57443.4375000000,
        'freq_hz': 100e6,
        'uv_grid_size': 64,
        'psf_im_size': 2048,
        'psf_fov_deg': 15.0,
        'obs_length_s': 0.0,
        'num_times': 1,
        'interval_s': 0.0,
        'results_dir': 'results_opt'
    }
    wavelength_m = 299792458.0 / settings['freq_hz']

    if not os.path.isdir(settings['results_dir']):
        os.makedirs(settings['results_dir'])

    # Generate the initial layout
    r_min = 70.0
    r_max = 5000.0
    layout = inner_arms(b=7.0, num_arms=3, n=9, r_min=r_min, r_max=r_max)
    # plot_layout(layout, settings, r_max, 'layout_0.png')

    # Generate baseline and station uvw coordinates.
    uu, vv, ww, u, v, w = generate_baseline_coordinates(layout, settings)

    # Plot baseline distribution
    # plot_baseline_distribution(uu, vv, settings, r_max, 'uv_dist_0.png')

    # Generate PSF (and PSF image coordinates)
    psf = generate_psf(uu, vv, ww, settings)
    l, m, r_lm = get_psf_coords(settings)
    beam_sigma = math.sin(wavelength_m / (settings['station_r'] * 2.0))
    psf *= np.exp(-(l**2 + m**2) / (2.0 * beam_sigma**2))

    # Find the peak in the PSF
    mask_r = math.sin(wavelength_m / (r_max * 2.0)) * 10.0
    l_peak, m_peak = find_psf_peak(psf, l, m, r_lm, mask_r, settings)

    # Plot the PSF
    # plot_psf(psf, settings, l_peak, m_peak, mask_r, 'psf_2d_0.png')
    # plot_psf_1d(psf, r_lm, (l_peak**2 + m_peak**2)**0.5, mask_r, settings,
    #             'psf_1d_0.png')

    # Pick a random antenna and find derivative of psf peak wrt shift of antenna
    # FIXME(BM) units in derivative calculation (u, v, l ...)
    i_ant, dp_u, dp_v = psf_peak_derivative(u, v, l_peak, m_peak, wavelength_m,
                                            drn_x=settings['station_r'] / 2.0,
                                            drn_y=settings['station_r'] / 2.0)

    # Calculate antenna move
    alpha = math.atan2(m_peak, l_peak)
    g = 0.01
    dx = -g * dp_u * math.cos(alpha)
    dy = -g * dp_v * math.sin(alpha)
    print('angle', math.degrees(alpha))
    print('derivative', dp_u, dp_v)
    print('move', i_ant, dx, dy)
    print('wavelength', wavelength_m)
    print('angle?', math.degrees(math.atan2(dp_v, dp_u)))


    # 1- Generate PSF
    # 2- Find peak
    # 3- Pick antenna and find derivative
    # 4- Move antenna
    # 5- Generate PSF
    # 6- Check if peak is reduced
    #   - If yes: goto 2
    #   - If no: ignore move and go 3



if __name__ == '__main__':
    main()
