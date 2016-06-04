# -*- coding: utf-8 -*-

import math
import os
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import oskar
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyuvwsim import (convert_enu_to_ecef,
                      evaluate_baseline_uvw,
                      evaluate_station_uvw)

from utilities.generators import inner_arms


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
        'psf_im_size': 4096,
        'psf_fov_deg': 20,
        'obs_length_s': 0.0,
        'num_times': 1,
        'interval_s': 0.0,
        'results_dir': 'results_opt'
    }

    r_min = 70.0
    r_max = 5000.0

    if not os.path.isdir(settings['results_dir']):
        os.makedirs(settings['results_dir'])

    # FIXME(BM) multiple PSF FoV's
    # FIXME(BM) think about PSF FoV vs frequency .... (cant just use lambda=1m!)
    # FIXME(BM) think about how to modularise analysis and plotting
    #           so that analysis is not run when plots are not used.
    #           ---> combine plotting and analysis?

    # Generate the initial layout
    layout = inner_arms(b=0.5, num_arms=3, n=20, r_min=r_min, r_max=r_max)
    x, y = layout['x'], layout['y']

    # Plot the layout
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(x.shape[0]):
        c = plt.Circle((x[i], y[i]), settings['station_r'], fill=True,
                       color='r', alpha=0.5)
        ax.add_artist(c)
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    plt.savefig(join(settings['results_dir'], 'layout.png'))
    # plt.show()
    plt.close(fig)

    # Generate baseline and station uvw coordinates.
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

    # Plot baseline distribution
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu * wavelength_m, vv * wavelength_m, 'k.', ms=3.0, alpha=0.5)
    ax.plot(-uu * wavelength_m, -vv * wavelength_m, 'k.', ms=3.0, alpha=0.5)
    ax.set_xlim(-r_max * 2.0, r_max * 2.0)
    ax.set_ylim(-r_max * 2.0, r_max * 2.0)
    plt.savefig(join(settings['results_dir'], 'uv_dist.png'))
    plt.close(fig)

    # Generate PSF
    t0 = time.time()
    im = oskar.imager.Imager('Single')
    im.set_fft_on_gpu(False)
    psf = im.make_image(uu, vv, ww, np.ones_like(uu, dtype='c8'),
                        np.ones_like(uu, 'c8'), settings['psf_fov_deg'],
                        settings['psf_im_size'])
    print('- PSF generation took %.1f s' % (time.time() - t0))


    # Plot PSF
    # psf = np.zeros((settings['psf_im_size'], settings['psf_im_size']))

    # psf[256, 150] = 0.5
    # psf[256, 256] = 10.0
    lm_max = math.sin(0.5 * math.radians(settings['psf_fov_deg']))
    lm_inc = 2.0 * lm_max / settings['psf_im_size']
    l = np.arange(-settings['psf_im_size'] / 2, settings['psf_im_size'] / 2)
    l = l.astype(dtype='f8') * lm_inc
    l, m = np.meshgrid(-l, l)
    r_lm = (l**2 + m**2)**0.5

    r_lm = (l ** 2 + m ** 2) ** 0.5
    # Mask areas of the PSF we don't want to find peaks in
    mask_r = 0.005
    # psf[70, 100] = 0.9
    masked_psf = np.copy(psf)
    masked_psf[np.where(r_lm <= mask_r)] = 0.0
    iy, ix = np.unravel_index(np.argmax(masked_psf), masked_psf.shape)
    l_peak = l[iy, ix]
    m_peak = m[iy, ix]
    r_peak = (l_peak**2 + m_peak**2)**0.5

    t0 = time.time()
    centre = settings['psf_im_size'] / 2.0
    extent = np.array([centre + 0.5, -centre + 0.5,
                       -centre - 0.5, centre - 0.5])
    lm_max = math.sin(0.5 * math.radians(settings['psf_fov_deg']))
    lm_inc = 2.0 * lm_max / settings['psf_im_size']
    extent *= lm_inc
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(l_peak, m_peak, 'gx', markersize=10.0, mew=2.0)
    c = plt.Circle((0.0, 0.0), r_peak, fill=False, color='g', alpha=1.0)
    ax.add_artist(c)
    c = plt.Circle((0.0, 0.0), mask_r, fill=True, color='w', alpha=0.5)
    ax.add_artist(c)
    im_ = ax.imshow(masked_psf, interpolation='nearest', cmap='inferno',
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
    # plt.savefig(join(settings['results_dir'], 'psf_2d.png'))
    plt.show()
    plt.close(fig)
    print('- Plotting 2D PSF took %.1f s' % (time.time() - t0))

    r_lm = r_lm.flatten()
    order = np.argsort(r_lm)
    r_lm = r_lm[order]
    psf_1d = psf.flatten()[order]

    t0 = time.time()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(r_lm, psf_1d, 'k.', markersize=3.0, alpha=0.8)
    ax.grid(True)
    # ax.set_yscale('symlog', linthresh=0.05)
    ax.set_ylim(-0.2, 0.3)
    ax.plot([r_peak, r_peak], ax.get_ylim(), 'g-', alpha=0.5)
    ax.plot([mask_r, mask_r], ax.get_ylim(), 'r-', alpha=0.5)
    plt.savefig(join(settings['results_dir'], 'psf_1d.png'))
    # plt.show()
    plt.close(fig)
    print('- Plotting 1D PSF took %.1f s' % (time.time() - t0))

    # Pick a random antenna
    i_ant = np.random.randint(0, u.shape[0])
    drn_x = 1.0
    drn_y = 1.0
    rn_x, rn_y = u[i_ant], v[i_ant]
    r_uu = u - rn_x
    r_vv = v - rn_y
    r_uu, r_vv = np.delete(r_uu, i_ant), np.delete(r_vv, i_ant)
    dp_u = np.sum(np.sin(2.0 * math.pi * r_uu / wavelength_m * l_peak))
    dp_u *= (4.0 * math.pi * l_peak * drn_x) / float(u.shape[0])**2
    dp_v = np.sum(np.sin(2.0 * math.pi * r_vv / wavelength_m * m_peak))
    dp_v *= (4.0 * math.pi * m_peak * drn_y) / float(u.shape[0]) ** 2
    print(i_ant, dp_u, dp_v)


if __name__ == '__main__':
    main()
