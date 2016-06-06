# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy
from matplotlib import ticker
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


def plot_layout(ax, layout, settings, r_min, r_max, fontsize='small'):
    x = layout['x'] / 1.0e3
    y = layout['y'] / 1.0e3
    color = 'k' if not 'color' in layout else layout['color']
    for i in range(x.shape[0]):
        c = plt.Circle((x[i], y[i]), settings['station_r'] / 1.0e3,
                       fill=True, color=color, alpha=0.5)
        ax.add_artist(c)
    if 'type' in layout and layout['type'] == 'clusters':
        cx = layout['cluster_x'] / 1.0e3
        cy = layout['cluster_y'] / 1.0e3
        cluster_r = layout['cluster_diameter'] / (2.0 * 1.0e3)
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
    ax.set_xlim(-r_max / 1.0e3, r_max / 1.0e3)
    ax.set_ylim(-r_max / 1.0e3, r_max / 1.0e3)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax.locator_params(axis='both', nbins=5)
    ax.set_xlabel('East (kilometres)', fontsize=fontsize)
    ax.set_ylabel('North (kilometres)', fontsize=fontsize)
    ax.set_title('Station positions (n=%i)' % x.shape[0], fontsize=fontsize)
    ax.grid()


def plot_uv_scatter_2(ax, uu, vv, r_max, settings, fontsize):
    wavelength_m = 299792458.0 / settings['freq_hz']
    if settings['num_times'] == 1:
        alpha = 0.2
    else:
        alpha = max(0.002, 0.2 / (settings['num_times']))
    uu *= wavelength_m
    vv *= wavelength_m
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
                 (settings['obs_length_s'] / 3600.0,
                  settings['num_times']), fontsize=fontsize)


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


def plot_uv_hist(ax, result, settings, r_min, r_max, fontsize):
    wavelength_m = 299792458.0 / settings['freq_hz']
    uu = result['uu'] * wavelength_m
    vv = result['vv'] * wavelength_m
    uv_dist = (uu**2 + vv**2)**0.5
    uv_dist.sort()
    num_bins = 20
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
    num_bins = 20
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


def plot_psf_2(ax, psf, settings):
    centre = settings['psf_im_size'] / 2.0
    extent = numpy.array([centre + 0.5, -centre + 0.5,
                          -centre - 0.5, centre - 0.5])
    lm_max = math.sin(0.5 * math.radians(settings['psf_fov_deg']))
    lm_inc = 2.0 * lm_max / settings['psf_im_size']
    extent *= lm_inc
    im_ = ax.imshow(psf, interpolation='nearest', cmap='inferno',
                    origin='lower',
                    norm=SymLogNorm(linthresh=0.05, linscale=1.0,
                                    vmin=-0.1, vmax=0.5, clip=False),
                    extent=extent)

    # Color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    cbar = ax.figure.colorbar(im_, cax=cax, format='%.1f')
    cbar.ax.tick_params(labelsize='smaller')
    ax.grid(True)


def plot_psf_1d_new(ax, psf, r_lm, settings, r_max, color='b', label=None):
    r_lm = r_lm.flatten()
    order = numpy.argsort(r_lm)
    r_lm = r_lm[order]
    psf_1d = psf.flatten()[order]

    wavelength_m = 299792458.0 / settings['freq_hz']
    psf_hwhm = (wavelength_m / (r_max * 2.0)) / 2.0  # FIXME(BM) better expression?
    x = r_lm / psf_hwhm

    num_bins = 500
    bin_mean, edges, number = \
        stats.binned_statistic(x, psf_1d, statistic='mean', bins=num_bins)
    bin_x = (edges[1:] + edges[:-1]) / 2

    def bin_max(values):
        return numpy.abs(values).max()

    bin_max, edges, number = \
        stats.binned_statistic(x, psf_1d, statistic=bin_max, bins=num_bins)

    # ax.plot(x, psf_1d, 'k.', markersize=3.0, alpha=0.5)
    ax.plot(bin_x, bin_mean, '--', color=color, linewidth=1.0, alpha=0.6)
    ax.plot(bin_x, bin_max, '-', color=color, linewidth=1.0, alpha=0.6,
            label=label)
    ax.grid(True)


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
    num_antennas = result['num_antennas']
    x = result['psf_bins']['r']
    y1 = result['psf_bins']['min']
    y2 = result['psf_bins']['max']
    ax.plot(x, y1, '-', color='k', linewidth=0.5)
    ax.plot(x, y2, '-', color='k', linewidth=0.5)
    ax.fill_between(x, y1, y2, edgecolor='k', facecolor='k', alpha=0.2)
    y = result['psf_bins']['mean']
    std = result['psf_bins']['std']
    ax.plot(x, y, '-', color='r')
    ax.plot(x, std, color='b')
    # ax.fill_between(x, y - std, y + std, edgecolor='b', facecolor='b',
    #                 alpha=0.1)
    # ax.plot(x, y + std, '-', color='b', linewidth=0.5)
    # ax.plot(x, y - std, '-', color='b', linewidth=0.5)
    yy = (1.0 / num_antennas) * (1.0 / math.sqrt(result['obs']['num_times']))
    ax.plot(ax.get_xlim(), [yy, yy], 'g--')
    yy = -(1.0 / float(num_antennas - 1)) * \
         (1.0 / math.sqrt(result['obs']['num_times']))
    ax.plot(ax.get_xlim(), [yy, yy], 'g--')
    yy = (2.0 / float(num_antennas)) * math.log(10e3 / 40.0)
    ax.plot(ax.get_xlim(), [yy, yy], 'm--')

    if math.sqrt(result['obs']['num_times']) > 1:
        yy = -(1.0 / float(num_antennas - 1))
        ax.plot(ax.get_xlim(), [yy, yy], '-', color='c')

    # ax.set_ylim(vmin, linthresh)
    # ax.set_xlim(0, (settings['psf_im_size'] / 2) * result['psf_cell_size'])
    # ax.set_yscale('symlog', linthresh=0.01)
    # ax.set_xscale('symlog', linthresh=0.2)
    ax.set_xscale('log')
    # ax.locator_params(axis='x', nbins=6)  # Not possible with log axes

    ax.grid(True)
    ax.set_xlabel('PSF radius', fontsize=fontsize)
    ax.set_ylabel('PSF amplitude', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plot_psf_histogram(ax, result, settings, fontsize, y_max):
    psf = result['psf'].flatten()
    n, _ = numpy.histogram(psf, bins=200, density=True)
    x = (_[1:] + _[:-1])/2
    ax.plot(x, n, '-', color='b')
    # ax.set_xlim(0.00, y_max)
    if result['obs']['num_times'] == 1:
        N = result['num_antennas']
        N0 = n[numpy.argmax(x >= 0.0)]
        g = N0 * numpy.exp(-N * x)
        ax.plot(x, g, '--', color='r')
        ax.set_ylim(1.0e-6, n.max())
    ax.set_xlim(0.005, 1.0)
    ax.set_yscale('log', nonposy='mask')
    # ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('sidelobe magitude', fontsize=fontsize)
    ax.set_ylabel('number of samples', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)


def plot_psf_histogram_2(ax, psf, color='b', fontsize='small', label=None,
                         num_antennas=None, mag=None):
    data = numpy.abs(psf.flatten())
    n, _ = numpy.histogram(data, bins=500, density=True)
    x = (_[1:] + _[:-1])/2
    ax.plot(x, n, '-', color=color, label=label)
    ax.set_xlim(0.0, 0.3)
    y_max = 10**math.ceil(math.log10(max(num_antennas, n.max())))
    ax.set_ylim(1.0e-5, y_max)
    ax.set_yscale('log', nonposy='mask')
    ax.grid(True)
    if num_antennas:
        ax.plot(ax.get_xlim(), [num_antennas, num_antennas], '-', color=color)
        g = num_antennas * numpy.exp(-num_antennas * x)
        ax.plot(x, g, ':', color=color, alpha=0.5)
    if mag:
        s_max = (2.0 / float(num_antennas)) * math.log(mag)
        ax.plot([s_max, s_max], ax.get_ylim(), '-', color=color)
    ax.set_xlabel('sidelobe magitude', fontsize=fontsize)
    ax.set_ylabel('number of samples', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)

