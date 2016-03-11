# -*- coding: utf-8 -*-
from __future__ import print_function
from os.path import join

import math
import numpy
import matplotlib.pyplot as pyplot
from math import radians, ceil
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from pyuvwsim import (load_station_coords, convert_enu_to_ecef,
                          evaluate_baseline_uvw)
    uvwsim_found = True
except ImportError:
    print('pyuvwsim not found, skipping uvw co-ordiante generation.')
    print('see: https://github.com/SKA-ScienceDataProcessor/uvwsim, pyuvwsim.rst')
    uvwsim_found = False
from layout_utils import (generate_baseline_uvw)
import time
import os
try:
    from oskar.imager import Imager
    oskar_imager_found = True
except ImportError:
    print('OSKAR python imager not found, PSF images wont be made.')
    oskar_imager_found = False
import pyfits


def save_fits_image(file_name, image, cell_size_deg, ra0, dec0, freq_hz):

    # FIXME-BM get ra0, dec0 from the oskar vis file?!
    data = numpy.reshape(image, (1, 1, image.shape[0], image.shape[1]))

    header = pyfits.header.Header()
    header.append(('BUNIT', 'Jy/beam'))
    header.append(('CTYPE1', 'RA--SIN'))
    header.append(('CRVAL1', ra0))
    header.append(('CDELT1', -cell_size_deg))
    header.append(('CUNIT1', 'deg'))
    # Note: Assumes even image dims and that image pixels start at 1 (not 0)
    header.append(('CRPIX1', image.shape[1] / 2 + 1))
    header.append(('CTYPE2', 'DEC-SIN'))
    header.append(('CRVAL2', dec0))
    header.append(('CDELT2', cell_size_deg))
    header.append(('CRPIX2', image.shape[0] / 2 + 1))
    header.append(('CUNIT2', 'deg'))
    header.append(('CTYPE3', 'FREQ'))
    header.append(('CRVAL3', freq_hz))
    header.append(('CDELT3', 1.0))
    header.append(('CRPIX3', 1.0))
    header.append(('CUNIT3', 'Hz'))
    header.append(('CTYPE4', 'STOKES'))
    header.append(('CRVAL4', 1.0))
    header.append(('CDELT4', 1.0))
    header.append(('CRPIX4', 1.0))
    header.append(('CUNIT4', ''))
    if os.path.exists(file_name):
        print('- WARNING: Overwriting FITS file: %s' % file_name)
        os.remove(file_name)
    print('- Saving FITS image:', file_name)
    pyfits.writeto(file_name, data, header)


def plot_layouts(v4d, v5d, station_radius_m):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(v4d.shape[0]):
        circle = pyplot.Circle((v4d[i, 0], v4d[i, 1]), station_radius_m,
                               color='b', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_title('v4d')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(True)
    pyplot.show()

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(v4d.shape[0]):
        circle = pyplot.Circle((v5d[i, 0], v5d[i, 1]), station_radius_m,
                               color='b', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_title('v5d')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(True)
    pyplot.show()


def uv_plot(uu_v4d, vv_v4d, uu_v5d, vv_v5d, out_dir):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4d, vv_v4d, 'k.', alpha=0.1, ms=2.0)
    ax.plot(-uu_v4d, -vv_v4d, 'k.', alpha=0.1, ms=2.0)
    ax.set_title('v4d')
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    pyplot.savefig(join(out_dir, 'scatter_v4d_300m.png'))
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    pyplot.savefig(join(out_dir, 'scatter_v4d_1000m.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'scatter_v4d_3000m.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'scatter_v4d_5000m.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v5d, vv_v5d, 'k.', alpha=0.1, ms=2.0)
    ax.plot(-uu_v5d, -vv_v5d, 'k.', alpha=0.1, ms=2.0)
    ax.set_title('v5d')
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    pyplot.savefig(join(out_dir, 'scatter_v5d_300m.png'))
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    pyplot.savefig(join(out_dir, 'scatter_v5d_1000m.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'scatter_v5d_3000m.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'scatter_v5d_5000m.png'))
    pyplot.close(fig)


def hist_plot_1(v4d_uv_dist, v5d_uv_dist, bin_width, uv_dist_max,
                out_dir):
    v4d_uv_dist = v4d_uv_dist[v4d_uv_dist < uv_dist_max * 2.0]
    v5d_uv_dist = v5d_uv_dist[v5d_uv_dist < uv_dist_max * 2.0]
    num_bins = int(ceil(uv_dist_max / bin_width))
    bins = numpy.arange(num_bins) * bin_width
    v4d_hist, v4d_bin_edges = numpy.histogram(v4d_uv_dist, bins=bins)
    v5d_hist, v5d_bin_edges = numpy.histogram(v5d_uv_dist, bins=bins)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1]
    y = v4d_hist
    ax.bar(x, y, width=numpy.diff(v4d_bin_edges), color='none', lw=1.5,
           label='v4d')
    x = v5d_bin_edges[:-1]
    y = v5d_hist
    ax.bar(x, y, width=numpy.diff(v5d_bin_edges), color='r',
           alpha=0.5, edgecolor='k', label='v5d')
    ax.set_xlim(0, uv_dist_max)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Number')
    ax.legend(fontsize='small')
    pyplot.savefig(join(out_dir, 'hist_%04.1fm_%06.1fm.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + numpy.diff(v4d_bin_edges) / 2.0
    y = v4d_hist
    ax.plot(x, y, 'k-', lw=1.5, label='v4d')
    x = v5d_bin_edges[:-1] + numpy.diff(v5d_bin_edges) / 2.0
    y = v5d_hist
    ax.plot(x, y, 'r-', lw=1.5, label='v5d')
    ax.set_xlim(0, uv_dist_max)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Number')
    ax.legend()
    pyplot.savefig(join(out_dir, 'hist_%04.1fm_%06.1fm_v2.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)


def hist_plot_2(v4d_uv_dist, v5d_uv_dist, wave_length, out_dir):
    bin_width = wave_length
    num_vis = v4d_uv_dist.shape[0]
    assert(v4d_uv_dist.shape[0] == v5d_uv_dist.shape[0])
    uv_dist_max = 70.0e3
    uv_dist_min = 20.0
    num_bins = int(ceil(uv_dist_max / bin_width))
    bins = numpy.arange(num_bins) * bin_width
    v4d_hist, v4d_bin_edges = numpy.histogram(v4d_uv_dist, bins=bins)
    v5d_hist, v5d_bin_edges = numpy.histogram(v5d_uv_dist, bins=bins)
    v4d_hist = numpy.array(v4d_hist)
    v5d_hist = numpy.array(v5d_hist)
    v4d_bin_widths = numpy.diff(v4d_bin_edges)
    v5d_bin_widths = numpy.diff(v5d_bin_edges)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.divide(v4d_hist, v4d_bin_widths)
    ax.loglog(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v5d_bin_edges[:-1] + v5d_bin_widths / 2.0
    y = numpy.divide(v5d_hist, v5d_bin_widths)
    ax.loglog(x, y, 'r-', label='v5d', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Visibility count (m$^{-1}$)')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'hist_loglog_%04.1fm_%06.1fm.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.cumsum(v4d_hist) / float(num_vis)
    ax.semilogx(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v5d_bin_edges[:-1] + v5d_bin_widths / 2.0
    y = numpy.cumsum(v5d_hist) / float(num_vis)
    ax.semilogx(x, y, 'r-', label='v5d', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Cumulative visibility density')
    ax.legend(loc=2)
    ax.grid()
    v4d_median = numpy.median(v4d_uv_dist)
    v5d_median = numpy.median(v5d_uv_dist)
    ax.plot([v4d_median, v4d_median], ax.get_ylim(), 'k:')
    ax.plot([v5d_median, v5d_median], ax.get_ylim(), 'r--')
    ax.text(0.02, 0.73, 'v4d median = %.3f km' % (v4d_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    ax.text(0.02, 0.65, 'v5d median = %.3f km' % (v5d_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    pyplot.savefig(join(out_dir, 'hist_cumsum_%04.1fm_%06.1fm.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)


def hist_plot_3(v4d_uv_dist, v5d_uv_dist, wave_length, out_dir):
    bin_width = wave_length
    num_vis = v4d_uv_dist.shape[0]
    assert(v4d_uv_dist.shape[0] == v5d_uv_dist.shape[0])
    uv_dist_max = 70.0e3
    uv_dist_min = 20.0
    num_bins = 200
    bins = numpy.logspace(numpy.log10(uv_dist_min),
                          numpy.log10(uv_dist_max), num_bins)
    v4d_hist, v4d_bin_edges = numpy.histogram(v4d_uv_dist, bins=bins)
    v5d_hist, v5d_bin_edges = numpy.histogram(v5d_uv_dist, bins=bins)
    v4d_hist = numpy.array(v4d_hist)
    v5d_hist = numpy.array(v5d_hist)
    v4d_bin_widths = numpy.diff(v4d_bin_edges)
    v5d_bin_widths = numpy.diff(v5d_bin_edges)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.divide(v4d_hist, v4d_bin_widths)
    ax.loglog(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v5d_bin_edges[:-1] + v5d_bin_widths / 2.0
    y = numpy.divide(v5d_hist, v5d_bin_widths)
    ax.loglog(x, y, 'r-', label='v5d', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Visibility count (m$^{-1}$)')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'hist_loglog_%04.1fm_%06.1fm_log_bins.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.cumsum(v4d_hist) / float(num_vis)
    ax.semilogx(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v5d_bin_edges[:-1] + v5d_bin_widths / 2.0
    y = numpy.cumsum(v5d_hist) / float(num_vis)
    ax.semilogx(x, y, 'r-', label='v5d', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Cumulative visibility density')
    ax.legend(loc=2)
    ax.grid()
    v4d_median = numpy.median(v4d_uv_dist)
    v5d_median = numpy.median(v5d_uv_dist)
    ax.plot([v4d_median, v4d_median], ax.get_ylim(), 'k:')
    ax.plot([v5d_median, v5d_median], ax.get_ylim(), 'r--')
    ax.text(0.02, 0.73, 'v4d median = %.3f km' % (v4d_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    ax.text(0.02, 0.65, 'v5d median = %.3f km' % (v5d_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    pyplot.savefig(join(out_dir, 'hist_cumsum_%04.1fm_%06.1fm_log_bins.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)


def plot_uv_image(uu, vv, cell_size, uv_max, station_radius_m, file_name):
    num_cells = int(ceil(uv_max / cell_size)) + 10
    num_cells += num_cells % 2
    num_cells *= 2
    print('- uv image size = %i x %i' % (num_cells, num_cells))
    image = numpy.zeros((num_cells, num_cells))
    centre = num_cells / 2
    for i in range(uu.shape[0]):
        if math.fabs(uu[i]) > uv_max or math.fabs(vv[i]) > uv_max:
            continue
        x = int(round(uu[i] / cell_size)) + centre
        y = int(round(vv[i] / cell_size)) + centre
        image[y, x] += 1
        x = int(round(-uu[i] / cell_size)) + centre
        y = int(round(-vv[i] / cell_size)) + centre
        image[y, x] += 1
    print('- uv_image: min %f max %f mean: %f std: %f'
          % (image.min(), image.max(), numpy.mean(image),
             numpy.std(image)))
    off = cell_size / 2.0
    extent = [-centre * cell_size - off, centre * cell_size - off,
              -centre * cell_size + off, centre * cell_size + off]

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(image, interpolation='nearest', extent=extent, cmap='gray_r')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(image.min(), image.max(), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if uv_max / station_radius_m < 20.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m,
                               color='r', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.set_xlim(-uv_max, uv_max)
    ax.set_ylim(-uv_max, uv_max)
    pyplot.savefig(file_name + '_%04.1f_%05.1fm.png' % (cell_size, uv_max))
    pyplot.close(fig)

    image /= image.max()
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='gray_r', norm=LogNorm(vmin=0.01, vmax=1.0))
    t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.ax.tick_params(labelsize='small')
    cbar.set_label('normalised uv count', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if uv_max / station_radius_m < 20.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m,
                               color='r', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.set_xlim(-uv_max, uv_max)
    ax.set_ylim(-uv_max, uv_max)
    pyplot.savefig(file_name + '_%04.1f_%05.1fm_log.png' % (cell_size, uv_max))
    pyplot.close(fig)


def make_psf(uu, vv, ww, ra, dec, freq, fov, im_size, file_name):
    imager = Imager('single')
    wave_length = 299792458.0 / freq
    uu_ = uu / wave_length
    vv_ = vv / wave_length
    ww_ = ww / wave_length
    amp = numpy.ones(uu.shape, dtype='c16')
    weight = numpy.ones(uu.shape, dtype='f8')
    image = imager.make_image(uu_, vv_, ww_, amp, weight, fov, im_size)

    cell = math.degrees(imager.fov_to_cellsize(math.radians(fov), im_size))
    lm_max = math.sin(0.5 * math.radians(fov))
    lm_inc = (2.0 * lm_max) / im_size
    off = lm_inc / 2.0
    extent = [-lm_max - off, lm_max - off,
              -lm_max + off, lm_max + off]

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='inferno')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(image.min(), image.max(), 7)
    ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    ax.set_xlabel('l', fontsize='small')
    ax.set_ylabel('m', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    pyplot.savefig(file_name + '_%05.2f_%04i.png' % (fov, im_size))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    # http://matplotlib.org/api/colors_api.html
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='inferno', norm=SymLogNorm(vmin=-0.01,
                                                   vmax=1.0,
                                                   linthresh=0.005,
                                                   linscale=0.75))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    ax.figure.colorbar(im, cax=cax)
    ax.set_xlabel('l', fontsize='small')
    ax.set_ylabel('m', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    pyplot.savefig(file_name + '_%05.2f_%04i_log.png' % (fov, im_size))
    pyplot.close(fig)

    save_fits_image(file_name + '_%05.2f_%04i.fits'
                    % (fov, im_size), image, cell, math.degrees(ra),
                    math.degrees(dec), freq)


def make_psf_images(uu_v4d, vv_v4d, ww_v4d, uu_v5d, vv_v5d, ww_v5d,
                    ra, dec, freq, out_dir):
    t0 = time.time()
    make_psf(uu_v4d, vv_v4d, ww_v4d, ra, dec, freq, 0.05, 512,
             join(out_dir, 'psf_v4d'))
    make_psf(uu_v5d, vv_v5d, ww_v5d, ra, dec, freq, 0.05, 512,
             join(out_dir, 'psf_v5d'))
    make_psf(uu_v4d, vv_v4d, ww_v4d, ra, dec, freq, 0.5, 2048,
             join(out_dir, 'psf_v4d'))
    make_psf(uu_v5d, vv_v5d, ww_v5d, ra, dec, freq, 0.5, 2048,
             join(out_dir, 'psf_v5d'))
    make_psf(uu_v4d, vv_v4d, ww_v4d, ra, dec, freq, 1.0, 2048,
             join(out_dir, 'psf_v4d'))
    make_psf(uu_v5d, vv_v5d, ww_v5d, ra, dec, freq, 1.0, 2048,
             join(out_dir, 'psf_v5d'))
    make_psf(uu_v4d, vv_v4d, ww_v4d, ra, dec, freq, 5.0, 2048,
             join(out_dir, 'psf_v4d'))
    make_psf(uu_v5d, vv_v5d, ww_v5d, ra, dec, freq, 5.0, 2048,
             join(out_dir, 'psf_v5d'))
    make_psf(uu_v4d, vv_v4d, ww_v4d, ra, dec, freq, 20.0, 2048,
             join(out_dir, 'psf_v4d'))
    make_psf(uu_v5d, vv_v5d, ww_v5d, ra, dec, freq, 20.0, 2048,
             join(out_dir, 'psf_v5d'))
    make_psf(uu_v4d, vv_v4d, ww_v4d, ra, dec, freq, 40.0, 2048,
             join(out_dir, 'psf_v4d'))
    make_psf(uu_v5d, vv_v5d, ww_v5d, ra, dec, freq, 40.0, 2048,
             join(out_dir, 'psf_v5d'))
    print('- psf images took %.2f s' % (time.time() - t0))


def plot_uv_images(uu_v4d, vv_v4d, uu_v5d, vv_v5d,
                   wave_length, station_radius_m, out_dir):
    t0 = time.time()
    plot_uv_image(uu_v4d, vv_v4d, wave_length, 100.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v5d, vv_v5d, wave_length, 100.0, station_radius_m,
                  join(out_dir, 'uv_image_v5d'))
    plot_uv_image(uu_v4d, vv_v4d, wave_length, 300.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v5d, vv_v5d, wave_length, 300.0, station_radius_m,
                  join(out_dir, 'uv_image_v5d'))
    plot_uv_image(uu_v4d, vv_v4d, wave_length * 10.0, 3000.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v5d, vv_v5d, wave_length * 10.0, 3000.0, station_radius_m,
                  join(out_dir, 'uv_image_v5d'))
    print('- uv images took %.2f s' % (time.time() - t0))


def plot_az_rms(uu_v4d, vv_v4d, wave_length, out_dir):
    v4d_uv_dist = (uu_v4d**2 + vv_v4d**2)**0.5
    v4d_uv_dist_idx = numpy.argsort(v4d_uv_dist)
    v4d_uv_dist = v4d_uv_dist[v4d_uv_dist_idx]
    uu_v4d = uu_v4d[v4d_uv_dist_idx]
    vv_v4d = vv_v4d[v4d_uv_dist_idx]

    bin_width = wave_length
    uv_dist_max = 70.0e3
    uv_dist_min = 20.0
    num_bins = int(ceil(uv_dist_max / bin_width))
    bins = numpy.arange(num_bins) * bin_width

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4d[:1000], vv_v4d[:1000], '+')
    for i in range(num_bins):
        if i < 20:
            circle = pyplot.Circle((0.0, 0.0), bins[i],
                                   color='r', fill=False, alpha=0.5,
                                   linewidth=1.0)
            ax.add_artist(circle)

    pyplot.show()


def plot_az_rms_2(uu_v4d, vv_v4d, uu_v5d, vv_v5d, wave_length, out_dir):

    v4d_r = (uu_v4d**2 + vv_v4d**2)**0.5
    v4d_theta = numpy.arctan2(vv_v4d, uu_v4d)
    v4d_r_idx = numpy.argsort(v4d_r)
    v4d_r = v4d_r[v4d_r_idx]
    v4d_theta = v4d_theta[v4d_r_idx]

    v5d_r = (uu_v5d**2 + vv_v5d**2)**0.5
    v5d_theta = numpy.arctan2(vv_v5d, uu_v5d)
    v5d_r_idx = numpy.argsort(v5d_r)
    v5d_r = v5d_r[v5d_r_idx]
    v5d_theta = v5d_theta[v5d_r_idx]

    num_bins_r = 200
    uv_dist_max = 70.0e3
    uv_dist_min = 20.0
    r_bins = numpy.logspace(numpy.log10(uv_dist_min),
                            numpy.log10(uv_dist_max), num_bins_r)
    num_bins_theta = 100
    theta_bins = numpy.linspace(-math.pi, math.pi, num_bins_theta)

    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.semilogx(v4d_r, v4d_theta, 'k.', alpha=0.3, ms=2.0)
    image_v4d = numpy.zeros((num_bins_theta-1, num_bins_r-1))
    for i in range(num_bins_r-1):
        rb_idx = numpy.where(numpy.logical_and(
            numpy.greater(v4d_r, r_bins[i]),
            numpy.less_equal(v4d_r, r_bins[i+1])))[0]
        rb_theta = v4d_theta[rb_idx]
        for j in range(num_bins_theta-1):
            tb_idx = numpy.where(numpy.logical_and(
                numpy.greater(rb_theta, theta_bins[j]),
                numpy.less_equal(rb_theta, theta_bins[j+1])))[0]
            image_v4d[j, i] = tb_idx.shape[0]
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('Azimuth [radians]')
    pyplot.savefig(join(out_dir, 'scatter_v4d_r_theta.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.semilogx(v5d_r, v5d_theta, 'k.', alpha=0.3, ms=2.0)
    image_v5d = numpy.zeros((num_bins_theta-1, num_bins_r-1))
    for i in range(num_bins_r-1):
        rb_idx = numpy.where(numpy.logical_and(
            numpy.greater(v5d_r, r_bins[i]),
            numpy.less_equal(v5d_r, r_bins[i+1])))[0]
        rb_theta = v5d_theta[rb_idx]
        for j in range(num_bins_theta-1):
            tb_idx = numpy.where(numpy.logical_and(
                numpy.greater(rb_theta, theta_bins[j]),
                numpy.less_equal(rb_theta, theta_bins[j+1])))[0]
            image_v5d[j, i] = tb_idx.shape[0]
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('Azimuth [radians]')
    pyplot.savefig(join(out_dir, 'scatter_v5d_r_theta.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(numpy.flipud(image_v4d), interpolation='nearest', cmap='gray_r',
                   norm=LogNorm(vmin=1.0, vmax=image_v4d.max()))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    cbar.set_label('uv count', fontsize='small')
    ax.set_xlabel('Radius [m]', fontsize='small')
    ax.set_ylabel('Azimuth [radians]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Azimuth')
    pyplot.savefig(join(out_dir, 'scatter_v4d_r_theta_image.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(numpy.flipud(image_v5d), interpolation='nearest', cmap='gray_r',
                   norm=LogNorm(vmin=1.0, vmax=image_v5d.max()))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    cbar.set_label('uv count', fontsize='small')
    ax.set_xlabel('Radius [m]', fontsize='small')
    ax.set_ylabel('Azimuth [radians]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Azimuth')
    pyplot.savefig(join(out_dir, 'scatter_v5d_r_theta_image.png'))
    pyplot.close(fig)

    az_rms_v4d = numpy.std(image_v4d, axis=0)
    az_rms_v5d = numpy.std(image_v5d, axis=0)
    rx = r_bins[:-1] + numpy.diff(r_bins) / 2.0
    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.loglog(rx, az_rms_v4d, 'k-', label='v4d')
    ax.loglog(rx, az_rms_v5d, 'r-', label='v5d')
    ax.plot([35, 35], ax.get_ylim(), '--')
    ax.legend()
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('uv count RMS')
    pyplot.savefig(join(out_dir, 'theta_rms.png'))
    pyplot.close(fig)

    az_sum_v4d = numpy.sum(image_v4d, axis=0)
    az_sum_v5d = numpy.sum(image_v5d, axis=0)
    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.loglog(rx, az_sum_v4d, 'k-', label='v4d')
    ax.loglog(rx, az_sum_v5d, 'r-', label='v5d')
    # ax.plot([35, 35], ax.get_ylim(), 'b:')
    ax.legend()
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('uv count sum')
    pyplot.savefig(join(out_dir, 'theta_sum.png'))
    pyplot.close(fig)

    rx = r_bins[:-1] + numpy.diff(r_bins) / 2.0
    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    idx_non_zero_v4d = numpy.where(az_sum_v4d > 0.0)
    idx_non_zero_v5d = numpy.where(az_sum_v5d > 0.0)
    az_rms_v4d = az_rms_v4d[idx_non_zero_v4d]
    az_rms_v5d = az_rms_v5d[idx_non_zero_v5d]
    az_sum_v4d = az_sum_v4d[idx_non_zero_v4d]
    az_sum_v5d = az_sum_v5d[idx_non_zero_v5d]
    ax.loglog(rx[idx_non_zero_v4d], numpy.divide(az_rms_v4d, az_sum_v4d),
              'k-', label='v4d')
    ax.loglog(rx[idx_non_zero_v5d], numpy.divide(az_rms_v5d, az_sum_v5d),
              'r-', label='v5d')
    # ax.plot([35, 35], ax.get_ylim(), 'b:')
    ax.legend(loc=4)
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('uv count rms / uv count sum')
    pyplot.savefig(join(out_dir, 'theta_norm_rms.png'))
    pyplot.close(fig)




def main():
    # Load station positions
    t0 = time.time()
    v4d_file = join('v4d.tm', 'layout_enu_stations.txt')
    v5d_file = join('v5d.tm', 'layout_enu_stations.txt')
    v4d = numpy.loadtxt(v4d_file)
    v5d = numpy.loadtxt(v5d_file)
    station_radius_m = 35.0 / 2.0
    num_stations = v4d.shape[0]
    assert(v5d.shape[0] == v4d.shape[0])
    print('- loading coordinates took %.2f s' % (time.time() - t0))

    freq = 100.0e6
    wave_length = 299792458.0 / freq
    lon = radians(116.63128900)
    lat = radians(-26.69702400)
    alt = 0.0
    ra = radians(68.698903779331502)
    dec = radians(-26.568851215532160)
    mjd_mid = 57443.4375000000

    snapshot = True
    if snapshot:
        mjd_start = mjd_mid
        obs_length = 0.0
        dt_s = 0.0
        num_times = 1
    else:
        obs_length = 4.0 * 3600.0  # seconds
        num_times = int(obs_length / (3 * 60.0))
        dt_s = obs_length / float(num_times)
        mjd_start = mjd_mid - (obs_length / 2.0) / (3600.0 * 24.0)

    print('- obs_length = %.2f s (%.2f h)' % (obs_length, obs_length / 3600.0))
    print('- num_times =', num_times)

    num_baselines = num_stations * (num_stations - 1) / 2
    out_dir = 'uv_%3.1fh' % (obs_length / 3600.0)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    t0 = time.time()
    x, y, z = convert_enu_to_ecef(v4d[:, 0], v4d[:, 1], v4d[:, 2],
                                  lon, lat, alt)
    uu_v4d, vv_v4d, ww_v4d = \
        generate_baseline_uvw(x, y, z, ra, dec, num_times, num_baselines,
                              mjd_start, dt_s)
    x, y, z = convert_enu_to_ecef(v5d[:, 0], v5d[:, 1], v5d[:, 2],
                                  lon, lat, alt)
    uu_v5d, vv_v5d, ww_v5d = \
        generate_baseline_uvw(x, y, z, ra, dec, num_times, num_baselines,
                              mjd_start, dt_s)
    print('- coordinate generation took %.2f s' % (time.time() - t0))
    print('- num vis = %i' % uu_v4d.shape[0])

    # t0 = time.time()
    # uv_plot(uu_v4d, vv_v4d, uu_v5d, vv_v5d, out_dir)
    # print('- uv scatter plot took %.2f s' % (time.time() - t0))
    #
    # t0 = time.time()
    # v4d_uv_dist = (uu_v4d**2 + vv_v4d**2)**0.5
    # v4d_uv_dist.sort()
    # v5d_uv_dist = (uu_v5d**2 + vv_v5d**2)**0.5
    # v5d_uv_dist.sort()
    # hist_plot_1(v4d_uv_dist, v5d_uv_dist, wave_length, 300.0, out_dir)
    # hist_plot_1(v4d_uv_dist, v5d_uv_dist, wave_length * 5.0, 1500.0, out_dir)
    # hist_plot_1(v4d_uv_dist, v5d_uv_dist, wave_length, 1500.0, out_dir)
    # hist_plot_1(v4d_uv_dist, v5d_uv_dist, wave_length * 10.0, 3000.0, out_dir)
    # print('- histograms took %.2f s' % (time.time() - t0))
    #
    # hist_plot_2(v4d_uv_dist, v5d_uv_dist, wave_length, out_dir)
    # hist_plot_3(v4d_uv_dist, v5d_uv_dist, wave_length, out_dir)
    #
    # plot_uv_images(uu_v4d, vv_v4d, uu_v5d, vv_v5d, wave_length,
    #                station_radius_m, out_dir)
    #
    # make_psf_images(uu_v4d, vv_v4d, ww_v4d, uu_v5d, vv_v5d, ww_v5d, ra, dec,
    #                 freq, out_dir)

    # plot_az_rms(uu_v4d, vv_v4d, wave_length, out_dir)
    plot_az_rms_2(uu_v4d, vv_v4d, uu_v5d, vv_v5d, wave_length, out_dir)


if __name__ == '__main__':
    main()
