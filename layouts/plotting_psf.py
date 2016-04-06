# -*- coding: utf-8 -*-

from __future__ import print_function
from os.path import join
import os
import numpy
import math
import matplotlib.pyplot as pyplot
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
try:
    from oskar.imager import Imager
    oskar_imager_found = True
except ImportError:
    print('OSKAR python imager not found, PSF images wont be made.')
    oskar_imager_found = False


def plot_psf(uu_v4d, vv_v4d, ww_v4d, uu_v4o1, vv_v4o1, ww_v4o1,
             freq, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    t0 = time.time()
    psf_v4d = make_psf(uu_v4d, vv_v4d, ww_v4d, freq, 15.0, 1024)
    psf_v4o1 = make_psf(uu_v4o1, vv_v4o1, ww_v4o1, freq, 15.0, 1024)
    plot_psf_images(psf_v4d, psf_v4o1, out_dir)
    plot_psf_az_ave(psf_v4d, psf_v4o1, out_dir)
    print('- PSF plotting took %.2f s' % (time.time() - t0))


def image_coords(im_size, lm_inc):
    x = numpy.arange(-im_size / 2, im_size / 2) * lm_inc
    x, y = numpy.meshgrid(x, x[::-1])
    r = (x**2 + y**2)**0.5
    return x, y, r


def bin_az_ave_psf(x, y, num_bins, im_size, lm_inc):
    abs_y = numpy.abs(y)
    y_max = numpy.nanmax(abs_y)
    y_db = 10.0 * numpy.log10(abs_y / y_max)
    bins = {'x': numpy.zeros(num_bins),
            'mean': numpy.zeros(num_bins),
            'std': numpy.zeros(num_bins),
            'mean_db': numpy.zeros(num_bins),
            'std_db': numpy.zeros(num_bins)}
    bin_inc = (im_size / 2.0 * lm_inc) / float(num_bins)
    for i in range(num_bins):
        r0 = i * bin_inc
        r1 = r0 + bin_inc
        bins['x'][i] = r0 + (r1 - r0) / 2.0
        idx = numpy.where(numpy.logical_and(x > r0, x <= r1))
        y_bin = y[idx]
        bins['mean'][i] = y_bin.mean()
        bins['std'][i] = y_bin.std()
        y_db_bin = y_db[idx]
        bins['mean_db'][i] = y_db_bin.mean()
        bins['std_db'][i] = y_db_bin.std()
    return bins, y_db


def plot_psf_az_ave(psf_v4d, psf_v4o1, out_dir):

    image_v4d = psf_v4d['image']
    image_v4o1 = psf_v4o1['image']
    lm_inc = psf_v4d['lm_inc']
    fov = psf_v4d['fov']
    im_size = psf_v4d['im_size']
    num_bins = 200

    # Radially average PSF image, sort points by radius.
    _, _, r = image_coords(im_size, lm_inc)
    x = r.flatten()
    sort_idx = numpy.argsort(x)
    x = x[sort_idx]
    y_v4d = image_v4d.flatten()
    y_v4d = y_v4d[sort_idx]
    y_v4o1 = image_v4o1.flatten()
    y_v4o1 = y_v4o1[sort_idx]

    # Bin
    bins_v4d, y_v4d_db = bin_az_ave_psf(x, y_v4d, num_bins, im_size, lm_inc)
    bins_v4o1, y_v4o1_db = bin_az_ave_psf(x, y_v4o1, num_bins, im_size, lm_inc)

    # Plot with linear scale
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    xb1 = bins_v4d['x']
    yb1 = bins_v4d['mean']
    yb1_std = bins_v4d['std']
    ax.plot(xb1, yb1, 'b-', label='v4d')
    ax.fill_between(xb1, yb1 - yb1_std, yb1 + yb1_std, edgecolor='blue',
                    facecolor='blue', alpha=0.2)
    xb2 = bins_v4o1['x']
    yb2 = bins_v4o1['mean']
    yb2_std = bins_v4o1['std']
    ax.plot(xb2, yb2, 'r-', label='v4o1')
    ax.fill_between(xb2, yb2 - yb2_std, yb2 + yb2_std, edgecolor='red',
                    facecolor='red', alpha=0.2)
    ax.set_ylim(-0.005, 0.025)
    ax.set_xlim(0.0, (im_size / 2) * lm_inc)
    ax.set_xlabel('Radius (direction cosine)')
    ax.set_ylabel('Radially averaged PSF')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'az_ave_%03.1f_lin.png' % fov))
    pyplot.close(fig)

    # Plot with log scale
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    xb1 = bins_v4d['x']
    yb1 = numpy.abs(bins_v4d['mean'])
    yb1_std = bins_v4d['std']
    ax.plot(xb1, yb1, 'b-', label='v4d')
    ax.fill_between(xb1, yb1 - yb1_std, yb1 + yb1_std, edgecolor='blue',
                    facecolor='blue', alpha=0.2)
    xb2 = bins_v4o1['x']
    yb2 = numpy.abs(bins_v4o1['mean'])
    yb2_std = bins_v4o1['std']
    ax.plot(xb2, yb2, 'r-', label='v4o1')
    ax.fill_between(xb2, yb2 - yb2_std, yb2 + yb2_std, edgecolor='red',
                    facecolor='red', alpha=0.2)
    ax.set_xlim(0.0, (im_size / 2) * lm_inc)
    ax.set_xlabel('Radius (direction cosine)')
    ax.set_ylabel('Abs. radially averaged PSF')
    ax.legend()
    ax.grid()
    ax.set_yscale('log', noposy='clip')
    ax.set_ylim(1.0e-5, 1.0)
    pyplot.savefig(join(out_dir, 'az_ave_%03.1f_log.png' % fov))
    pyplot.close(fig)


def make_psf(uu, vv, ww, freq, fov, im_size):
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
    return {'image': image, 'fov': fov, 'im_size': im_size, 'cell': cell,
            'lm_max': lm_max, 'lm_inc': lm_inc}


def plot_image_lin(image, extent, fov, file_name):
    font_size_ = 'small'
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='inferno')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size_)
    t = numpy.linspace(image.min(), image.max(), 7)
    ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    ax.set_xlabel('l', fontsize=font_size_)
    ax.set_ylabel('m', fontsize=font_size_)
    ax.tick_params(axis='both', which='major', labelsize=font_size_)
    ax.tick_params(axis='both', which='minor', labelsize=font_size_)
    pyplot.savefig(file_name + '_%03.1f_image_lin.png' % fov)
    pyplot.close(fig)


def plot_image_log(image, extent, fov, file_name):
    font_size_ = 'small'
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='inferno', norm=SymLogNorm(vmin=-0.01, vmax=1.0,
                                                   linthresh=0.01,
                                                   linscale=0.75))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size_)
    ax.figure.colorbar(im, cax=cax)
    ax.set_xlabel('l', fontsize=font_size_)
    ax.set_ylabel('m', fontsize=font_size_)
    ax.tick_params(axis='both', which='major', labelsize=font_size_)
    ax.tick_params(axis='both', which='minor', labelsize=font_size_)
    pyplot.savefig(file_name + '_%03.1f_image_log.png' % fov)
    pyplot.close(fig)


def plot_psf_images(psf_v4d, psf_v4o1, out_dir):
    lm_max = psf_v4d['lm_max']
    lm_inc = psf_v4d['lm_inc']
    fov = psf_v4d['fov']
    off = lm_inc / 2.0
    extent = [-lm_max - off, lm_max - off, -lm_max + off, lm_max + off]
    # plot_image_lin(psf_v4d['image'], extent, fov, join(out_dir, 'v4d'))
    # plot_image_lin(psf_v4o1['image'], extent, fov, join(out_dir, 'v4o1'))
    plot_image_log(psf_v4d['image'], extent, fov, join(out_dir, 'v4d'))
    plot_image_log(psf_v4o1['image'], extent, fov, join(out_dir, 'v4o1'))
