# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy
from math import radians, ceil
import shutil
import matplotlib.pyplot as pyplot
from matplotlib.colors import SymLogNorm, LogNorm
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
from os.path import join
try:
    from oskar.imager import Imager
    oskar_imager_found = True
except ImportError:
    print('OSKAR python imager not found, PSF images wont be made.')
    oskar_imager_found = False
import pyfits
import math
import scipy.signal
from plotting_uv_hist import plot_uv_hist


def bin_az_ave(x, y, num_bins, im_size, cell_size_uv):
    bins = {'x': numpy.zeros(num_bins),
            'mean': numpy.zeros(num_bins),
            'std': numpy.zeros(num_bins),
            'sum': numpy.zeros(num_bins),
            'n': numpy.zeros(num_bins),}
    bin_inc = (im_size / 2.0 * cell_size_uv) / float(num_bins)
    y_total = 0
    sum_total = 0.0
    for i in range(num_bins):
        r0 = i * bin_inc
        r1 = r0 + bin_inc
        bins['x'][i] = r0 + (r1 - r0) / 2.0
        idx = numpy.where(numpy.logical_and(x > r0, x <= r1))
        y_bin = y[idx]
        if len(idx[0]) == 0:
            continue
        bins['mean'][i] = y_bin.mean()
        bins['std'][i] = y_bin.std()
        bins['sum'][i] = numpy.sum(y_bin)
        bins['n'][i] = len(idx[0])
        y_total += bins['n'][i]
        sum_total += numpy.sum(y_bin)
    print(y_total, sum_total)
    return bins


def plot_image_lin(image, extent, file_name):
    font_size_ = 'small'
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im_min = image.min()
    im_max = image.max()
    im_min = 1.0e-6
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='inferno', vmin=im_min, vmax=im_max)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size_)
    t = numpy.linspace(im_max, im_min, 7)
    ax.figure.colorbar(im, cax=cax, ticks=t, format='%.3f')
    ax.set_xlabel('l', fontsize=font_size_)
    ax.set_ylabel('m', fontsize=font_size_)
    ax.tick_params(axis='both', which='major', labelsize=font_size_)
    ax.tick_params(axis='both', which='minor', labelsize=font_size_)
    ax.grid(True)
    pyplot.savefig(file_name + '_lin.png')
    pyplot.close(fig)


def plot_image_log(image, extent, file_name):
    font_size_ = 'small'
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im_min = image.min()
    im_max = image.max()
    # im = ax.imshow(image, interpolation='nearest', extent=extent,
    #                cmap='inferno', norm=SymLogNorm(vmin=-0.001, vmax=1.0,
    #                                                linthresh=0.001,
    #                                                linscale=0.75))
    im_min = 1.0e-6
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='inferno', norm=LogNorm(vmin=im_min, vmax=1.0))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size_)
    ax.figure.colorbar(im, cax=cax)
    ax.set_xlabel('l', fontsize=font_size_)
    ax.set_ylabel('m', fontsize=font_size_)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size_)
    ax.tick_params(axis='both', which='minor', labelsize=font_size_)
    pyplot.savefig(file_name + '_log.png')
    pyplot.close(fig)


def nanargmax(a):
    idx = numpy.argmax(a, axis=None)
    multi_idx = numpy.unravel_index(idx, a.shape)
    if numpy.isnan(a[multi_idx]):
        nan_count = numpy.sum(numpy.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = numpy.argpartition(a, -nan_count - 1, axis=None)[-nan_count - 1]
        multi_idx = numpy.unravel_index(idx, a.shape)
    return multi_idx


def generate_uv_coordinates(x, y, z, lon, lat, alt, ra, dec, num_times,
                            num_baselines, mjd_start, dt_s, freq_hz,
                            uv_cut_radius_wavelengths):
    x, y, z = convert_enu_to_ecef(x, y, z, lon, lat, alt)
    uu, vv, ww = generate_baseline_uvw(x, y, z, ra, dec, num_times,
                                       num_baselines, mjd_start, dt_s)
    wave_length = 299792458.0 / freq_hz
    uu /= wave_length
    vv /= wave_length
    ww /= wave_length
    # uv_r_cut = uv_cut_radius_wavelengths
    # uv_r = (uu ** 2 + vv ** 2) ** 0.5
    # uv_sort_idx = numpy.argsort(uv_r)
    # uv_r = uv_r[uv_sort_idx]
    # uu = uu[uv_sort_idx]
    # vv = vv[uv_sort_idx]
    # ww = ww[uv_sort_idx]
    # i_uv_max = numpy.argmax(uv_r >= uv_r_cut)
    # uu = uu[:i_uv_max]
    # vv = vv[:i_uv_max]
    # ww = ww[:i_uv_max]
    uu = numpy.hstack((uu, -uu))
    vv = numpy.hstack((vv, -vv))
    ww = numpy.hstack((ww, -ww))
    return uu, vv, ww


def plot_uv_grid(grid_v4d, grid_v4o1, extent_m, station_radius_m, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # Plot UV grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4d
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4d, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(grid_v4d.min(), grid_v4d.max(), 10)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if abs(extent_m[1]) / station_radius_m < 5.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    # ax.plot(uu_v4d * wave_length, vv_v4d * wave_length, 'r.', ms=2.0,
    # alpha=0.5)
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_v4d.png'))
    pyplot.close(fig)

    # Plot UV grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4o1
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4o1, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(grid_v4o1.min(), grid_v4o1.max(), 14)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if abs(extent_m[1]) / station_radius_m < 5.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_v4o1.png'))
    pyplot.close(fig)

    # Plot UV grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4d
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4d, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower',
                   norm=LogNorm(vmin=1.0, vmax=grid_v4d.max()))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.logspace(numpy.log10(1.0), numpy.log10(grid_v4d.max()), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if abs(extent_m[1]) / station_radius_m < 5.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_v4d_log.png'))
    pyplot.close(fig)

    # Plot UV grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4o1
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4o1, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower',
                   norm=LogNorm(vmin=1.0, vmax=grid_v4o1.max()))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.logspace(numpy.log10(1.0), numpy.log10(grid_v4o1.max()), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if abs(extent_m[1]) / station_radius_m < 5.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_v4o1_log.png'))
    pyplot.close(fig)


def plot_uv_scatter(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length,
                    extent_m, out_dir):
    # Plot UV scatter comparing the two telescopes
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4d * wave_length, vv_v4d * wave_length, 'b.', ms=2.0,
            alpha=0.2, label='v4d')
    ax.plot(uu_v4o1 * wave_length, vv_v4o1 * wave_length, 'r.', ms=2.0,
            alpha=0.2, label='v4o1')
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.legend()
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    fig.savefig(join(out_dir, 'uv_scatter.png'))
    ax.grid(True)
    pyplot.close(fig)


def grid_uv_data(uu_v4d, vv_v4d, ww_v4d, uu_v4o1, vv_v4o1, ww_v4o1, im_size,
                 imager):
    # Create UV grid images
    grid_v4d = numpy.zeros([im_size, im_size], dtype='c16')
    amps_v4d = numpy.ones(uu_v4d.shape, dtype='c16')
    weight_v4d = numpy.ones(uu_v4d.shape, dtype='f8')
    plane_norm = 0.0
    imager.update_plane(uu_v4d, vv_v4d, ww_v4d, amps_v4d, weight_v4d,
                        grid_v4d, plane_norm)
    assert (numpy.sum(numpy.abs(grid_v4d.imag)) == 0.0)

    grid_v4d = numpy.real(grid_v4d)
    grid_v4o1 = numpy.zeros((im_size, im_size), dtype='c16')
    amps_v4o1 = numpy.ones(uu_v4o1.shape, dtype='c16')
    weight_v4o1 = numpy.ones(uu_v4o1.shape, dtype='f8')
    imager.update_plane(uu_v4o1, vv_v4o1, ww_v4o1, amps_v4o1, weight_v4o1,
                        grid_v4o1, 0.0)
    assert (numpy.sum(numpy.abs(grid_v4o1.imag)) == 0.0)
    grid_v4o1 = numpy.real(grid_v4o1)
    return grid_v4d, grid_v4o1


def plot_uv_beam_kernel(uv_beam, y_cut, x_cut, pb_kernel,
                        cell_size_uv_wavelengths, wave_length, station_radius_m,
                        uv_dist_max, uv_pb_cut_level, dx, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # Plot the kernel cut profile
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    zz = min(200, uv_beam.shape[0] / 2)
    xx = numpy.arange(-zz, zz) * cell_size_uv_wavelengths * wave_length
    ax.semilogy(xx, y_cut[uv_beam.shape[0] / 2 - zz: uv_beam.shape[0] / 2 + zz],
                'r')
    ax.semilogy(xx, x_cut[uv_beam.shape[1] / 2 - zz: uv_beam.shape[1] / 2 + zz],
                'b')
    ax.plot([xx[0], xx[-1]], [uv_pb_cut_level, uv_pb_cut_level], 'k', lw=3.0,
            alpha=2.0)
    ax.grid(True)
    ax.set_ylabel('abs. uv beam response')
    ax.set_xlabel('metres')
    ax.set_xlim(-zz * dx * wave_length, zz * dx * wave_length)
    fig.savefig(join(out_dir, 'uv_beam_cut_abs_log.png'))
    pyplot.close(fig)

    # Plot the kernel cut profile
    fig = pyplot.figure(figsize=(8, 8))
    x_cut = uv_beam[:, uv_beam.shape[0] / 2].real
    y_cut = uv_beam[uv_beam.shape[0] / 2, :].real
    ax = fig.add_subplot(111)
    zz = uv_beam.shape[0] / 2
    xx = numpy.arange(-zz, zz) * cell_size_uv_wavelengths * wave_length
    ax.plot(xx, y_cut, 'r')
    ax.plot(xx, x_cut, 'b')
    # pyplot.yscale('symlog', linthreshy=0.00001)
    # pyplot.yscale('symlog')
    ax.set_xlabel('metres')
    ax.set_xlim(-zz * dx * wave_length, zz * dx * wave_length)
    ax.plot([xx[0], xx[-1]], [uv_pb_cut_level, uv_pb_cut_level], 'k', lw=3.0,
            alpha=2.0)
    ax.grid(True)
    fig.savefig(join(out_dir, 'uv_beam_cut_lin.png'))
    pyplot.close(fig)

    # Plot the uv primary beam kernel
    c_pb = pb_kernel.shape[0] / 2
    extent_pb_wavelengths = [c_pb * dx, -c_pb * dx, -c_pb * dx, c_pb * dx]
    extent_pb_m = numpy.array(extent_pb_wavelengths) * wave_length
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(pb_kernel, interpolation='nearest',
                   extent=extent_pb_m, cmap='gray_r', origin='lower')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(pb_kernel.min(), pb_kernel.max(), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('amplitude', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlim(extent_pb_m[0], extent_pb_m[1])
    ax.set_ylim(extent_pb_m[2], extent_pb_m[3])
    circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                           color='b', fill=False, alpha=0.5,
                           linewidth=1.0)
    ax.add_artist(circle)
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_beam_kernel.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pbc = pb_kernel.shape[0] / 2
    pbx = numpy.arange(-pbc, pbc + 1) * cell_size_uv_wavelengths * wave_length
    ax.plot(pbx, pb_kernel[pbc, :])
    ax.grid()
    ax.set_xlabel('metres')
    pyplot.savefig(join(out_dir, 'uv_beam_kernel_1d.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    pbc = pb_kernel.shape[0] / 2
    pbx = numpy.arange(-pbc, pbc + 1) * cell_size_uv_wavelengths * wave_length
    ax.plot(pbx, pb_kernel[pbc, :])
    ax.grid()
    ax.set_xlabel('metres')
    ax.set_xlim(-uv_dist_max / 2, uv_dist_max / 2)
    pyplot.savefig(join(out_dir, 'uv_beam_kernel_1d_v2.png'))
    pyplot.close(fig)

    # plot uv beam kernel log scale
    c_pb = pb_kernel.shape[0] / 2
    temp = pb_kernel / numpy.nanmax(pb_kernel)
    extent_pb_wavelengths = [c_pb * dx, -c_pb * dx, -c_pb * dx, c_pb * dx]
    extent_pb_m = numpy.array(extent_pb_wavelengths) * wave_length
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(temp, interpolation='nearest',
                   extent=extent_pb_m, cmap='gray_r', origin='lower',
                   norm=LogNorm())
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    # t = numpy.linspace(pb_kernel.min(), pb_kernel.max(), 7)
    # cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar = ax.figure.colorbar(im, cax=cax, format='%.2e')
    cbar.set_label('amplitude', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlim(extent_pb_m[0], extent_pb_m[1])
    ax.set_ylim(extent_pb_m[2], extent_pb_m[3])
    circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                           color='b', fill=False, alpha=0.5,
                           linewidth=1.0)
    ax.add_artist(circle)
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_beam_kernel_log.png'))
    pyplot.close(fig)


def plot_convolved_grid(grid_v4d_pb, grid_v4o1_pb, extent_m,
                        station_radius_m, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # Plot UV convolved grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4d
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4d_pb, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(grid_v4d_pb.min(), grid_v4d_pb.max(), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if abs(extent_m[1]) / station_radius_m < 5.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_v4d.png'))
    pyplot.close(fig)

    # Plot UV convolved grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4o1
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4o1_pb, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(grid_v4o1_pb.min(), grid_v4o1_pb.max(), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if abs(extent_m[1]) / station_radius_m < 5.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_v4o1.png'))
    pyplot.close(fig)

    # Plot UV convolved grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4d
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4d_pb, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower',
                   norm=LogNorm(vmin=1.0, vmax=grid_v4d_pb.max()))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.logspace(numpy.log10(1.0), numpy.log10(grid_v4d_pb.max()), 9)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_v4d_log.png'))
    pyplot.close(fig)

    # Plot UV convolved grid image with scatter overlay of uv coordinates
    # (wavelength axis scale) v4o1
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4o1_pb, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower',
                   norm=LogNorm(vmin=1.0, vmax=grid_v4o1_pb.max()))
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.logspace(numpy.log10(1.0), numpy.log10(grid_v4o1_pb.max()), 9)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_v4o1_log.png'))
    pyplot.close(fig)


def plot_az_uv_grid_scatter(grid_r_wavelengths, y_v4d, y_v4o1,
                            cell_size_uv_wavelengths, wave_length,
                            im_size, out_dir):
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(grid_r_wavelengths * wave_length, y_v4d, 'bo', mew=0.0, ms=5.0,
            alpha=0.05, label='v4d')
    ax.plot(grid_r_wavelengths * wave_length, y_v4o1, 'rx', mew=1.0, ms=5.0,
            alpha=0.05, label='v4o1')
    ax.set_xlabel('Baseline length [m]')
    ax.set_ylabel('Baseline count')
    ax.grid(True)
    ax.set_xlim(0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    ax.legend()
    fig.savefig(join(out_dir, 'uv_grid_convolved_az_scatter.png'))
    pyplot.close(fig)


def plot_az_uv_grid_bin_mean_std(grid_r_wavelengths, y_v4d, y_v4o1,
                                 num_bins, wave_length,
                                 cell_size_uv_wavelengths, im_size, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    bins_v4d = bin_az_ave(grid_r_wavelengths, y_v4d, num_bins, im_size,
                          cell_size_uv_wavelengths)
    bins_v4o1 = bin_az_ave(grid_r_wavelengths, y_v4o1, num_bins, im_size,
                           cell_size_uv_wavelengths)

    # Plot with linear scale
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    xb1 = bins_v4d['x'] * wave_length
    yb1 = bins_v4d['mean']
    yb1_std = bins_v4d['std']
    ax.plot(xb1, yb1, 'b-', label='v4d')
    ax.fill_between(xb1, yb1 - yb1_std, yb1 + yb1_std, edgecolor='blue',
                    facecolor='blue', alpha=0.2)
    xb2 = bins_v4o1['x'] * wave_length
    yb2 = bins_v4o1['mean']
    yb2_std = bins_v4o1['std']
    ax.plot(xb2, yb2, 'r-', label='v4o1')
    ax.fill_between(xb2, yb2 - yb2_std, yb2 + yb2_std, edgecolor='red',
                    facecolor='red', alpha=0.2)
    ax.set_xlim(0.0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    ax.set_xlabel('Baseline length (metres)')
    ax.set_ylabel('Azimuthal bin mean')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_az_binned_mean.png'))
    pyplot.close(fig)

    # SUM - linear
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    xb1 = bins_v4d['x'] * wave_length
    yb1 = bins_v4d['sum']
    ax.plot(xb1, yb1, 'b-', label='v4d')
    xb2 = bins_v4o1['x'] * wave_length
    yb2 = bins_v4o1['sum']
    ax.plot(xb2, yb2, 'r-', label='v4o1')
    ax.set_xlim(0.0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    ax.set_xlabel('UV distance (metres)')
    ax.set_ylabel('Azimuthal bin baseline count')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_az_binned_sum.png'))
    pyplot.close(fig)

    # SUM - loglog
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    xb1 = bins_v4d['x'] * wave_length
    yb1 = bins_v4d['sum']
    ax.loglog(xb1, yb1, 'b-', label='v4d')
    xb2 = bins_v4o1['x'] * wave_length
    yb2 = bins_v4o1['sum']
    ax.loglog(xb2, yb2, 'r-', label='v4o1')
    ax.set_xlim(0.0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    ax.set_xlabel('Baseline length (metres)')
    ax.set_ylabel('Azimuthal bin baseline count')
    # http://matplotlib.org/api/legend_api.html
    ax.legend(loc=4)
    ax.grid()
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_az_binned_sum_log.png'))
    pyplot.close(fig)

    # STD - loglog
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    xb1 = bins_v4d['x'][1:] * wave_length
    yb1 = bins_v4d['std'][1:] / bins_v4d['n'][1:]
    ax.loglog(xb1, yb1, 'b-', label='v4d')
    xb2 = bins_v4o1['x'][1:] * wave_length
    yb2 = bins_v4o1['std'][1:] / bins_v4o1['n'][1:]
    ax.loglog(xb2, yb2, 'r-', label='v4o1')
    ax.set_xlim(0.0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    ax.set_xlabel('Baseline length (metres)')
    ax.set_ylabel('Azimuthal bin STD / count')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'uv_grid_convolved_az_binned_std_log.png'))
    pyplot.close(fig)


def plot_unconvolved_hist(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length,
                          uv_dist_max, bins, out_dir):
    v4d_uv_dist = ((uu_v4d*wave_length)**2 + (vv_v4d*wave_length)**2)**0.5
    v4d_uv_dist.sort()
    v4o1_uv_dist = ((uu_v4o1*wave_length)**2 + (vv_v4o1*wave_length)**2)**0.5
    v4o1_uv_dist.sort()
    v4d_uv_dist = v4d_uv_dist[v4d_uv_dist < uv_dist_max * 2.0]
    v4o1_uv_dist = v4o1_uv_dist[v4o1_uv_dist < uv_dist_max * 2.0]
    v4d_hist, v4d_bin_edges = numpy.histogram(v4d_uv_dist, bins=bins)
    v4o1_hist, v4o1_bin_edges = numpy.histogram(v4o1_uv_dist, bins=bins)
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1]
    y = v4d_hist
    y = numpy.array(y, dtype='f8')
    ax.bar(x, y, width=numpy.diff(v4d_bin_edges), color='k', fill=True,
           alpha=0.5, lw=0.0, edgecolor='k', label='v4d')
    x = v4o1_bin_edges[:-1]
    y = v4o1_hist
    y = numpy.array(y, dtype='f8')
    ax.bar(x, y, width=numpy.diff(v4o1_bin_edges), color='r', fill=True,
           alpha=0.5, lw=0.0, edgecolor='r', label='v4o1')
    ax.set_xlim(0, uv_dist_max)
    ax.set_xlabel('baseline length (metres)')
    ax.set_ylabel('Baseline count')
    ax.legend(fontsize='small')
    pyplot.savefig(join(out_dir, 'hist.png'))
    pyplot.close(fig)


def plot_convolved_hist(y_v4d, y_v4o1, grid_r_wavelengths, wave_length, bins,
                        uv_dist_max, out_dir):
    grid_r_wavelengths = grid_r_wavelengths[grid_r_wavelengths * wave_length <
                                            uv_dist_max * 2.0]
    y_v4d = y_v4d[grid_r_wavelengths * wave_length < uv_dist_max * 2.0]
    y_v4o1 = y_v4o1[grid_r_wavelengths * wave_length < uv_dist_max * 2.0]
    v4d_grid_hist, v4d_grid_bin_edges = numpy.histogram(
        grid_r_wavelengths * wave_length,
        weights=y_v4d, bins=bins)
    v4o1_grid_hist, v4o1_grid_bin_edges = \
        numpy.histogram(grid_r_wavelengths * wave_length,
                        weights=y_v4o1, bins=bins)
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_grid_bin_edges[:-1]
    y = v4d_grid_hist
    y = numpy.array(y, dtype='f8')
    ax.bar(x, y, width=numpy.diff(v4d_grid_bin_edges), color='k', fill=True,
           alpha=0.5, lw=0.0, edgecolor='k', label='v4d')
    x = v4o1_grid_bin_edges[:-1]
    y = v4o1_grid_hist
    y = numpy.array(y, dtype='f8')
    ax.bar(x, y, width=numpy.diff(v4o1_grid_bin_edges), color='r', fill=True,
           alpha=0.5, lw=0.0, edgecolor='r', label='v4o1')  # hatch='//'
    ax.set_xlim(0, uv_dist_max)
    ax.set_xlabel('Baseline length (metres)')
    ax.set_ylabel('Baseline count')
    ax.legend(fontsize='small')
    pyplot.savefig(join(out_dir, 'hist_uv_grid_convolved.png'))
    pyplot.close(fig)


def main():
    # Settings ----------------------------------------------------------------
    freq_hz = 120.0e6
    wave_length = 299792458.0 / freq_hz
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

    fov = 45.0  # deg
    im_size = 512
    uv_pb_cut_level = 1.0e-5
    out_dir = 'TEST_%05.1f_%03i_v2' % (fov, im_size)
    font_size_ = 'small'

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # === Load telescope coordinates. ========================================
    v4d_file = join('v4d.tm', 'layout_enu_stations.txt')
    v4d = numpy.loadtxt(v4d_file)
    v4o1_file = join('v4o1.tm', 'layout_enu_stations.txt')
    v4o1 = numpy.loadtxt(v4o1_file)
    station_radius_m = 35.0 / 2.0
    num_stations = v4d.shape[0]
    num_baselines = num_stations * (num_stations - 1) / 2
    print('- Loaded telescope model.')

    # === Create imager object and obtain uv cell size. =======================
    imager = Imager('Double')
    imager.set_fov(fov)
    imager.set_size(im_size)
    imager.set_grid_kernel('Pillbox', 1, 1)
    cell_size_uv_wavelengths = imager.fov_to_uv_cellsize(radians(fov), im_size)
    uv_cut_radius_wavelengths = ((im_size / 2) - 2) * cell_size_uv_wavelengths
    c = im_size / 2
    dx = cell_size_uv_wavelengths
    extent_wavelengths = [(c + 0.5) * dx, (-c + 0.5) * dx,
                          (-c - 0.5) * dx, (c - 0.5) * dx]
    extent_m = numpy.array(extent_wavelengths) * wave_length

    x = numpy.arange(-im_size / 2, im_size / 2) * cell_size_uv_wavelengths
    xg, yg = numpy.meshgrid(x[::-1], x[::-1])
    rg = (xg ** 2 + yg ** 2) ** 0.5
    grid_r_wavelengths = rg.flatten()
    sort_idx = numpy.argsort(grid_r_wavelengths)
    grid_r_wavelengths = grid_r_wavelengths[sort_idx]

    lm_max = math.sin(0.5 * radians(fov))
    lm_inc = (2.0 * lm_max) / im_size
    off = lm_inc / 2.0
    extent_lm = [-lm_max - off, lm_max - off, -lm_max + off, lm_max + off]

    uv_dist_max = (im_size / 2) * cell_size_uv_wavelengths * wave_length
    print('- wavelength [m]:', wave_length)
    print('- uv_pixel size [m]:', cell_size_uv_wavelengths* wave_length)
    print('- max uv [m]:', uv_dist_max)

    # === Generate UV coordinates =============================================
    uu_v4d, vv_v4d, ww_v4d = \
        generate_uv_coordinates(v4d[:, 0], v4d[:, 1], v4d[:, 2], lon, lat, alt,
                                ra, dec, num_times, num_baselines, mjd_start,
                                dt_s, freq_hz, uv_cut_radius_wavelengths)
    print('- No. coordinates v4d  : %i' % uu_v4d.shape[0])
    uu_v4o1, vv_v4o1, ww_v4o1 = \
        generate_uv_coordinates(v4o1[:, 0], v4o1[:, 1], v4o1[:, 2], lon, lat,
                                alt, ra, dec, num_times, num_baselines,
                                mjd_start, dt_s, freq_hz,
                                uv_cut_radius_wavelengths)
    print('- No. coordinates v4o1 : %i' % uu_v4o1.shape[0])

    # UV scatter plot
    plot_uv_scatter(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length, extent_m,
                    out_dir)

    # Create UV grid images
    grid_v4d, grid_v4o1 = grid_uv_data(uu_v4d, vv_v4d, ww_v4d, uu_v4o1,
                                       vv_v4o1, ww_v4o1, im_size, imager)

    # Load PB and FFT to UV plane
    t0 = time.time()
    prefix = 'b_TIME_AVG_CHAN_AVG_CROSS_POWER'
    beam_dir = 'beams_%05.1f_%04i_%05.1fMHz' % (fov, im_size, freq_hz/1.0e6)
    beam_amp = pyfits.getdata(join(beam_dir, prefix + '_AMP_I_I.fits'))
    beam_amp = numpy.squeeze(beam_amp)
    beam_amp[numpy.isnan(beam_amp)] = 0.0
    beam = numpy.array(beam_amp, dtype='c16')
    uv_beam = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(beam)))
    uv_beam /= numpy.max(uv_beam)
    print('- Generated UV beam response in %.2f s' % (time.time() - t0))
    t0 = time.time()
    y_cut = numpy.abs(uv_beam[:, uv_beam.shape[0] / 2])
    x_cut = numpy.abs(uv_beam[uv_beam.shape[1] / 2, :])
    z_y = numpy.argmax(y_cut > uv_pb_cut_level)
    z_x = numpy.argmax(x_cut > uv_pb_cut_level)
    cut_idx = min(z_y, z_x)
    centre = uv_beam.shape[0] / 2
    offset = uv_beam.shape[0] / 2 - cut_idx
    pb_kernel = numpy.abs(uv_beam[centre - offset: centre + offset,
                          centre - offset: centre + offset])
    if pb_kernel.shape[0] % 2 == 0:
        pb_kernel = pb_kernel[1:, 1:]
    pb_kernel /= numpy.sum(pb_kernel)
    pb_kernel = numpy.abs(pb_kernel)
    print('- Kernel size =', pb_kernel.shape)
    print('- Kernel sum =', numpy.sum(pb_kernel))

    # Plot image beam
    beam_amp_norm = beam_amp / numpy.nanmax(beam_amp)
    plot_image_log(beam_amp_norm, extent_lm, join(out_dir, 'beam_amp'))
    plot_image_lin(beam_amp_norm, extent_lm, join(out_dir, 'beam_amp'))

    # Plot uv beam kernel
    plot_uv_beam_kernel(uv_beam, y_cut, x_cut, pb_kernel,
                        cell_size_uv_wavelengths, wave_length,
                        station_radius_m, uv_dist_max, uv_pb_cut_level,
                        dx, join(out_dir, 'uv_beam'))

    # Convolve UV plane PB with UV image
    t0 = time.time()
    grid_v4d_pb = scipy.signal.convolve2d(grid_v4d, pb_kernel, mode='same')
    grid_v4o1_pb = scipy.signal.convolve2d(grid_v4o1, pb_kernel, mode='same')
    print('- Convolved with FT(PB) in %.2f s' % (time.time() - t0))

    plot_uv_grid(grid_v4d, grid_v4o1, extent_m, station_radius_m,
                 join(out_dir, 'uv_grid'))

    plot_convolved_grid(grid_v4d_pb, grid_v4o1_pb, extent_m,
                        station_radius_m, join(out_dir, 'uv_grid_convolved'))

    return
    # Sum of convolved and unconvolved should be the same ...
    # Wont be though unless convolve mode == Full due to uv points near the
    # edge of the grid.
    print('- sum unconvolved grid v4d:', numpy.sum(grid_v4d))
    print('- sum unconvolved grid v4o1:', numpy.sum(grid_v4o1))
    print('- sum convolved grid v4d :', numpy.sum(grid_v4d_pb))
    print('- sum convolved grid v4o1:', numpy.sum(grid_v4o1_pb))

    # Azimuthal average plot of the UV coverage and / or histogram
    y_v4d = grid_v4d_pb.flatten()
    y_v4d = y_v4d[sort_idx]
    y_v4o1 = grid_v4o1_pb.flatten()
    y_v4o1 = y_v4o1[sort_idx]
    plot_az_uv_grid_scatter(grid_r_wavelengths, y_v4d, y_v4o1,
                            cell_size_uv_wavelengths,
                            wave_length, im_size, out_dir)

    # Bin
    bin_width = wave_length * 1.0
    # num_bins = min(100, int(ceil(uv_dist_max / bin_width)))
    num_bins = int(ceil(uv_dist_max / bin_width))
    print('- num bins:', num_bins)
    print('- bin width:', uv_dist_max/float(num_bins))

    plot_az_uv_grid_bin_mean_std(grid_r_wavelengths, y_v4d, y_v4o1,
                                 num_bins, wave_length,
                                 cell_size_uv_wavelengths, im_size,
                                 join(out_dir, 'uv_grid_az_bins'))

    # TODO-BM histogram (grid_r x grid_value)
    bins = numpy.arange(num_bins) * bin_width

    plot_convolved_hist(y_v4d, y_v4o1, grid_r_wavelengths, wave_length, bins,
                        uv_dist_max, out_dir)

    plot_unconvolved_hist(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length,
                          uv_dist_max, bins, out_dir)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + numpy.diff(v4d_bin_edges) / 2.0
    y = v4d_hist
    y = numpy.array(y, dtype='f8')
    y /= float(v4d_hist.max())
    ax.plot(x, y, 'k-', lw=1.5, label='v4d')
    x = v4o1_bin_edges[:-1] + numpy.diff(v4o1_bin_edges) / 2.0
    y = v4o1_hist
    y = numpy.array(y, dtype='f8')
    y /= float(v4d_hist.max())
    ax.plot(x, y, 'r-', lw=1.5, label='v4o1')
    ax.set_xlim(0, uv_dist_max)
    ax.set_xlabel('UV distance (metres)')
    ax.set_ylabel('Number')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'hist_%04.1fm_%06.1fm_v2.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)


if __name__ == '__main__':
    main()
