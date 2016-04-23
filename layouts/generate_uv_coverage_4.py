# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy
from math import radians
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


def bin_az_ave(x, y, num_bins, im_size, cell_size_uv):
    bins = {'x': numpy.zeros(num_bins),
            'mean': numpy.zeros(num_bins),
            'std': numpy.zeros(num_bins)}
    bin_inc = (im_size / 2.0 * cell_size_uv) / float(num_bins)
    for i in range(num_bins):
        r0 = i * bin_inc
        r1 = r0 + bin_inc
        bins['x'][i] = r0 + (r1 - r0) / 2.0
        idx = numpy.where(numpy.logical_and(x > r0, x <= r1))
        y_bin = y[idx]
        bins['mean'][i] = y_bin.mean()
        bins['std'][i] = y_bin.std()
    return bins


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
    ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2e')
    ax.set_xlabel('l', fontsize=font_size_)
    ax.set_ylabel('m', fontsize=font_size_)
    ax.tick_params(axis='both', which='major', labelsize=font_size_)
    ax.tick_params(axis='both', which='minor', labelsize=font_size_)
    ax.grid(True)
    pyplot.savefig(file_name + '_lin.png')
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
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size_)
    ax.tick_params(axis='both', which='minor', labelsize=font_size_)
    pyplot.savefig(file_name + '_%03.1f_image_log.png' % fov)
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
    uv_r_cut = uv_cut_radius_wavelengths
    uv_r = (uu ** 2 + vv ** 2) ** 0.5
    uv_sort_idx = numpy.argsort(uv_r)
    uv_r = uv_r[uv_sort_idx]
    uu = uu[uv_sort_idx]
    vv = vv[uv_sort_idx]
    ww = ww[uv_sort_idx]
    i_uv_max = numpy.argmax(uv_r >= uv_r_cut)
    uu = uu[:i_uv_max]
    vv = vv[:i_uv_max]
    ww = ww[:i_uv_max]
    return uu, vv, ww


def main():
    # Settings
    # -------------------------------------------------------------------------
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

    out_dir = 'TEMP_UV'

    fov = 180.0  # deg
    im_size = 256
    uv_pb_cut_level = 1.0e-5
    # -------------------------------------------------------------------------

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Load telescope coordinates.
    v4d_file = join('v4d.tm', 'layout_enu_stations.txt')
    v4d = numpy.loadtxt(v4d_file)
    v4o1_file = join('v4o1.tm', 'layout_enu_stations.txt')
    v4o1 = numpy.loadtxt(v4o1_file)
    station_radius_m = 35.0 / 2.0
    num_stations = v4d.shape[0]
    num_baselines = num_stations * (num_stations - 1) / 2
    print('- Loaded telescope model.')

    # Create imager object and obtain uv cell size.
    imager = Imager('Double')
    imager.set_fov(fov)
    imager.set_size(im_size)
    imager.set_grid_kernel('pillbox', 1, 1)
    # imager.set_grid_kernel('s', 3, 100)
    cell_size_uv_wavelengths = imager.fov_to_uv_cellsize(radians(fov), im_size)
    uv_cut_radius_wavelengths = ((im_size / 2) - 2) * cell_size_uv_wavelengths

    c = im_size / 2
    dx = cell_size_uv_wavelengths
    extent_wavelengths = [(c + 0.5) * dx, (-c + 0.5) * dx,
                          (-c - 0.5) * dx, (c - 0.5) * dx]
    extent_m = numpy.array(extent_wavelengths) * wave_length

    # Grid pixel coordinates and pixel radii
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

    # Generate UV coordinates
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
    # uu_v4d = numpy.append(uu_v4d, 0.0)
    # vv_v4d = numpy.append(vv_v4d, 0.0)
    # ww_v4d = numpy.append(ww_v4d, 0.0)

    # Create UV grid images
    grid_v4d = numpy.zeros((im_size, im_size), dtype='c16')
    amps_v4d = numpy.ones(uu_v4d.shape, dtype='c16')
    weight_v4d = numpy.ones(uu_v4d.shape, dtype='f8')
    imager.update_plane(uu_v4d, vv_v4d, ww_v4d, amps_v4d, weight_v4d,
                        grid_v4d, 0.0)
    assert(numpy.sum(numpy.abs(grid_v4d.imag)) == 0.0)
    print(numpy.sum(grid_v4d.real), uu_v4d.shape[0])
    assert(round(numpy.sum(grid_v4d.real)) == uu_v4d.shape[0])
    grid_v4d = numpy.real(grid_v4d)

    grid_v4o1 = numpy.zeros((im_size, im_size), dtype='c16')
    amps_v4o1 = numpy.ones(uu_v4o1.shape, dtype='c16')
    weight_v4o1 = numpy.ones(uu_v4o1.shape, dtype='f8')
    imager.update_plane(uu_v4o1, vv_v4o1, ww_v4o1, amps_v4o1, weight_v4o1,
                        grid_v4o1, 0.0)
    assert (numpy.sum(numpy.abs(grid_v4o1.imag)) == 0.0)
    grid_v4o1 = numpy.real(grid_v4o1)

    # Plot uv scatter comparing the two telescopes
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4d * wave_length, vv_v4d * wave_length, 'b.', ms=2.0,
            alpha=0.2)
    ax.plot(uu_v4o1 * wave_length, vv_v4o1 * wave_length, 'r.', ms=2.0,
            alpha=0.2)
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    fig.savefig(join(out_dir, 'uv_scatter.png'))
    ax.grid(True)
    pyplot.close(fig)

    # Plot UV grid image with scatter overlay of uv coordinates
    # (wavelength axis scale)
    # v4d
    file_name = join(out_dir, 'uv_grid_v4d')
    fig = pyplot.figure(figsize=(8, 8))
    font_size_ = 'small'
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4d, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(grid_v4d.min(), grid_v4d.max(), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if extent_m[1] / station_radius_m < 20.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.plot(uu_v4d * wave_length, vv_v4d * wave_length, 'r.', ms=2.0, alpha=0.2)
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(file_name + '_lin.png')
    # pyplot.show()
    pyplot.close(fig)

    # Plot UV grid image with scatter overlay of uv coordinates
    # (wavelength axis scale)
    # v4o1
    file_name = join(out_dir, 'uv_grid_v4o1')
    fig = pyplot.figure(figsize=(8, 8))
    font_size_ = 'small'
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(grid_v4o1, interpolation='nearest',
                   extent=extent_m, cmap='gray_r', origin='lower')
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    t = numpy.linspace(grid_v4o1.min(), grid_v4o1.max(), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    if extent_m[1] / station_radius_m < 20.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.plot(uu_v4o1 * wave_length, vv_v4o1 * wave_length, 'r.', ms=2.0,
            alpha=0.2)
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(file_name + '_lin.png')
    pyplot.close(fig)

    # Load PB and FFT to UV plane
    t0 = time.time()
    prefix = 'b_TIME_AVG_CHAN_AVG_CROSS_POWER'
    beam_dir = 'beams_%05.1f_%04i_%05.1fMHz' % (fov, im_size, freq_hz/1.0e6)
    beam_amp = pyfits.getdata(join(beam_dir, prefix + '_AMP_I_I.fits'))
    beam_amp = numpy.squeeze(beam_amp)
    beam_amp[numpy.isnan(beam_amp)] = 0.0
    plot_image_log(beam_amp, extent_lm, fov, join(out_dir, 'beam_amp'))
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
    pb_kernel = numpy.abs(uv_beam[centre-offset: centre+offset,
                                  centre-offset: centre+offset])
    pb_kernel /= numpy.sum(pb_kernel)
    pb_kernel = numpy.abs(pb_kernel)
    print('- Kernel size =', pb_kernel.shape)
    print('- Kernel sum =', numpy.sum(pb_kernel))

    # Plot the kernel cut profile
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    zz = min(200, uv_beam.shape[0] / 2)
    xx = numpy.arange(-zz, zz)
    ax.semilogy(xx, y_cut[uv_beam.shape[0] / 2 - zz: uv_beam.shape[0] / 2 + zz],
                'r')
    ax.semilogy(xx, x_cut[uv_beam.shape[1] / 2 - zz: uv_beam.shape[1] / 2 + zz],
                'b')
    ax.plot([xx[0], xx[-1]], [uv_pb_cut_level, uv_pb_cut_level], 'k', lw=3.0,
            alpha=2.0)
    ax.grid(True)
    fig.savefig(join(out_dir, 'xy_cut.png'))
    pyplot.close(fig)

    # Plot the uv primary beam kernel
    c_pb = pb_kernel.shape[0] / 2
    extent_pb_wavelengths = [(c_pb + 0.5) * dx, (-c_pb + 0.5) * dx,
                             (-c_pb - 0.5) * dx, (c_pb - 0.5) * dx]
    extent_pb_m = numpy.array(extent_pb_wavelengths) * wave_length
    file_name = join(out_dir, 'uv_pb')
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
    cbar.set_label('uv points per pixel', fontsize='small')
    ax.set_xlabel('uu [m]', fontsize='small')
    ax.set_ylabel('vv [m]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlim(extent_pb_m[0], extent_pb_m[1])
    ax.set_ylim(extent_pb_m[2], extent_pb_m[3])
    ax.grid(True)
    pyplot.savefig(file_name + '_lin.png')
    # pyplot.show()
    pyplot.close(fig)

    # Convolve UV plane PB with UV image
    t0 = time.time()
    grid_v4d_pb = scipy.signal.convolve2d(grid_v4d, pb_kernel, mode='same')
    grid_v4o1_pb = scipy.signal.convolve2d(grid_v4o1, pb_kernel, mode='same')
    # Normalise
    grid_pb_max = max(numpy.max(grid_v4d_pb), numpy.max(grid_v4o1_pb))
    grid_v4d_pb /= grid_pb_max
    grid_v4o1_pb /= grid_pb_max
    print('- Convolved with FT(PB) in %.2f s' % (time.time() - t0))

    # Plot UV convolved grid image with scatter overlay of uv coordinates
    # (wavelength axis scale)
    # v4d
    file_name = join(out_dir, 'uv_grid_pb_v4d')
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
    if extent_m[1] / station_radius_m < 20.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.plot(uu_v4d * wave_length, vv_v4d * wave_length, 'r.', ms=2.0, alpha=0.2)
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(file_name + '_lin.png')
    # pyplot.show()
    pyplot.close(fig)

    # Plot UV convolved grid image with scatter overlay of uv coordinates
    # (wavelength axis scale)
    # v4o1
    file_name = join(out_dir, 'uv_grid_pb_v4o1')
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
    if extent_m[1] / station_radius_m < 20.0:
        circle = pyplot.Circle((0.0, 0.0), station_radius_m, linestyle='-',
                               color='b', fill=False, alpha=0.5,
                               linewidth=1.0)
        ax.add_artist(circle)
        ax.grid()
    ax.plot(uu_v4o1 * wave_length, vv_v4o1 * wave_length, 'r.', ms=2.0, alpha=0.2)
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.grid(True)
    pyplot.savefig(file_name + '_lin.png')
    # pyplot.show()
    pyplot.close(fig)

    # Azimuthal average plot of the UV coverage and / or histogram
    y_v4d = grid_v4d_pb.flatten()
    y_v4d = y_v4d[sort_idx]
    y_v4o1 = grid_v4o1_pb.flatten()
    y_v4o1 = y_v4o1[sort_idx]
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(grid_r_wavelengths * wave_length, y_v4d, 'k.', ms=2.0, alpha=0.2)
    ax.set_xlabel('Radius [m]')
    ax.set_xlim(0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    fig.savefig(join(out_dir, 'az_ave_scatter_v4d_grid_pb.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(grid_r_wavelengths * wave_length, y_v4o1, 'k.', ms=2.0, alpha=0.2)
    ax.set_xlabel('Radius [m]')
    ax.set_xlim(0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    fig.savefig(join(out_dir, 'az_ave_scatter_v4o1_grid_pb.png'))
    pyplot.close(fig)

    # Bin
    num_bins = 100
    bins_v4d = bin_az_ave(grid_r_wavelengths, y_v4d, num_bins, im_size,
                          cell_size_uv_wavelengths)
    bins_v4o1 = bin_az_ave(grid_r_wavelengths, y_v4o1, num_bins, im_size,
                           cell_size_uv_wavelengths)

    # Plot with linear scale
    fig = pyplot.figure(figsize=(8, 8))
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
    ax.set_xlabel('UV distance (metres)')
    ax.set_ylabel('Relative sensitivity')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'az_ave_lin_grid_pb.png'))
    pyplot.close(fig)

    # ============
    # Azimuthal average plot of the UV coverage and / or histogram
    y_v4d = grid_v4d.flatten()
    y_v4d = y_v4d[sort_idx]
    y_v4o1 = grid_v4o1.flatten()
    y_v4o1 = y_v4o1[sort_idx]
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(grid_r_wavelengths * wave_length, y_v4d, 'k.', ms=2.0, alpha=0.2)
    ax.set_xlabel('Radius [m]')
    ax.set_xlim(0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    fig.savefig(join(out_dir, 'az_ave_scatter_v4d.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(grid_r_wavelengths * wave_length, y_v4o1, 'k.', ms=2.0, alpha=0.2)
    ax.set_xlabel('Radius [m]')
    ax.set_xlim(0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    fig.savefig(join(out_dir, 'az_ave_scatter_v4o1.png'))
    pyplot.close(fig)

    # Bin
    num_bins = 100
    bins_v4d = bin_az_ave(grid_r_wavelengths, y_v4d, num_bins, im_size,
                          cell_size_uv_wavelengths)
    bins_v4o1 = bin_az_ave(grid_r_wavelengths, y_v4o1, num_bins, im_size,
                           cell_size_uv_wavelengths)

    # Plot with linear scale
    fig = pyplot.figure(figsize=(8, 8))
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
    ax.set_xlabel('UV distance (metres)')
    ax.set_ylabel('Relative sensitivity')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'az_ave_lin_grid.png'))
    pyplot.close(fig)


    # ============
    # Azimuthal average plot of the UV coverage and / or histogram
    r_v4d = (uu_v4d**2 + vv_v4d**2)**0.5
    y_v4d = numpy.ones(uu_v4d.shape)
    r_v4o1 = (uu_v4o1**2 + vv_v4o1**2)**0.5
    y_v4o1 = numpy.ones(uu_v4o1.shape)

    # Bin
    num_bins = 100
    bins = {'x': numpy.zeros(num_bins),
            'mean': numpy.zeros(num_bins),
            'std': numpy.zeros(num_bins),
            'count': numpy.zeros(num_bins)}
    bin_inc = (im_size / 2.0 * cell_size_uv_wavelengths) / float(num_bins)
    print(bin_inc)
    print(r_v4d.min(), r_v4d.max())
    for i in range(num_bins):
        r0 = i * bin_inc
        r1 = r0 + bin_inc
        bins['x'][i] = r0 + (r1 - r0) / 2.0
        idx = numpy.where(numpy.logical_and(x > r0, x <= r1))
        y_bin = y_v4d[idx]
        print(i, y_bin.shape)
        bins['mean'][i] = y_bin.mean()
        bins['std'][i] = y_bin.std()
        bins['count'][i] = y_bin.shape[0]
    # return bins
    bins_v4d = bins

    # bins_v4d = bin_az_ave(r_v4d, y_v4d, num_bins, im_size,
    #                       cell_size_uv_wavelengths)
    # bins_v4o1 = bin_az_ave(r_v4o1, y_v4o1, num_bins, im_size,
    #                        cell_size_uv_wavelengths)

    # Plot with linear scale
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    xb1 = bins_v4d['x'] * wave_length
    yb1 = bins_v4d['count']
    # yb1_std = bins_v4d['std']
    ax.plot(xb1, yb1, 'b-', label='v4d')
    # ax.fill_between(xb1, yb1 - yb1_std, yb1 + yb1_std, edgecolor='blue',
    #                 facecolor='blue', alpha=0.2)
    # xb2 = bins_v4o1['x'] * wave_length
    # yb2 = bins_v4o1['mean']
    # yb2_std = bins_v4o1['std']
    # ax.plot(xb2, yb2, 'r-', label='v4o1')
    # ax.fill_between(xb2, yb2 - yb2_std, yb2 + yb2_std, edgecolor='red',
    #                 facecolor='red', alpha=0.2)
    ax.set_xlim(0.0, (im_size / 2) * cell_size_uv_wavelengths * wave_length)
    ax.set_xlabel('UV distance (metres)')
    ax.set_ylabel('Relative sensitivity')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'az_ave_lin_ungridded.png'))
    pyplot.close(fig)


if __name__ == '__main__':
    main()
