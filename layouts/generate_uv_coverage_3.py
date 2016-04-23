# -*- coding: utf-8 -*-
from __future__ import print_function
from os.path import join
import numpy
from math import radians
import math
import shutil
import matplotlib.pyplot as pyplot
import pyfits
from matplotlib.colors import SymLogNorm
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
from plotting_uv_image import plot_uv_image


def image_coords(im_size, lm_inc):
    x = numpy.arange(-im_size / 2, im_size / 2) * lm_inc
    x, y = numpy.meshgrid(x, x[::-1])
    r = (x**2 + y**2)**0.5
    return x, y, r


def fov_to_cell_size(fov, im_size):
    """Evaluate image pixel size (in arcseconds) for a given FoV and number of pixels."""
    r_max = math.sin(math.radians(fov) / 2.0)
    inc = r_max / (0.5 * im_size)
    return math.degrees(math.asin(inc)) * 3600.0


def grid_cell_size(cell_size_lm_arcsec, im_size):
    """Obtain grid cell size from image cell size."""
    return (180. * 3600.) / (im_size * cell_size_lm_arcsec * math.pi)


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


def nanargmax(a):
    idx = numpy.argmax(a, axis=None)
    multi_idx = numpy.unravel_index(idx, a.shape)
    if numpy.isnan(a[multi_idx]):
        nan_count = numpy.sum(numpy.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = numpy.argpartition(a, -nan_count - 1, axis=None)[-nan_count - 1]
        multi_idx = numpy.unravel_index(idx, a.shape)
    return multi_idx


def main():
    # Load station positions
    t0 = time.time()
    v4d_file = join('v4d.tm', 'layout_enu_stations.txt')
    v4o1_file = join('v4o1.tm', 'layout_enu_stations.txt')
    v4d = numpy.loadtxt(v4d_file)
    v4o1 = numpy.loadtxt(v4o1_file)
    station_radius_m = 35.0 / 2.0
    num_stations = v4d.shape[0]
    assert(v4o1.shape[0] == v4d.shape[0])
    print('- loading coordinates took %.2f s' % (time.time() - t0))

    freq = 120.0e6
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

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # UV coordinate generation ===============================================
    t0 = time.time()
    x, y, z = convert_enu_to_ecef(v4d[:, 0], v4d[:, 1], v4d[:, 2],
                                  lon, lat, alt)
    uu_v4d, vv_v4d, ww_v4d = \
        generate_baseline_uvw(x, y, z, ra, dec, num_times, num_baselines,
                              mjd_start, dt_s)
    print('- Coordinate generation took %.2f s' % (time.time() - t0))
    print('- Num vis = %i' % uu_v4d.shape[0])

    fov = 180.0  # deg
    im_size = 8192
    n = im_size
    c = n / 2
    z = 128

    cell_size_lm_arcsec = fov_to_cell_size(fov, im_size)
    cell_size_uv = grid_cell_size(cell_size_lm_arcsec, im_size)
    uv_max = ((im_size / 2) - 5) * cell_size_uv

    uv_r_cut = uv_max * wave_length
    uv_r = (uu_v4d**2 + vv_v4d**2)**0.5
    uv_sort_idx = numpy.argsort(uv_r)
    uv_r = uv_r[uv_sort_idx]
    uu_v4d = uu_v4d[uv_sort_idx]
    vv_v4d = vv_v4d[uv_sort_idx]
    ww_v4d = ww_v4d[uv_sort_idx]

    i_uv_max = numpy.argmax(uv_r >= uv_r_cut)
    uu_v4d = uu_v4d[:i_uv_max]
    vv_v4d = vv_v4d[:i_uv_max]
    ww_v4d = ww_v4d[:i_uv_max]
    print('- No. uv points after radial cut = %i' % uu_v4d.shape[0])

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4d, vv_v4d, '.', ms=2.0, alpha=0.2)
    ax.set_xlim(-uv_max * wave_length, uv_max * wave_length)
    ax.set_ylim(-uv_max * wave_length, uv_max * wave_length)
    fig.savefig(join(out_dir, 'uv_scatter.png'))
    pyplot.close(fig)

    psf_v4d = make_psf(uu_v4d, vv_v4d, ww_v4d, freq, fov, im_size)
    lm_max = psf_v4d['lm_max']
    lm_inc = psf_v4d['lm_inc']
    off = lm_inc / 2.0
    extent = [-lm_max - off, lm_max - off, -lm_max - off, lm_max - off]
    plot_image_log(psf_v4d['image'], extent, fov, join(out_dir, 'psf'))

    print('- Loading beam image.')
    prefix = 'b_TIME_AVG_CHAN_AVG_CROSS_POWER'
    beam_amp = pyfits.getdata(join('beams_180.0_120.0MHz',
                                   prefix + '_AMP_I_I.fits'))
    beam_phase = pyfits.getdata(join('beams_180.0_120.0MHz',
                                     prefix + '_PHASE_I_I.fits'))
    beam_amp = numpy.squeeze(beam_amp)
    beam_phase = numpy.squeeze(beam_phase)
    beam_amp[numpy.isnan(beam_amp)] = 0.0
    beam_phase[numpy.isnan(beam_phase)] = 0.0
    beam = beam_amp * numpy.exp(1.0j * beam_phase)
    # beam /= numpy.sum(beam)
    print('- beam sum = %f %f' % (numpy.sum(beam.real), numpy.sum(beam_amp)))

    beam_amp = beam_amp / numpy.nanmax(beam_amp)
    plot_image_lin(beam_amp, extent, fov, join(out_dir, 'beam_amp'))
    plot_image_lin(beam_phase, extent, fov, join(out_dir, 'beam_phase'))
    plot_image_lin(beam_amp[c - z:c + z, c - z:c + z],
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5],
                   0, join(out_dir, 'beam_amp_zoom'))
    plot_image_lin(numpy.real(beam[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'beam_re_zoom'))
    plot_image_lin(numpy.imag(beam[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'beam_im_zoom'))

    # Beam uv plane response
    uv_beam = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(beam)))
    plot_image_lin(numpy.real(uv_beam), extent, 0, join(out_dir, 'uv_beam_re'))
    plot_image_lin(numpy.imag(uv_beam), extent, 0, join(out_dir, 'uv_beam_im'))
    plot_image_lin(numpy.real(uv_beam[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'uv_beam_re_zoom'))
    plot_image_lin(numpy.imag(uv_beam[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'uv_beam_im_zoom'))

    print('- Beam x PSF.')
    beam_psf = beam * psf_v4d['image']
    plot_image_log(numpy.real(beam_psf), extent, fov,
                   join(out_dir, 'psf_beam_re'))
    plot_image_log(numpy.imag(beam_psf), extent, fov,
                   join(out_dir, 'psf_beam_im'))

    print('- UV image (beam convolved).')
    uv_psf_beam = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(beam_psf)))
    plot_image_lin(numpy.real(uv_psf_beam), [-1, 1, -1, 1], 0,
                   join(out_dir, 'uv_psf_beam_re'))
    plot_image_lin(numpy.imag(uv_psf_beam), [-1, 1, -1, 1], 0,
                   join(out_dir, 'uv_psf_beam_im'))

    print('- UV image (PSF only).')
    uv_psf = numpy.fft.fftshift(
        numpy.fft.ifft2(numpy.fft.fftshift(psf_v4d['image'])))
    print('- uv_psf grid sum = %f %f' % (numpy.sum(uv_psf.real),
                                         numpy.sum(uv_psf.real)))
    print('- uv_psf_beam grid sum = %f %f' % (numpy.sum(uv_psf_beam.real),
                                              numpy.sum(uv_psf_beam.real)))
    uv_psf /= numpy.sum(uv_psf)
    uv_psf_beam /= numpy.sum(uv_psf_beam)
    uv_psf *= uu_v4d.shape[0]
    uv_psf_beam *= uu_v4d.shape[0]

    plot_image_lin(numpy.real(uv_psf), [-1, 1, -1, 1], 0,
                   join(out_dir, 'uv_psf_re'))
    plot_image_lin(numpy.imag(uv_psf), [-1, 1, -1, 1], 0,
                   join(out_dir, 'uv_psf_im'))
    plot_image_lin(numpy.real(uv_psf[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'uv_psf_re_zoom'))
    plot_image_lin(numpy.imag(uv_psf[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'uv_psf_im_zoom'))
    plot_image_lin(numpy.real(uv_psf_beam[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'uv_psf_beam_re_zoom'))
    plot_image_lin(numpy.imag(uv_psf_beam[c - z: c + z, c - z: c + z]),
                   [-z - 0.5, z - 0.5, -z + 0.5, z + 0.5], 0,
                   join(out_dir, 'uv_psf_beam_im_zoom'))

    # # Grid uv data to compare with FT(psf) wrt normalisation
    # uv_grid = numpy.zeros((im_size, im_size))
    # for i in range(uu_v4d.shape[0]):
    #     gx = round(uu_v4d[i] / cell_size_uv) + (im_size / 2)
    #     gy = round(vv_v4d[i] / cell_size_uv) + (im_size / 2)
    #     uv_grid[gy, gx] += 1

    # print('- uv_grid sum = %f' % numpy.sum(uv_grid))
    print('- uv_psf sum = %f' % numpy.sum(uv_psf.real))
    plot_uv_image(uu_v4d, vv_v4d, cell_size_uv, 250 * cell_size_uv,
                  station_radius_m, join(out_dir, 'uv_image'))

    print('- Producing Azimuthal averaged plots')
    # Azimuthal average and plot vs radius
    _, _, r = image_coords(im_size, lm_inc)
    x = r.flatten()
    sort_idx = numpy.argsort(x)
    x = x[sort_idx]
    y1 = numpy.real(uv_psf).flatten()
    y1 = y1[sort_idx]
    y2 = numpy.real(uv_psf_beam).flatten()
    y2 = y2[sort_idx]

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, y1, 'b.', ms=3.0, alpha=0.1, label='uv')
    ax.plot(x, y2, 'r.', ms=3.0, alpha=0.1, label='uv convolved with PB')
    ax.legend()
    ax.set_xlim(0, 0.1)
    fig.savefig(join(out_dir, 'az_uv_r.png'))
    pyplot.close(fig)


if __name__ == '__main__':
    main()
