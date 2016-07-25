# -*- coding: utf-8 -*-

import math

import numpy
import oskar
import pyuvwsim


def uv_cell_size_to_fov(cell_size_uv_wavelengths, size):
    cell_size_lm = 1.0 / (size * cell_size_uv_wavelengths)
    lm_max = (size * math.sin(cell_size_lm)) / 2.0
    theta_max_rad = math.asin(lm_max)
    return math.degrees(theta_max_rad) * 2.0


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


def generate_baseline_coords(layout, settings):
    x, y = layout['x'], layout['y']
    z = numpy.zeros_like(x)
    wavelength_m = 299792458.0 / settings['freq_hz']
    mjd_start = settings['mjd_mid'] - (settings['obs_length_s'] / 2.0) / 86400.0
    x, y, z = numpy.array(x), numpy.array(y), numpy.zeros_like(x)
    x_ecef, y_ecef, z_ecef = pyuvwsim.convert_enu_to_ecef(x, y, z,
                                                          settings['lon'],
                                                          settings['lat'])
    num_stations = x_ecef.shape[0]
    num_baselines = num_stations * (num_stations - 1) / 2
    num_coords = settings['num_times'] * num_baselines
    uu = numpy.zeros(num_coords, dtype='f4')
    vv, ww = numpy.zeros_like(uu), numpy.zeros_like(uu)
    for i in range(settings['num_times']):
        mjd = mjd_start + (i * settings['interval_s'] +
                           settings['interval_s'] / 2.0) / 86400.0
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


def generate_uv_grid_2(uu, vv, ww, r_max, settings):
    uv_cell_size_m = (settings['station_r'] * 2.0) / 3.0
    wavelength_m = 299792458.0 / settings['freq_hz']
    uv_max = r_max * 2.0
    grid_size = int((uv_max * 2.0) / uv_cell_size_m)
    if grid_size % 2 == 1:
        grid_size += 1
    uv_cell_size_wavelengths = uv_cell_size_m / wavelength_m
    fov_deg = uv_cell_size_to_fov(uv_cell_size_wavelengths, grid_size)
    im = oskar.imager.Imager('Single')
    im.set_fft_on_gpu(False)
    im.set_fov(fov_deg)
    im.set_size(grid_size)
    im.set_grid_kernel('Pillbox', 1, 1)
    uv_grid = numpy.zeros((grid_size, grid_size), dtype='c8')
    im.update_plane(uu, vv, ww, numpy.ones_like(uu, dtype='c8'),
                    numpy.ones_like(uu), uv_grid, 0.0)
    im.update_plane(-uu, -vv, -ww, numpy.ones_like(uu, dtype='c8'),
                    numpy.ones_like(uu), uv_grid, 0.0)
    return uv_grid, uv_cell_size_wavelengths, fov_deg


def generate_psf_2(uu, vv, ww, settings):
    im = oskar.imager.Imager('Single')
    im.set_fft_on_gpu(False)
    psf = im.make_image(uu, vv, ww, numpy.ones_like(uu, dtype='c8'),
                        numpy.ones_like(uu, dtype='f8'), settings['psf_fov_deg'],
                        settings['psf_im_size'])
    return psf


def generate_psf_3(uu, vv, ww, im_size, fov_deg):
    im = oskar.imager.Imager('Single')
    im.set_fft_on_gpu(False)
    psf = im.make_image(uu, vv, ww, numpy.ones_like(uu, dtype='c8'),
                        numpy.ones_like(uu), fov_deg, im_size)
    return psf


def get_psf_coords(settings):
    lm_max = math.sin(0.5 * math.radians(settings['psf_fov_deg']))
    lm_inc = 2.0 * lm_max / settings['psf_im_size']
    l = numpy.arange(-settings['psf_im_size'] / 2, settings['psf_im_size'] / 2)
    l = l.astype(dtype='f8') * lm_inc
    l, m = numpy.meshgrid(-l, l)
    r = (l**2 + m**2)**0.5
    return l, m, r


def generate_psf(uu, vv, ww, settings):
    im_size = settings['psf_im_size']
    fov_deg = settings['psf_fov_deg']
    im = oskar.imager.Imager('Single')
    im.set_grid_kernel('Spheroidal', 3, 100)
    im.set_fft_on_gpu(False)
    psf = im.make_image(uu, vv, ww, numpy.ones_like(uu, dtype='c16'),
                        numpy.ones_like(uu), settings['psf_fov_deg'],
                        settings['psf_im_size'])

    psf_cell_size = im.fov_to_cellsize(math.radians(fov_deg), im_size)
    l = numpy.arange(-im_size / 2, im_size / 2) * psf_cell_size
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
