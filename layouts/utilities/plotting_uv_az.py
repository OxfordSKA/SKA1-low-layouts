# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as pyplot
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join
import os
import numpy
from math import ceil
import math


def plot_az_rms(uu_v4d, vv_v4d, wave_length, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
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


def plot_az_rms_2(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    v4d_r = (uu_v4d**2 + vv_v4d**2)**0.5
    v4d_theta = numpy.arctan2(vv_v4d, uu_v4d)
    v4d_r_idx = numpy.argsort(v4d_r)
    v4d_r = v4d_r[v4d_r_idx]
    v4d_theta = v4d_theta[v4d_r_idx]

    v4o1_r = (uu_v4o1**2 + vv_v4o1**2)**0.5
    v4o1_theta = numpy.arctan2(vv_v4o1, uu_v4o1)
    v4o1_r_idx = numpy.argsort(v4o1_r)
    v4o1_r = v4o1_r[v4o1_r_idx]
    v4o1_theta = v4o1_theta[v4o1_r_idx]

    num_bins_r = 200
    uv_dist_max = 70.0e3
    uv_dist_min = 20.0
    r_bins = numpy.logspace(numpy.log10(uv_dist_min),
                            numpy.log10(uv_dist_max), num_bins_r)
    num_bins_theta = 100
    theta_bins = numpy.linspace(-math.pi, math.pi, num_bins_theta)

    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.semilogx(v4d_r, v4d_theta, 'k.', alpha=0.01, ms=2.0)
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
    ax.semilogx(v4o1_r, v4o1_theta, 'k.', alpha=0.01, ms=2.0)
    image_v4o1 = numpy.zeros((num_bins_theta-1, num_bins_r-1))
    for i in range(num_bins_r-1):
        rb_idx = numpy.where(numpy.logical_and(
            numpy.greater(v4o1_r, r_bins[i]),
            numpy.less_equal(v4o1_r, r_bins[i+1])))[0]
        rb_theta = v4o1_theta[rb_idx]
        for j in range(num_bins_theta-1):
            tb_idx = numpy.where(numpy.logical_and(
                numpy.greater(rb_theta, theta_bins[j]),
                numpy.less_equal(rb_theta, theta_bins[j+1])))[0]
            image_v4o1[j, i] = tb_idx.shape[0]
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('Azimuth [radians]')
    pyplot.savefig(join(out_dir, 'scatter_v4o1_r_theta.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(numpy.flipud(image_v4d), interpolation='nearest',
                   cmap='gray_r', norm=LogNorm(vmin=1.0, vmax=image_v4d.max()))
    i_xlabel = [0, 50, 100, 150, 199]
    ax.set_xticks(i_xlabel)
    xlabels = ['%.2f' % r_bins[ix] for ix in i_xlabel]
    ax.set_xticklabels(xlabels)
    ax.set_yticks(numpy.linspace(0, num_bins_theta-1, 7))
    ylabels = ['%.2f' % l for l in numpy.linspace(-math.pi, math.pi, 7)]
    ax.set_yticklabels(ylabels)
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
    im = ax.imshow(image_v4o1, interpolation='nearest',
                   cmap='gray_r', norm=LogNorm(vmin=1.0, vmax=image_v4o1.max()),
                   origin='lower')
    i_xlabel = [0, 50, 100, 150, 199]
    ax.set_xticks(i_xlabel)
    xlabels = ['%.2f' % r_bins[ix] for ix in i_xlabel]
    ax.set_xticklabels(xlabels)
    ax.set_yticks(numpy.linspace(0, num_bins_theta-1, 7))
    ylabels = ['%.2f' % l for l in numpy.linspace(-math.pi, math.pi, 7)]
    ax.set_yticklabels(ylabels)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    cbar.set_label('uv count', fontsize='small')
    ax.set_xlabel('Radius [m]', fontsize='small')
    ax.set_ylabel('Azimuth [radians]', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('Azimuth [radians]')
    pyplot.savefig(join(out_dir, 'scatter_v4o1_r_theta_image.png'))
    pyplot.close(fig)

    az_rms_v4d = numpy.std(image_v4d, axis=0)
    az_rms_v4o1 = numpy.std(image_v4o1, axis=0)
    rx = r_bins[:-1] + numpy.diff(r_bins) / 2.0
    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.loglog(rx, az_rms_v4d, 'k-', label='v4d')
    ax.loglog(rx, az_rms_v4o1, 'r-', label='v4o1')
    ax.plot([35, 35], ax.get_ylim(), '--')
    ax.legend()
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('uv count RMS')
    pyplot.savefig(join(out_dir, 'theta_rms.png'))
    pyplot.close(fig)

    az_sum_v4d = numpy.sum(image_v4d, axis=0)
    az_sum_v4o1 = numpy.sum(image_v4o1, axis=0)
    fig = pyplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.loglog(rx, az_sum_v4d, 'k-', label='v4d')
    ax.loglog(rx, az_sum_v4o1, 'r-', label='v4o1')
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
    idx_non_zero_v4o1 = numpy.where(az_sum_v4o1 > 0.0)
    az_rms_v4d = az_rms_v4d[idx_non_zero_v4d]
    az_rms_v4o1 = az_rms_v4o1[idx_non_zero_v4o1]
    az_sum_v4d = az_sum_v4d[idx_non_zero_v4d]
    az_sum_v4o1 = az_sum_v4o1[idx_non_zero_v4o1]
    ax.loglog(rx[idx_non_zero_v4d], numpy.divide(az_rms_v4d, az_sum_v4d),
              'k-', label='v4d')
    ax.loglog(rx[idx_non_zero_v4o1], numpy.divide(az_rms_v4o1, az_sum_v4o1),
              'r-', label='v4o1')
    # ax.plot([35, 35], ax.get_ylim(), 'b:')
    ax.legend(loc=4)
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('uv count rms / uv count sum')
    pyplot.savefig(join(out_dir, 'theta_norm_rms.png'))
    pyplot.close(fig)


