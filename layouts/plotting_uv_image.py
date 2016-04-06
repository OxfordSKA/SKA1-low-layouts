# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as pyplot
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join
import numpy
import os
from math import ceil
import time
import math


def plot_uv_images(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length,
                   station_radius_m, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    t0 = time.time()
    plot_uv_image(uu_v4d, vv_v4d, wave_length, 100.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v4o1, vv_v4o1, wave_length, 100.0, station_radius_m,
                  join(out_dir, 'uv_image_v4o1'))
    plot_uv_image(uu_v4d, vv_v4d, wave_length, 300.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v4o1, vv_v4o1, wave_length, 300.0, station_radius_m,
                  join(out_dir, 'uv_image_v4o1'))
    plot_uv_image(uu_v4d, vv_v4d, wave_length, 1000.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v4o1, vv_v4o1, wave_length, 1000.0, station_radius_m,
                  join(out_dir, 'uv_image_v4o1'))
    plot_uv_image(uu_v4d, vv_v4d, wave_length * 3.0, 1000.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v4o1, vv_v4o1, wave_length * 3.0, 1000.0, station_radius_m,
                  join(out_dir, 'uv_image_v4o1'))
    plot_uv_image(uu_v4d, vv_v4d, wave_length * 10.0, 3000.0, station_radius_m,
                  join(out_dir, 'uv_image_v4d'))
    plot_uv_image(uu_v4o1, vv_v4o1, wave_length * 10.0, 3000.0, station_radius_m,
                  join(out_dir, 'uv_image_v4o1'))
    print('- uv images took %.2f s' % (time.time() - t0))


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

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(image, interpolation='nearest', extent=extent,
                   cmap='gray_r', norm=LogNorm(vmin=1.0, vmax=image.max()))
    t = numpy.logspace(numpy.log10(1.0), numpy.log10(image.max()), 7)
    # t = numpy.linspace(1.0, image.max(), 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.ax.tick_params(labelsize='small')
    cbar.set_label('uv count', fontsize='small')
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
