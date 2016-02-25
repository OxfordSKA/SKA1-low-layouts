# -*- coding: utf-8 -*-
from __future__ import print_function
import pyfits
import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from os.path import join


def main():
    model_dir = 'models'
    telescope_models = [d for d in os.listdir(model_dir)
                        if os.path.isdir(join(model_dir, d))
                        and d.endswith('.tm')]
    beam_root_dir = 'horizon_zenith_beams'
    freqs = [50.0e6, 350.0e6]

    beam_dirs = [d for d in os.listdir(beam_root_dir)
                 if os.path.isdir(join(beam_root_dir, d))
                 and d.startswith('beams_')]


    i = 0

    layout = numpy.loadtxt(join(model_dir, telescope_models[i], 'layout.txt'))
    num_stations = layout.shape[0]
    st = numpy.loadtxt(join(model_dir, telescope_models[i], 'station000',
                            'layout.txt'))
    num_antennas = st.shape[0]
    norm = num_antennas**2
    print(telescope_models[i])
    print(num_antennas, num_stations)

    beam_file = join(beam_root_dir, beam_dirs[i],
                     'b_TIME_AVG_CHAN_AVG_CROSS_POWER_AMP_I_I.fits')
    data = numpy.squeeze(pyfits.getdata(beam_file))
    print(numpy.nanmax(data))

    im_size = data.shape[0]
    im_size = 10
    lm_inc = 2.0 / im_size  # Not 100% sure this is right... CHECK
    x = numpy.arange(im_size) * lm_inc - 1.0
    x, y = numpy.meshgrid(x, x[::-1])
    r = (x**2 + y**2)**0.5
    test_data = numpy.copy(r)
    extent = [-1.0 - lm_inc / 2.0, 1.0 - lm_inc / 2.0,
              -1.0 - lm_inc / 2.0, 1.0 - lm_inc / 2.0]

    test_data[r < 0.5] = numpy.nan

    fig = pyplot.figure(figsize=(14, 7))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.92,
                        hspace=0.0, wspace=0.2)
    ax1 = fig.add_subplot(111)
    circle = pyplot.Circle((0.0, 0.0), 0.5,
                           color='k', linestyle='--',
                           fill=False, alpha=1.0, lw=1.0)
    ax1.add_artist(circle)
    im = ax1.imshow(test_data, interpolation='nearest', cmap='inferno',
                    extent=extent)
    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax1.figure.colorbar(im, cax=cax2)
    cbar.set_label('Power', fontsize='small')
    cbar.ax.tick_params(labelsize='small')
    ax1.grid()
    pyplot.show()


    return
    # ============================
    fig = pyplot.figure(figsize=(14, 7))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.92,
                        hspace=0.0, wspace=0.2)

    ax1 = fig.add_subplot(121)
    im = ax1.imshow(data, interpolation='nearest', cmap='inferno')
    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax1.figure.colorbar(im, cax=cax2)
    cbar.set_label('Power', fontsize='small')
    cbar.ax.tick_params(labelsize='small')
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax1.set_xlabel('East <-> West', fontsize='small')
    ax1.set_ylabel('North <-> South', fontsize='small')
    ax1.grid()
    ax1.set_title('Average cross-power beam', fontsize='small')

    ax2 = fig.add_subplot(122)

    data_max = numpy.nanmax(data)
    norm_data = (data / data_max)
    data_db = 10.0 * numpy.log10(norm_data)
    im = ax2.imshow(data_db, interpolation='nearest', cmap='inferno',
                    vmin=-40, vmax=0.0)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax2.figure.colorbar(im, cax=cax2)
    cbar.set_label('Decibels', fontsize='small')
    cbar.ax.tick_params(labelsize='small')
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax2.set_xlabel('East <-> West', fontsize='small')
    ax2.set_ylabel('North <-> South', fontsize='small')
    ax2.grid()
    ax2.set_title('Average cross-power beam', fontsize='small')
    pyplot.show()
    # ============================

if __name__ == '__main__':
    main()
