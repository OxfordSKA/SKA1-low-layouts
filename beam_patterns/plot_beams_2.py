# -*- coding: utf-8 -*-
from __future__ import print_function
import pyfits
import numpy
from os.path import join
import os
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
import time
import aplpy


def plot(file_name, cmin, cmax, clabel, stretch, out_name):
    gc = aplpy.FITSFigure(file_name, figsize=(9, 7))
    gc.show_colorscale(vmin=cmin, vmax=cmax, cmap='seismic', aspect='equal',
                       interpolation='nearest', stretch=stretch)
    # gc.show_colorscale(vmax=cmax, cmap='seismic', aspect='equal',
    #                    interpolation='nearest', stretch=stretch)
    gc.add_colorbar()
    # gc.colorbar.set_axis_label_text(r'Flux (Jy/beam)')
    gc.colorbar.set_axis_label_text(clabel)
    gc.colorbar.set_axis_label_font(size='small', weight='bold')
    gc.colorbar.set_font(size='small')
    gc.tick_labels.set_font(size='small')
    # gc.add_grid()
    # gc.grid.set_color('white')
    gc.grid.set_color('black')
    gc.grid.set_alpha(0.2)
    gc.set_title('', fontsize='medium')
    # plt.savefig(out_name)
    plt.show()


if __name__ == '__main__':
    i = 25
    # beams_dir = join('beams_v4a_random_lattice_aligned', 'e')
    beams_dir = join('beams_v4a_fixed_lattice_not_aligned', 'e')
    # file_name = join(beams_dir, 'beam_S%04i_TIME_AVG_CHAN_AVG_AUTO_'
    #                             'POWER_I_I.fits' % i)
    file_name = join(beams_dir, 'beam_TIME_AVG_CHAN_AVG_CROSS_'
                                'POWER_AMP_I_I.fits')
    data = numpy.squeeze(pyfits.getdata(file_name))
    data_db = 10.0 * numpy.log10(data / numpy.nanmax(data))

    fig = pyplot.figure(figsize=(9, 7))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=1.0, top=0.92,
                        hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(111, aspect='equal')
    # http://matplotlib.org/users/colormaps.html
    im = ax.imshow(data_db, interpolation='nearest', cmap='magma',
                   vmin=-60.0, vmax=0.0)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Decibels', fontsize='small')
    cbar.ax.tick_params(labelsize='small')
    # cbar.set_clim(-80, 0)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlabel('East <-> West', fontsize='small')
    ax.set_ylabel('North <-> South', fontsize='small')
    ax.grid()
    ax.set_title(os.path.basename(file_name), fontsize='small')
    pyplot.show()

    # c_label = r'Average stokes-I beam response.'
    # plot(file_name, 1.e-3, 1.0, c_label, 'linear', 'test_beam.png')
