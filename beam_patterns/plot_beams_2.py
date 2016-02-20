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


if __name__ == '__main__':
    i = 25
    beams_dir = join('beams_v4a_random_lattice_aligned', 'e')
    file_name = join(beams_dir, 'beam_S%04i_TIME_AVG_CHAN_AVG_AUTO_'
                                'POWER_I_I.fits' % i)
    data = numpy.squeeze(pyfits.getdata(file_name))
    data_db = 10.0 * numpy.log10(data / numpy.nanmax(data))

    fig = pyplot.figure(figsize=(9, 7))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=1.0, top=0.92,
                        hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(111, aspect='equal')
    im = ax.imshow(data_db, interpolation='nearest', cmap='seismic',
                   vmin=-40.0, vmax=0.0)
    # im = ax.imshow(data, interpolation='nearest', cmap='seismic')
    ax.figure.colorbar(im, ax=ax)
    pyplot.show()

    # label2 = ax.text(0.02, 0.98, '', ha='left', va='top', style='italic',
    #                 color='k', transform=ax.transAxes)
    # im = ax.imshow(numpy.random.random((512, 512)), interpolation='nearest',
    #                animated=True)
    # ax.figure.colorbar(im, ax=ax)
    # ax.grid()
    # # ax.set_title('')
    #
    # def animate(i):
    #     t0 = time.time()
    #     file_name = join(beams_dir, 'beam_S%04i_TIME_AVG_CHAN_AVG_AUTO_'
    #                                 'POWER_I_I.fits' % i)
    #     data = numpy.squeeze(pyfits.getdata(file_name))
    #     im.set_data(data)
    #     im.autoscale()
    #     label2.set_text('station-%04i' % i)
    #     print('Frame %i took %.2f s' % (i, time.time() - t0))
    #     return im, label2
    #
    # ani = animation.FuncAnimation(fig, animate, interval=100,
    #                               frames=range(564), blit=False)
    # mp4_file = 'beams.mp4'
    # if os.path.isfile(mp4_file):
    #     os.remove(mp4_file)
    # ani.save(mp4_file, fps=30, bitrate=5000)
