# -*- coding: utf-8 -*-
from __future__ import print_function
import pyfits
import numpy
from os.path import join
import os
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
import time

if __name__ == '__main__':

    # beams_dir = join('beams_v4a_fixed_lattice_aligned', 'e')
    model_dir = join('..', 'oskar_models', 'v4a_random_lattice_aligned.tm')
    beams_dir = join('beams_v4a_random_lattice_aligned', 'e')
    fig = pyplot.figure(figsize=(16, 7))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=1.0, top=0.92,
                        hspace=0.0, wspace=0.0)
    ax1 = fig.add_subplot(121, aspect='equal')
    ax2 = fig.add_subplot(122, aspect='equal')
    label2 = ax2.text(0.02, 0.98, '', ha='left', va='top', style='italic',
                      color='k', transform=ax2.transAxes, fontsize='x-small')
    im = ax2.imshow(numpy.random.random((512, 512)), interpolation='nearest',
                    animated=True, vmin=-60, vmax=0.0)
    cbar = ax2.figure.colorbar(im, ax=ax2)
    cbar.set_label('db', fontsize='small')
    cbar.ax.tick_params(labelsize='small')
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax2.set_xlabel('East <-> West', fontsize='small')
    ax2.set_ylabel('North <-> South', fontsize='small')
    ax2.grid()
    ax2.set_title('Auto-power beam')

    line1, = ax1.plot([], [], 'k+')
    label1 = ax1.text(0.02, 0.98, '', ha='left', va='top', style='italic',
                      color='k', transform=ax1.transAxes, fontsize='x-small')
    ax1.set_title('Station antenna positions')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.grid()

    def animate(i):
        t0 = time.time()
        file_name = join(beams_dir, 'beam_S%04i_TIME_AVG_CHAN_AVG_AUTO_'
                                    'POWER_I_I.fits' % i)
        data = numpy.squeeze(pyfits.getdata(file_name))
        data_db = 10.0 * numpy.log10(data / numpy.nanmax(data))
        im.set_data(data_db)
        # im.autoscale()
        label2.set_text('station-%04i' % i)
        print('Frame %i took %.2f s' % (i, time.time() - t0))
        file_name = join(model_dir, 'station%03i' % i, 'layout.txt')
        coords = numpy.loadtxt(file_name)
        x = coords[:, 0]
        y = coords[:, 1]
        line1.set_data(x, y)
        label1.set_text('station-%04i' % i)
        return im, label2, line1, label1

    ani = animation.FuncAnimation(fig, animate, interval=100,
                                  frames=range(20), blit=False)
    mp4_file = 'beams.mp4'
    if os.path.isfile(mp4_file):
        os.remove(mp4_file)
    ani.save(mp4_file, fps=30, bitrate=5000)
