# -*- coding: utf-8 -*-
from __future__ import print_function
import pyfits
import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from os.path import join
from math import ceil


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

    # for i in range(len(beam_dirs)):
    for i in range(5):
        print()
        print('=== %s ===' % beam_dirs[i])
        telescope_id = int(beam_dirs[i][11:13])

        layout = numpy.loadtxt(join(model_dir, telescope_models[telescope_id],
                                    'layout.txt'))
        num_stations = layout.shape[0]
        st = numpy.loadtxt(join(model_dir, telescope_models[telescope_id],
                                'station000', 'layout.txt'))
        num_antennas = st.shape[0]
        norm = num_antennas**2
        print(telescope_models[telescope_id])
        print(num_antennas, num_stations)

        beam_file = join(beam_root_dir, beam_dirs[i],
                         'b_TIME_AVG_CHAN_AVG_CROSS_POWER_AMP_I_I.fits')
        data = numpy.squeeze(pyfits.getdata(beam_file))
        print(numpy.nanmax(data))

        im_size = data.shape[0]
        lm_inc = 2.0 / im_size  # Not 100% sure this is right... CHECK
        x = numpy.arange(im_size) * lm_inc - 1.0
        x, y = numpy.meshgrid(x, x[::-1])
        r = (x**2 + y**2)**0.5
        extent = [-1.0 - lm_inc / 2.0, 1.0 - lm_inc / 2.0,
                  -1.0 - lm_inc / 2.0, 1.0 - lm_inc / 2.0]
        data_max = numpy.nanmax(data)
        norm_data = (data / data_max)
        data_db = 10.0 * numpy.log10(norm_data)

        x_ = r.flatten()
        y_ = data.flatten()
        sort_idx = numpy.argsort(x_)
        x_ = x_[sort_idx]
        y_ = y_[sort_idx]
        y_max = numpy.nanmax(y_)
        norm_y = (y_ / y_max)
        y_db = 10.0 * numpy.log10(norm_y)
        print('min y_db = ', numpy.min(y_db))

        db_lim = -50.0
        i0 = numpy.where(y_db < db_lim)[0][0]
        # Find min near each index
        cw = int(ceil(i0 / 2))
        min0 = y_db[i0-cw:i0+cw].min()
        i0 = numpy.where(y_db == min0)[0][0]
        r0 = x_[i0]
        i1 = numpy.where(x_ >= 2.0 * r0)[0][0]
        r1 = x_[i1]

        fig = pyplot.figure(figsize=(12, 5))
        fig.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.98,
                            hspace=0.0, wspace=0.25)
        ax0 = fig.add_subplot(121)
        data_max = numpy.nanmax(data)
        norm_data = (data / data_max)
        data_db = 10.0 * numpy.log10(norm_data)
        pyplot.tick_params(axis='both', which='major', labelsize='small')
        pyplot.tick_params(axis='both', which='minor', labelsize='small')

        circle = pyplot.Circle((0.0, 0.0), r0,
                           color='c', linestyle='-',
                           fill=False, alpha=1.0, lw=1.0)
        ax0.add_artist(circle)

        circle = pyplot.Circle((0.0, 0.0), r1,
                           color='c', linestyle='-',
                           fill=False, alpha=1.0, lw=1.0)
        ax0.add_artist(circle)


        im = ax0.imshow(data_db, interpolation='nearest', cmap='inferno',
                        extent=extent, vmin=-60.0, vmax=0.0)
        divider = make_axes_locatable(ax0)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax0.figure.colorbar(im, cax=cax2)
        cbar.set_label('Decibels', fontsize='small')
        cbar.ax.tick_params(labelsize='small')
        pyplot.tick_params(axis='both', which='major', labelsize='small')
        pyplot.tick_params(axis='both', which='minor', labelsize='small')

        ax1 = fig.add_subplot(122)
        ax1.plot(x_, y_db, 'k.', alpha=0.2, ms=1.0)
        ax1.plot(ax1.get_xlim(), [-60, -60], 'r--', alpha=0.2)
        ax1.set_ylim(-80.0, 0.0)
        ax1.set_xlim(0.0, 1.0)
        ax1.grid()
        ax1.plot([r0, r0], ax1.get_ylim(), 'b--')
        ax1.plot([r1, r1], ax1.get_ylim(), 'b--')
        # ax1.set_title('%s\n%s' %
        #               (beam_dirs[i], telescope_models[telescope_id]),
        #               fontsize='small')
        ax1.set_ylabel('Average cross-power Stokes-I beam, decibels',
                       fontsize='small')
        ax1.set_xlabel('Phase centre distance, direction cosine',
                       fontsize='small')

        pyplot.tick_params(axis='both', which='major', labelsize='small')
        pyplot.tick_params(axis='both', which='minor', labelsize='small')
        # pyplot.savefig(join(beam_root_dir, beam_dirs[i],
        #                     'ave_beam_metrics_3d_2d_%02i.png' % i))
        pyplot.savefig('ave_beam_metrics_3d_2d_%02i.png' % i)


if __name__ == '__main__':
    main()
