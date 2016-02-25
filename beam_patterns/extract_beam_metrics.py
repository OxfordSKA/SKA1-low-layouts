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

    for i in range(len(beam_dirs)):
        print('** %s **' % beam_dirs[i])
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

        r0 = 0.22
        r1 = 0.4

        # data[r < r0] = numpy.nan

        fig = pyplot.figure(figsize=(14, 7))
        fig.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.92,
                            hspace=0.0, wspace=0.2)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(data, interpolation='nearest', cmap='inferno',
                        extent=extent)
        circle = pyplot.Circle((0.0, 0.0), r0,
                               color='c', linestyle='--',
                               fill=False, alpha=1.0, lw=2.0)
        ax1.add_artist(circle)
        circle = pyplot.Circle((0.0, 0.0), r1,
                               color='c', linestyle='--',
                               fill=False, alpha=1.0, lw=2.0)
        ax1.add_artist(circle)
        divider = make_axes_locatable(ax1)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax1.figure.colorbar(im, cax=cax2)
        cbar.set_label('Power', fontsize='small')
        cbar.ax.tick_params(labelsize='small')
        ax1.grid()
        ax1.set_title('%s\n%s' % (telescope_models[telescope_id],
                                  beam_dirs[i]), fontsize='small')

        ax2 = fig.add_subplot(122)
        data_max = numpy.nanmax(data)
        norm_data = (data / data_max)
        data_db = 10.0 * numpy.log10(norm_data)
        im = ax2.imshow(data_db, interpolation='nearest', cmap='inferno',
                        vmin=-40, vmax=0.0, extent=extent)
        circle = pyplot.Circle((0.0, 0.0), r0,
                               color='c', linestyle='--',
                               fill=False, alpha=1.0, lw=2.0)
        ax2.add_artist(circle)
        circle = pyplot.Circle((0.0, 0.0), r1,
                               color='c', linestyle='--',
                               fill=False, alpha=1.0, lw=2.0)
        ax2.add_artist(circle)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax2.figure.colorbar(im, cax=cax2)
        cbar.set_label('Decibels', fontsize='small')
        cbar.ax.tick_params(labelsize='small')
        ax2.grid()
        ax2.set_title('%s\n%s' % (telescope_models[telescope_id],
                                  beam_dirs[i]), fontsize='small')
        pyplot.show()
        # ============================


def main2():
    model_dir = 'models'
    telescope_models = [d for d in os.listdir(model_dir)
                        if os.path.isdir(join(model_dir, d))
                        and d.endswith('.tm')]
    beam_root_dir = 'horizon_zenith_beams'
    freqs = [50.0e6, 350.0e6]

    beam_dirs = [d for d in os.listdir(beam_root_dir)
                 if os.path.isdir(join(beam_root_dir, d))
                 and d.startswith('beams_')]

    for i in range(len(beam_dirs)):
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

        fig = pyplot.figure(figsize=(12, 6))
        fig.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.90,
                            hspace=0.0, wspace=0.0)
        ax1 = fig.add_subplot(111)
        x_ = r.flatten()
        y_ = data.flatten()
        sort_idx = numpy.argsort(x_)
        x_ = x_[sort_idx]
        y_ = y_[sort_idx]
        y_max = numpy.nanmax(y_)
        norm_y = (y_ / y_max)
        y_db = 10.0 * numpy.log10(norm_y)
        print('min y_db = ', numpy.min(y_db))

        # Find rough minimum
        lim = -50.0
        i0 = numpy.where(y_db < lim)[0][0]
        i_start = i0 * 1.9
        i1 = numpy.where(y_db[i_start:] < lim)[0][0] + i_start
        # i_start = i1 + i0
        # i2 = numpy.where(y_db[i_start:] < lim)[0][0] + i_start
        # Find min near each index
        cw = int(ceil(i0 / 2))
        min0 = y_db[i0-cw:i0+cw].min()
        min1 = y_db[i1-cw:i1+cw].min()
        # min2 = y_db[i2-cw:i2+cw].min()
        i0 = numpy.where(y_db == min0)[0][0]
        i1 = numpy.where(y_db == min1)[0][0]
        # i2 = numpy.where(y_db == min2)[0][0]
        print('i0', i0, x_[i0], y_db[i0])
        print('i1', i1, x_[i1], y_db[i1])
        # print('i2', i2, x_[i2], y_db[i2])

        cw = int(ceil((i1 - i0) / 3.0))
        max1 = numpy.nanmax(y_db[i0+cw:i1-cw])
        max2 = numpy.nanmax(y_db[i1+cw:])
        # max2 = numpy.nanmax(y_db[i1+cw:i2-cw])
        # max3 = numpy.nanmax(y_db[i2+cw:])
        i_max1 = numpy.where(y_db == max1)[0][0]
        i_max2 = numpy.where(y_db == max2)[0][0]
        # i_max3 = numpy.where(y_db == max3)[0][0]
        r1 = x_[i_max1]
        r2 = x_[i_max2]
        # r3 = x_[i_max3]
        median2 = numpy.nanmedian(y_db[i1:])
        mean2 = numpy.nanmean(y_db[i1:])
        # median3 = numpy.nanmedian(y_db[i2:])
        # mean3 = numpy.nanmean(y_db[i2:])

        ax1.plot(x_, y_db, 'k.', alpha=0.2, ms=1.0)
        ax1.plot([r1], [max1], 'rx', ms=10.0)
        ax1.plot([r2], [max2], 'rx', ms=10.0)
        # ax1.plot([r3], [max3], 'rx', ms=10.0)
        ax1.plot([x_[i1], 1], [max2, max2], 'r--')
        # ax1.plot([x_[i2], 1.0], [median3, median3], 'y-', lw=3.0, alpha=0.5)
        # ax1.plot([x_[i2], 1.0], [mean3, mean3], 'c--', lw=1.0)
        ax1.plot([x_[i1], 1.0], [median2, median2], 'm-', lw=3.0, alpha=0.5)
        ax1.plot([x_[i1], 1.0], [mean2, mean2], 'g--', lw=1.0)

        ax1.plot([x_[i0], x_[i0]], ax1.get_ylim(), 'b--')
        ax1.plot([x_[i1], x_[i1]], ax1.get_ylim(), 'b--')
        # ax1.plot([x_[i2], x_[i2]], ax1.get_ylim(), 'b--')
        ax1.set_ylim(-80.0, 0.0)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_title('%s\n%s' %
                      (beam_dirs[i], telescope_models[telescope_id]),
                      fontsize='small')
        ax1.set_ylabel('Average cross-power Stokes-I beam, decibels',
                       fontsize='small')
        ax1.set_xlabel('Phase centre distance, direction cosine',
                       fontsize='small')
        pyplot.savefig(join(beam_root_dir, beam_dirs[i],
                            'metrics_2d_%02i.png' % i))
        # pyplot.savefig('test_%02i.png' % i)

        f = open(join(beam_root_dir, beam_dirs[i], 'metrics_%02i.txt' % i), 'w')
        f.write('%s\n' % beam_dirs[i])
        f.write('%s\n' % telescope_models[telescope_id])
        f.write('max1    %.1f (r=%.2f)\n' % (max1, r1))
        f.write('max2    %.1f (r=%.2f)\n' % (max2, r2))
        # f.write('max3    %.1f (r=%.2f)\n' % (max3, r3))
        f.write('median2 %.1f (r>%.2f)\n' % (median2, r2))
        # f.write('median3 %.1f (r>%.2f)\n' % (median3, r3))
        f.write('mean2   %.1f (r>%.2f)\n' % (mean2, r2))
        # f.write('mean3   %.1f (r>%.2f)\n' % (mean3, r3))
        f.close()


if __name__ == '__main__':
    # main()
    main2()
