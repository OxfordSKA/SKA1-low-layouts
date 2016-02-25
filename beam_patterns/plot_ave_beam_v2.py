# -*- coding: utf-8 -*-
from __future__ import print_function
import pyfits
import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from os.path import join


# def cross_beam_image(model, out_dir, imsize, min_db):
#     cmap = 'inferno'
#     plot_file = join(out_dir, 'ave_cross_power_beam_%.1fdb.png' % min_db)
#     if os.path.isfile(plot_file):
#         return
#     file_name = join(out_dir, 'b_TIME_AVG_CHAN_AVG_CROSS_POWER_AMP_I_I.fits')
#     data = numpy.squeeze(pyfits.getdata(file_name))
#     data_max = numpy.nanmax(data)
#     norm_data = (data / data_max)
#     data_db = 10.0 * numpy.log10(norm_data)
#     fig = pyplot.figure(figsize=(8, 8))
#     fig.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.92,
#                         hspace=0.0, wspace=0.05)
#     ax1 = fig.add_subplot(111, aspect='equal')
#     im = ax1.imshow(data_db, interpolation='nearest', vmin=min_db, vmax=0.0,
#                     cmap=cmap)
#     divider = make_axes_locatable(ax1)
#     cax2 = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = ax1.figure.colorbar(im, cax=cax2)
#     cbar.set_label('Decibels', fontsize='small')
#     cbar.ax.tick_params(labelsize='small')
#     ax1.axes.get_xaxis().set_ticks([])
#     ax1.axes.get_yaxis().set_ticks([])
#     ax1.set_xlabel('East <-> West', fontsize='small')
#     ax1.set_ylabel('North <-> South', fontsize='small')
#     ax1.grid()
#     ax1.set_title('Average cross-power beam', fontsize='small')
#     fig.savefig(plot_file)


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

        # =====================================================================
        # fig = pyplot.figure(figsize=(8, 8))
        # fig.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.92,
        #                     hspace=0.0, wspace=0.05)
        # ax = fig.add_subplot(111, aspect='equal')
        # data_max = numpy.nanmax(data)
        # norm_data = (data / data_max)
        # data_db = 10.0 * numpy.log10(norm_data)
        # pyplot.tick_params(axis='both', which='major', labelsize='small')
        # pyplot.tick_params(axis='both', which='minor', labelsize='small')
        # im = ax.imshow(data_db, interpolation='nearest', cmap='inferno',
        #                extent=extent, vmin=-40.0, vmax=0.0)
        # divider = make_axes_locatable(ax)
        # cax2 = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = ax.figure.colorbar(im, cax=cax2)
        # cbar.set_label('Decibels', fontsize='small')
        # cbar.ax.tick_params(labelsize='small')
        # ax.axes.get_xaxis().set_ticks([])
        # ax.axes.get_yaxis().set_ticks([])
        # ax.set_xlabel('East <-> West', fontsize='small')
        # ax.set_ylabel('North <-> South', fontsize='small')
        # ax.set_title('Average cross-power beam', fontsize='small')
        # pyplot.savefig(join(beam_root_dir, beam_dirs[i],
        #                 'ave_beam_3d_2d_40db_%02i.png' % i))

        # =====================================================================
        x_ = r.flatten()
        y_ = data.flatten()
        sort_idx = numpy.argsort(x_)
        x_ = x_[sort_idx]
        y_ = y_[sort_idx]
        y_max = numpy.nanmax(y_)
        norm_y = (y_ / y_max)
        y_db = 10.0 * numpy.log10(norm_y)

        fig = pyplot.figure(figsize=(8, 8))
        fig.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.92,
                            hspace=0.0, wspace=0.05)
        ax = fig.add_subplot(111)
        ax.plot(x_, y_db, 'k.', alpha=0.2, ms=1.0)
        ax.plot(ax.get_xlim(), [-40, -40], 'r--', alpha=0.5)
        ax.set_ylim(-80.0, 0.0)
        ax.set_xlim(0.0, 1.0)
        ax.grid()
        ax.set_ylabel('Decibels', fontsize='small')
        ax.set_xlabel('Phase centre distance, direction cosine',
                       fontsize='small')
        ax.set_title('Average cross-power beam (radial average)', fontsize='small')
        pyplot.tick_params(axis='both', which='major', labelsize='small')
        pyplot.tick_params(axis='both', which='minor', labelsize='small')
        fig.savefig(join(beam_root_dir, beam_dirs[i],
                         'ave_beam_radial_%02i.png' % i))

if __name__ == '__main__':
    main()
