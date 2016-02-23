# -*- coding: utf-8 -*-
from __future__ import print_function
import subprocess
import os
from os.path import join
import numpy
import pyfits
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time


class OskarSettings(object):
    """Interface between OSKAR settings ini files with Python dictionaries."""

    def __init__(self):
        self.ini_separator = '/'
        self.ini = ''

    def dict_to_ini(self, settings, ini_file, overwrite=False, verbose=False):
        """Convert dictionary of settings to ini file."""
        self.ini = os.path.abspath(ini_file)
        if overwrite and os.path.exists(self.ini):
            if verbose:
                print("Removing ini file", self.ini)
            os.remove(self.ini)
        ini_dir = os.path.dirname(self.ini)
        if not ini_dir == "" and not os.path.isdir(ini_dir):
            os.makedirs(ini_dir)
        for key in sorted(settings):
            if isinstance(settings[key], dict):
                self.__iterate_settings(settings[key],
                                        key + self.ini_separator)

    def __iterate_settings(self, node, group):
        for key in sorted(node):
            if isinstance(node[key], dict):
                self.__iterate_settings(node[key],
                                        group + key + self.ini_separator)
            else:
                self.__set_setting(group + key, node[key])

    def __set_setting(self, key, value):
        subprocess.call(["oskar_settings_set", "-q",
                         self.ini, key, str(value)])


def create_settings(out_dir, ini_file, mjd, ra, dec, freq_hz, telescope_model,
                    coord_frame='Equatorial', imsize=512):
    s = dict()
    s['simulator'] = {
        'double_precision': False,
        'max_sources_per_chunk': 8192
    }
    s['observation'] = {
        'phase_centre_ra_deg': ra,
        'phase_centre_dec_deg': dec,
        'start_time_utc': mjd,
        'start_frequency_hz': freq_hz,
        'length': 1.0
    }
    s['telescope'] = {
         'input_directory': telescope_model,
         'latitude_deg': -26.697024,
         'longitude_deg': 116.631289,
         'normalise_beams_at_phase_centre': False,
         'pol_mode': 'Scalar'
    }
    s['beam_pattern'] = {
        'all_stations': True,
        'coordinate_frame': coord_frame,
        'beam_image': {
            'size': imsize,
            'fov_deg': 180.0
        },
        'root_path': join(out_dir, 'b'),
        'output': {
            'average_time_and_channel': True,
            'separate_time_and_channel': False
        },
        'telescope_outputs': {
            'fits_image': {
                'cross_power_stokes_i_amp': True,
            }
        },
        'station_outputs': {
            'fits_image': {
                'auto_power_stokes_i': True,
            }
        }
    }
    settings = OskarSettings()
    settings.dict_to_ini(s, ini_file, overwrite=True, verbose=True)


def auto_beam_movie(model_dir, beams_dir, imsize):
    mp4_file = join(beams_dir, 'auto_beams.mp4')
    if os.path.isfile(mp4_file):
        return
    t0 = time.time()
    fig = pyplot.figure(figsize=(14, 6.5))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92,
                        hspace=0.0, wspace=0.05)
    ax1 = fig.add_subplot(121, aspect='equal')
    line1, = ax1.plot([], [], 'k+')
    label1 = ax1.text(0.02, 0.98, '', ha='left', va='top', style='italic',
                      color='k', transform=ax1.transAxes, fontsize='x-small')
    ax1.set_title('Station antenna positions', fontsize='small')
    ax1.set_xlabel('East [m]', fontsize='small')
    ax1.set_ylabel('North [m]', fontsize='small')
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.grid()

    ax2 = fig.add_subplot(122, aspect='equal')
    label2 = ax2.text(0.02, 0.98, '', ha='left', va='top', style='italic',
                      color='k', transform=ax2.transAxes, fontsize='x-small')
    im = ax2.imshow(numpy.random.random((imsize, imsize)),
                    interpolation='nearest', animated=True,
                    vmin=-40, vmax=0.0, cmap='seismic')
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
    ax2.set_title('Auto-power beam', fontsize='small')

    def animate(i):
        file_name = join(beams_dir, 'b_S%04i_TIME_AVG_CHAN_AVG_AUTO_'
                                    'POWER_I_I.fits' % i)
        data = numpy.squeeze(pyfits.getdata(file_name))
        data_max = numpy.nanmax(data)
        norm_data = (data / data_max)
        data_db = 10.0 * numpy.log10(norm_data)
        im.set_data(data_db)
        label2.set_text('station-%04i' % i)
        file_name = join(model_dir, 'station%03i' % i, 'layout.txt')
        coords = numpy.loadtxt(file_name)
        x = coords[:, 0]
        y = coords[:, 1]
        line1.set_data(x, y)
        label1.set_text('station-%04i' % i)
        return im, label2, line1, label1

    num_stations = len([s for s in os.listdir(model_dir)
                        if os.path.isdir(join(model_dir, s)) and
                        s.startswith('station')])
    ani = animation.FuncAnimation(fig, animate, interval=50,
                                  frames=range(num_stations), blit=False)
    if os.path.isfile(mp4_file):
        os.remove(mp4_file)
    ani.save(mp4_file, fps=10, bitrate=5000)
    print('- Movie took %.2f s' % (time.time() - t0))


def cross_beam_image(model, out_dir, imsize):
    plot_file = join(out_dir, 'ave_cross_power_beam.png')
    if os.path.isfile(plot_file):
        return
    file_name = join(out_dir, 'b_TIME_AVG_CHAN_AVG_CROSS_POWER_AMP_I_I.fits')
    data = numpy.squeeze(pyfits.getdata(file_name))
    data_max = numpy.nanmax(data)
    norm_data = (data / data_max)
    data_db = 10.0 * numpy.log10(norm_data)
    fig = pyplot.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92,
                        hspace=0.0, wspace=0.05)
    ax1 = fig.add_subplot(111, aspect='equal')
    im = ax1.imshow(data_db, interpolation='nearest', vmin=-60, vmax=0.0,
                    cmap='seismic')
    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax1.figure.colorbar(im, cax=cax2)
    cbar.set_label('Decibels', fontsize='small')
    cbar.ax.tick_params(labelsize='small')
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax1.set_xlabel('East <-> West', fontsize='small')
    ax1.set_ylabel('North <-> South', fontsize='small')
    ax1.grid()
    ax1.set_title('Average cross-power beam', fontsize='small')
    fig.savefig(plot_file)


def main():
    # Pointings
    pointings_ = numpy.loadtxt('pointings.txt')
    idx = [0, 2]  # Selection indices for pointings.
    az = pointings_[idx, 3]
    el = pointings_[idx, 4]
    pointings = zip(pointings_[idx, 0], pointings_[idx, 1], pointings_[idx, 2])

    # Frequencies
    # freq_hz = [50.0e6, 350.0e6]
    freq_hz = [350.0e6]

    # Telescopes
    model_dir = 'models'
    models = ['v4a_fixed_lattice_aligned.tm',
              'v4a_fixed_lattice_not_aligned.tm']

    # Coord. frames
    # frames = ['equatorial', 'horizon']
    frames = ['equatorial']

    imsize = 512

    total = len(models) * len(frames) * len(az) * len(freq_hz)
    ii = 0
    for t, model in enumerate(models):
        model = join(model_dir, model)
        for frame in frames:
            for p, (ra, dec, mjd) in enumerate(pointings):
                for freq in freq_hz:
                    if ii >= 2:
                        continue
                    out_dir = ('beams_%03i_t%02i_%c_p%i_%05.1fMHz' %
                               (ii, t, frame[0], p, freq / 1.0e6))
                    ini_file = 'config_%03i.ini' % ii
                    print('=' * 80)
                    print('%03i/%03i : %s' % (ii + 1, total, out_dir))
                    print(' - model:', model)
                    print(' - elevation:', el[p])
                    ii += 1
                    if os.path.isdir(out_dir):
                        print(' - INFO: Results already exists, skipping.')
                        continue
                    create_settings(out_dir, ini_file, mjd, ra, dec, freq,
                                    model, frame, imsize)
                    os.makedirs(out_dir)
                    subprocess.call(['oskar_sim_beam_pattern', '-q', ini_file])
                    os.remove(ini_file)

    print()
    print()
    ii = 0
    for t, model in enumerate(models):
        model = join(model_dir, model)
        for frame in frames:
            for p, (ra, dec, mjd) in enumerate(pointings):
                for freq in freq_hz:
                    out_dir = ('beams_%03i_t%02i_%c_p%i_%05.1fMHz' %
                               (ii, t, frame[0], p, freq / 1.0e6))
                    ii += 1
                    if not os.path.isdir(out_dir):
                        continue
                    print('*' * 80)
                    print('%03i/%03i : %s' % (ii, total, out_dir))
                    print('*' * 80)
                    print('-', out_dir)
                    print('-', model)
                    auto_beam_movie(model, out_dir, imsize)
                    cross_beam_image(model, out_dir, imsize)

    ii = 0
    for t, model in enumerate(models):
        for frame in frames:
            for p, (ra, dec, mjd) in enumerate(pointings):
                for freq in freq_hz:
                    out_dir = ('beams_%03i_t%02i_%c_p%i_%05.1fMHz' %
                               (ii, t, frame[0], p, freq / 1.0e6))
                    ii += 1
                    if not os.path.exists(out_dir):
                        continue
                    fits_files = [f for f in os.listdir(out_dir)
                                  if f.endswith('.fits')
                                  and os.path.isfile(join(out_dir, f))]
                    for file in fits_files:
                        os.remove(join(out_dir, file))

if __name__ == '__main__':
    main()
