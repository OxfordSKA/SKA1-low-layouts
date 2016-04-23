# -*- coding: utf-8 -*-
"""Generate PB image for use with uv distribution plots.

The UV distribution should be convolved with the FT(PB) to get a better
estimate of the sensitivity of the power spectrum.
"""

from __future__ import print_function

import csv
import subprocess
import os
from os.path import join
import numpy
import pyfits
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import math


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
                    coord_frame='Equatorial', im_size=512, fov=20.0):
    s = dict()
    s['simulator'] = {
        'double_precision': False,
        'max_sources_per_chunk': 4096
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
            'size': im_size,
            'fov_deg': fov
        },
        'root_path': join(out_dir, 'b'),
        'output': {
            'average_time_and_channel': True,
            'separate_time_and_channel': False
        },
        'telescope_outputs': {
            'fits_image': {
                'cross_power_stokes_i_amp': True,
                'cross_power_stokes_i_phase': True
            }
        },
        'station_outputs': {
            'fits_image': {
                'auto_power_stokes_i': False,
            }
        }
    }
    settings = OskarSettings()
    settings.dict_to_ini(s, ini_file, overwrite=True, verbose=True)


def main():
    # Pointings
    pointings_ = numpy.loadtxt('pointings.txt')
    idx = [0]  # Selection indices for pointings.
    az = pointings_[idx, 3]
    el = pointings_[idx, 4]
    pointings = zip(pointings_[idx, 0], pointings_[idx, 1], pointings_[idx, 2])

    # Frequencies
    freq_hz = [120.0e6]

    # Telescopes
    # model_dir = 'models'
    # models = [d for d in os.listdir(model_dir)
    #           if os.path.isdir(join(model_dir, d))
    #           and d.endswith('.tm')]
    model_dir = os.path.abspath(os.path.curdir)
    models = ['v4d.tm']
    for i, model in enumerate(models):
        print('%i %s' % (i, model))
    # Verify models exist
    for t, model in enumerate(models):
        model = join(model_dir, model)
        if not os.path.isdir(model):
            print('model %s not found!' % model)
            return
        with open(join(model, 'layout.txt')) as f:
            print(sum(1 for _ in f))

    # Coord. frames
    # frames = ['equatorial', 'horizon']
    frames = ['equatorial']

    fov = 180.0
    im_size = 256

    total = len(models) * len(frames) * len(az) * len(freq_hz)
    ii = 0
    for t, model in enumerate(models):
        model = join(model_dir, model)
        for frame in frames:
            for p, (ra, dec, mjd) in enumerate(pointings):
                for freq in freq_hz:
                    t1 = time.time()
                    out_dir = ('beams_%05.1f_%04i_%5.1fMHz' %
                               (fov, im_size, freq / 1.0e6))
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
                                    model, frame, im_size, fov)
                    os.makedirs(out_dir)
                    subprocess.call(['oskar_sim_beam_pattern', ini_file])
                    os.remove(ini_file)
                    print('= DONE! %.2f s' % (time.time() - t1))


if __name__ == '__main__':
    main()
