# -*- coding: utf-8 -*-
from __future__ import print_function
import subprocess
import os
from os.path import join
import shutil
import numpy


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
                    coord_frame='Equatorial'):
    s = dict()
    s['simulator'] = {
        'double_precision': False,
        'max_sources_per_chunk': 1024
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
            'size': 512,
            'fov_deg': 180.0
        },
        'root_path': join(out_dir, 'e/beam'),
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


def main():
    # Pointings
    pointings_ = numpy.loadtxt('pointings.txt')
    idx = [0, 2]  # Selection indices for pointings.
    az = pointings_[idx, 3]
    el = pointings_[idx, 4]
    pointings = zip(pointings_[idx, 0], pointings_[idx, 1], pointings_[idx, 2])
    num_pointings = len(az)

    # Frequencies
    freq_hz = [50.0e6, 350.0e6]

    # Telescopes
    model_dir = 'models'
    models = ['v4a_fixed_lattice_aligned.tm',
              'v4a_fixed_lattice_not_aligned.tm']

    # Coord. frames
    # frames = ['equatorial', 'horizon']
    frames = ['equatorial']

    ii = 0
    for t, model in enumerate(models):
        for frame in frames:
            for p, (ra, dec, mjd) in enumerate(pointings):
                for freq in freq_hz:
                    if ii >= 1:
                        continue
                    out_dir = ('beams_%03i_t%02i_%c_p%i_%05.1fMHz' %
                               (ii, t, frame[0], p, freq / 1.0e6))
                    print('=' * 80)
                    print('%03i : %s' % (ii, out_dir))
                    ini_file = 'config_%03i.ini' % ii
                    create_settings(out_dir, ini_file, mjd, ra, dec, freq,
                                    model, frame)
                    if os.path.isdir(out_dir):
                        shutil.rmtree(out_dir)
                    os.makedirs(out_dir)
                    subprocess.call(['oskar_sim_beam_pattern', ini_file])
                    os.remove(ini_file)
                    ii += 1


if __name__ == '__main__':
    main()
