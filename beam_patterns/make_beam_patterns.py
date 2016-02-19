# -*- coding: utf-8 -*-
from __future__ import print_function
import subprocess
import os
from os.path import join
import shutil


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


def create_settings(ini_file, telescope_model, freq_hz, elevation, out_dir):
    s = dict()
    s['simulator'] = {
        'double_precision': False,
        'max_sources_per_chunk': 1024
    }
    s['observation'] = {
        'phase_centre_dec_deg': elevation,
        'start_frequency_hz': freq_hz,
        'start_time_utc': 0.0,
        'length': 0.0
    }
    s['telescope'] = {
         'input_directory': telescope_model,
         'latitude_deg': 90.0,
         'normalise_beams_at_phase_centre': False,
         'pol_mode': 'Scalar'
    }
    s['beam_pattern'] = {
        'all_stations': True,
        # 'coordinate_frame': 'Horizon',
        'coordinate_frame': 'Equatorial',
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


if __name__ == '__main__':

    # Create settings.
    ini_file = 'test.ini'
    # TODO include pointing frequency and telescope name in the folder?
    # TODO copy settings to output directory.
    out_dir = 'beams_v4a_fixed_lattice_aligned'
    telescope_model = join('..', 'oskar_models', 'v4a_fixed_lattice_aligned.tm')
    freq_hz = 350.0e6
    elevation = 45.0
    create_settings(ini_file, telescope_model, freq_hz, elevation, out_dir)

    # Remove existing output directory.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    # Create output directory.
    os.makedirs(out_dir)
    os.makedirs(join(out_dir, 'e'))

    # Run the beam pattern simulation.
    subprocess.call(['oskar_sim_beam_pattern', ini_file])


    # TODO plot nice images of the beams using python
    # aplpy ?
    # Movie of the auto-power beam from each station
    # Image of cross-power average beam
