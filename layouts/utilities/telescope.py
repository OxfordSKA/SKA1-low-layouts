# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from . import generators
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join


class Telescope(object):
    def __init__(self):
        self.lon_deg = 0
        self.lat_deg = 0
        self.alt_m = 0
        self.station_diameter_m = 35
        self.trail_timeout_s = 2.0  # s
        self.num_trials = 5
        self.verbose = False
        self.seed = None
        self.layout = dict()

    def add_uniform_core(self, num_stations, r_max_m):
        """Add uniform random core"""
        self.layout['uniform_core'] = generators.uniform_core(
            num_stations, r_max_m, self.station_diameter_m,
            self.trail_timeout_s, self.num_trials, self.seed,
            self.verbose)

    def add_tapered_core(self):
        """Add a tapered core"""
        pass

    def add_arms(self):
        pass

    def add_ska1_v5(self, r_min=None, r_max=None):
        """Add ska1 v5 layout for r >= r_min and r <= r_max if defined"""
        path = os.path.dirname(os.path.abspath(__file__))
        coords = np.loadtxt(join(path, 'data', 'v5_enu.txt'))
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        r = (x**2 + y**2)**0.5
        n0 = x.shape[0]
        if r_min and r_max:
            idx = np.where(np.logical_and(r >= r_min, r <= r_max))
            x, y, z = x[idx], y[idx], z[idx]
        elif r_min:
            idx = np.where(r >= r_min)
            x, y, z = x[idx], y[idx], z[idx]
        elif r_max:
            idx = np.where(r <= r_max)
            x, y, z = x[idx], y[idx], z[idx]
        self.layout['ska1_v5'] = {'x': x, 'y': y, 'z': z}
        n1 = x.shape[0]
        print('- Ska1 v5 radius filter removed', n0 - n1, 'stations')

    def num_stations(self):
        if not self.layout:
            raise RuntimeError('No layout defined!')
        n = 0
        for name in self.layout:
            layout = self.layout[name]
            n += layout['x'].shape[0]
        return n

    def coords_enu(self):
        if not self.layout:
            raise RuntimeError('No layout defined!')
        n = self.num_stations()
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        i = 0
        for name in self.layout:
            layout = self.layout[name]
            n0 = layout['x'].shape[0]
            x[i:i+n0] = layout['x']
            y[i:i+n0] = layout['y']
            if 'z' in layout:
                z[i:i+n0] = layout['z']
            i += n0
        return x, y, z

    def create_figure(self, plot_r=None):
        if not self.layout:
            raise RuntimeError('No layout defined, nothing to plot!')
        fig, ax = plt.subplots(figsize=(8, 8))
        r_max = 0
        for name in self.layout:
            layout = self.layout[name]
            print(name, layout.keys())
            r = (layout['x']**2 + layout['y']**2)**0.5
            r_max = max(np.max(r), r_max)
            for xy in zip(layout['x'], layout['y']):
                c = plt.Circle((xy), radius=(self.station_diameter_m / 2),
                               fill=False, color='k')
                ax.add_artist(c)
        if plot_r:
            r_max = plot_r
        ax.set_xlim(-r_max*1.1, r_max*1.1)
        ax.set_ylim(-r_max*1.1, r_max*1.1)
        ax.set_aspect('equal')
        return fig



