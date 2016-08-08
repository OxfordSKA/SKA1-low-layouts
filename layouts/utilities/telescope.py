# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from math import log, radians, degrees, cos, sin, pi, acosh
from .layout import Layout
from . import generators


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
        self.layouts = dict()

    def clear_layouts(self):
        self.layouts.clear()

    def save(self, filename):
        """save the telescope model as an npz file"""
        pass

    def load(self, filename):
        """save the telescope model as an npz file"""
        pass

    def to_oskar_telescope_model(self, filename):
        pass

    def add_uniform_core(self, num_stations, r_max_m):
        """Add uniform random core"""
        self.layouts['uniform_core'] = generators.uniform_core(
            num_stations, r_max_m, self.station_diameter_m,
            self.trail_timeout_s, self.num_trials, self.seed,
            self.verbose)

    def add_tapered_core(self, num_stations, r_max_m, taper_func, **kwargs):
        """Add a tapered core"""
        try:
            layout = Layout.rand_tapered_2d_trials(
                num_stations, r_max_m, self.station_diameter_m,
                self.trail_timeout_s, taper_func, self.num_trials, self.seed,
                0.0, **kwargs)
            self.layouts['tapered_core'] = dict(
                x=layout.x, y=layout.y, info=layout.info,
                taper_func=taper_func, r_max_m=r_max_m, kwargs=kwargs)
        except RuntimeError as e:
            print('*** ERROR ***:', e.message)


    @staticmethod
    def rotate_coords(x, y, angle):
        theta = radians(angle)
        xr = x * cos(theta) - y * sin(theta)
        yr = x * sin(theta) + y * cos(theta)
        return xr, yr

    @staticmethod
    def log_spiral(n, r0, r1, b):
        t_max = log(r1 / r0) * (1 /b)
        t = np.linspace(0, t_max, n)
        tmp = r0 * np.exp(b * t)
        x = tmp * np.cos(t)
        y = tmp * np.sin(t)
        return x, y

    @staticmethod
    def spiral_to_arms(x, y, num_arms, theta0_deg=0.0):
        delta_theta = 360 / num_arms
        for i in range(num_arms):
            x[i::num_arms], y[i::num_arms] = Telescope.rotate_coords(
                x[i::num_arms], y[i::num_arms], theta0_deg + delta_theta * i)
        return x, y

    def add_log_spiral(self, n, r0, r1, b, num_arms):
        """Add spiral arms by rotating a single spiral of n positions"""
        x, y = self.log_spiral(n, r0, r1, b)
        x, y = self.spiral_to_arms(x, y, num_arms)
        self.layouts['spiral_arms'] = {'x': x, 'y': y}

    def add_log_spiral_clusters(self, num_clusters, num_arms, r0, r1, b,
                                stations_per_cluster, cluster_radius_m):
        """Add spiral arm clusters.
        Spiral arm positions generated come from a single single ar
        Note: the random number generator respects class variables
          self.seed
          self.trail_timeout_s
          self.num_trials
        """
        x_, y_, info = Layout.generate_clusters(
            num_clusters, stations_per_cluster, cluster_radius_m,
            self.station_diameter_m, self.trail_timeout_s, self.num_trials,
            self.seed, r_min=0.0)

        cx, cy = self.log_spiral(num_clusters, r0, r1, b)
        cx, cy = self.spiral_to_arms(cx, cy, num_arms)
        x = np.zeros(num_clusters * stations_per_cluster)
        y = np.zeros(num_clusters * stations_per_cluster)
        for i in range(num_clusters):
            x[i * stations_per_cluster:(i + 1) * stations_per_cluster] = \
                x_[i] + cx[i]
            y[i * stations_per_cluster:(i + 1) * stations_per_cluster] = \
                y_[i] + cy[i]

        self.layouts['spiral_clusters'] = {'x': x, 'y': y, 'cx': cx, 'cy': cy,
                                           'cr': cluster_radius_m}

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
        self.layouts['ska1_v5'] = {'x': x, 'y': y, 'z': z}
        n1 = x.shape[0]
        print('- Ska1 v5 radius filter removed', n0 - n1, 'stations')

    def num_stations(self):
        if not self.layouts:
            raise RuntimeError('No layout defined!')
        n = 0
        for name in self.layouts:
            layout = self.layouts[name]
            n += layout['x'].shape[0]
        return n

    def get_coords_enu(self):
        if not self.layouts:
            raise RuntimeError('No layout defined!')
        n = self.num_stations()
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        i = 0
        for name in self.layouts:
            layout = self.layouts[name]
            n0 = layout['x'].shape[0]
            x[i:i+n0] = layout['x']
            y[i:i+n0] = layout['y']
            if 'z' in layout:
                z[i:i+n0] = layout['z']
            i += n0
        return x, y, z

    def plot_layout(self, plot_r=None, filename=None, show=False,
                    plot_decorations=False, plot_radii=[],
                    x_lim=None, y_lim=None):
        plot_nearest = False
        if not self.layouts:
            raise RuntimeError('No layout defined, nothing to plot!')
        fig, ax = plt.subplots(figsize=(8, 8))
        r_max = 0
        for name in self.layouts:
            layout = self.layouts[name]
            x_, y_ = layout['x'], layout['y']
            r = (x_**2 + y_**2)**0.5
            r_max = max(np.max(r), r_max)

            for xy in zip(x_, y_):
                c = plt.Circle(xy, radius=(self.station_diameter_m / 2),
                               fill=False, color='k')
                ax.add_artist(c)

            if plot_decorations:
                # Plot cluster radii, if present
                if 'cx' in layout and 'cy' in layout and 'cr' in layout:
                    for xy in zip(layout['cx'], layout['cy']):
                        ax.add_artist(plt.Circle(xy, radius=layout['cr'],
                                                 fill=False, color='b',
                                                 alpha=0.5))
                if 'taper_func' in layout and 'r_max_m' in layout:
                    for k, xy in enumerate(zip(x_, y_)):
                        r_ = (self.station_diameter_m / 2) / \
                             layout['taper_func'](r[k] / layout['r_max_m'],
                                                  **layout['kwargs'])
                        ax.add_artist(plt.Circle(xy, r_, fill=False,
                                                 linestyle='-', color='0.5',
                                                 alpha=0.5))

                if plot_nearest and 'info' in layout and \
                                'attempt_id' in layout['info']:
                    info = layout['info']
                    attempt_id = info['attempt_id']
                    if 'i_min' in info[attempt_id]:
                        i_min = info[attempt_id]['i_min']
                        for k, (x, y) in enumerate(zip(x_, y_)):
                            if i_min[k] < 0:
                                continue
                            dx = x_[i_min[k]] - x
                            dy = y_[i_min[k]] - y
                            ax.arrow(x, y, dx, dy,
                                     head_width=1.5, head_length=3,
                                     overhang=0, length_includes_head=False)

        for r in plot_radii:
            ax.add_artist(plt.Circle((0, 0), r, fill=False,
                                     color='r', alpha=0.5))
        if plot_r:
            r_max = plot_r
        if x_lim is None:
            ax.set_xlim(-r_max*1.1, r_max*1.1)
        else:
            ax.set_xlim(x_lim)
        if y_lim is None:
            ax.set_ylim(-r_max*1.1, r_max*1.1)
        else:
            ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        if filename is not None:
            fig.savefig(filename)
        if show:
            plt.show()
        if filename is not None or show:
            plt.close(fig)
        else:
            return fig

    @staticmethod
    def plot_taper(taper_func, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 8))
        r = np.linspace(0, 1, 100)
        y = taper_func(r, **kwargs)
        ax.plot(r, y)
        plt.show()
        plt.close(fig)

    def plot_min_sep(self, r_max, taper_func, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 8))
        r = np.linspace(0, 1, 100)
        y = self.station_diameter_m / taper_func(r, **kwargs)
        ax.plot(r * r_max, y)
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('Minimum separation (m)')
        plt.show()
        plt.close(fig)


def taylor_win(n, sll):
    """
    http://www.dsprelated.com/showcode/6.php

    from http://mathforum.org/kb/message.jspa?messageID=925929:

    A Taylor window is very similar to Chebychev weights. While Chebychev
    weights provide the tighest beamwidth for a given side-lobe level, taylor
    weights provide the least taper loss for a given sidelobe level.

    'Antenna Theory: Analysis and Design' by Constantine Balanis, 2nd
    edition, 1997, pp. 358-368, or 'Modern Antenna Design' by Thomas
    Milligan, 1985, pp.141-148.
    """
    def calculate_fm(m, sp2, a, nbar):
        n = np.arange(1, nbar)
        p = np.hstack([np.arange(1, m), np.arange(m + 1, nbar)])
        num = np.prod((1 - (m**2 / sp2) / (a**2 + (n - 0.5)**2)))
        den = np.prod(1 - m**2 / p**2)
        return ((-1)**(m + 1) * num) / (2 * den)
    nbar = int(np.ceil(2.0 * (acosh(10**(-sll / 20.0)) / pi)**2 + 0.5))
    n *= 2
    a = np.arccosh(10**(-sll / 20)) / pi
    sp2 = nbar**2 / (a**2 + (nbar - 0.5)**2)
    w = np.ones(n)
    fm = np.zeros(nbar)
    summation = 0
    k = np.arange(n)
    xi = (k - 0.5 * n + 0.5) / n
    for m in range(1, nbar):
        fm[m] = calculate_fm(m, sp2, a, nbar)
        summation += fm[m] * np.cos(2 * pi * m * xi)
    w += w * summation
    w /= w.max()
    w = w[n//2:]
    return w





