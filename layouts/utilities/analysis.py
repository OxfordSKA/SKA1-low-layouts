# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from os.path import join
from . import telescope
from pyuvwsim import (evaluate_baseline_uvw_ha_dec, convert_enu_to_ecef)
from math import pi, radians, degrees, ceil, asin, sin, log10
import oskar
from astropy import constants as const
from astropy.visualization import (HistEqStretch, SqrtStretch,
                                   LogStretch)
from astropy.visualization.mpl_normalize import ImageNormalize


class TelescopeAnalysis(telescope.Telescope):
    def __init__(self):
        telescope.Telescope.__init__(self)
        self.dec_deg = 0
        self.obs_length_h = 0
        self.num_times = 1
        self.uu_m = None
        self.vv_m = None
        self.ww_m = None
        self.r_uv_m = None
        self.uv_grid = None
        self.grid_cell_size_m = self.station_diameter_m
        self.freq_hz = 100e6

    def clear(self):
        self.clear_layouts()
        self.obs_length_h = 0
        self.num_times = 1

    def clear_layouts(self):
        super(TelescopeAnalysis, self).clear_layouts()
        self.uu_m = None
        self.vv_m = None
        self.ww_m = None
        self.r_uv_m = None
        self.uv_grid = None

    def gen_uvw_coords(self):
        """Generate uvw coordinates"""
        x, y, z = self.get_coords_enu()
        x, y, z = convert_enu_to_ecef(x, y, z, radians(self.lon_deg),
                                      radians(self.lat_deg), self.alt_m)
        num_stations = x.shape[0]
        num_baselines = num_stations * (num_stations - 1) // 2
        n = num_baselines * self.num_times
        self.uu_m, self.vv_m, self.ww_m = np.zeros(n), np.zeros(n), np.zeros(n)
        ha_off = ((self.obs_length_h / 2) / 24) * (2 * pi)
        for i, ha in enumerate(np.linspace(-ha_off, ha_off, self.num_times)):
            uu_, vv_, ww_ = evaluate_baseline_uvw_ha_dec(
                x, y, z, ha - radians(self.lon_deg), radians(self.dec_deg))
            self.uu_m[i * num_baselines: (i + 1) * num_baselines] = uu_
            self.vv_m[i * num_baselines: (i + 1) * num_baselines] = vv_
            self.ww_m[i * num_baselines: (i + 1) * num_baselines] = ww_
        self.r_uv_m = (self.uu_m**2 + self.vv_m**2)**0.5

    def num_coords(self):
        return self.uu_m.shape[0]

    def grid_uvw_coords(self):
        b_max = self.r_uv_m.max()
        grid_size = int(ceil(b_max / self.grid_cell_size_m)) * 2 + \
                    self.station_diameter_m
        if grid_size % 2 == 1:
            grid_size += 1
        wavelength = const.c.value / self.freq_hz
        # delta theta = 1 / (n * delta u)
        cell_lm = 1.0 / (grid_size * (self.grid_cell_size_m / wavelength))
        lm_max = grid_size * sin(cell_lm) / 2
        uv_grid_fov_deg = degrees(asin(lm_max)) * 2
        imager = oskar.Imager('single')
        imager.set_grid_kernel('pillbox', 1,  1)
        imager.set_size(grid_size)
        imager.set_fov(uv_grid_fov_deg)
        weight_ = np.ones_like(self.uu_m)
        amps_ = np.ones_like(self.uu_m, dtype='c8')
        self.uv_grid = np.zeros((grid_size, grid_size), dtype='c8')
        norm = imager.update_plane(self.uu_m / wavelength,
                                   self.vv_m / wavelength,
                                   self.ww_m / wavelength, amps_,
                                   weight_, self.uv_grid, plane_norm=0.0)
        norm = imager.update_plane(-self.uu_m / wavelength,
                                   -self.vv_m / wavelength,
                                   -self.ww_m / wavelength, amps_,
                                   weight_, self.uv_grid, plane_norm=norm)
        if not int(norm) == self.uu_m.shape[0] * 2:
            raise RuntimeError('Gridding uv coordinates failed, '
                               'grid sum = %i != number of points gridded = %i'
                               % (norm, self.uu_m.shape[0] * 2))

    def plot_grid(self, filename=None, show=False, plot_radii=[],
                  x_lim=None, y_lim=None):
        grid_size = self.uv_grid.shape[0]
        wavelength = const.c.value / self.freq_hz
        cell_wavelengths = self.grid_cell_size_m / wavelength
        extent0 = grid_size // 2 * cell_wavelengths
        offset = 0.5 * cell_wavelengths
        extent = [-(-extent0 - offset), -(extent0 - offset),
                  -extent0 - offset, extent0 - offset]
        extent = np.array(extent) * wavelength  # convert to metres

        fig, ax = plt.subplots(figsize=(8, 8), ncols=1, nrows=1)
        fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                            wspace=0.2, hspace=0.2)
        image = self.uv_grid.real
        options = dict(interpolation='nearest', cmap='gray_r', extent=extent,
                       origin='lower')
        im = ax.imshow(image, norm=ImageNormalize(stretch=LogStretch()),
                       **options)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.03)
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.set_label('baselines per pixel')
        cbar.ax.tick_params(labelsize='small')
        ticks = np.arange(5) * round(image.max() / 5)
        ticks = np.append(ticks, image.max())
        cbar.set_ticks(ticks, update_ticks=True)

        for r in plot_radii:
            ax.add_artist(plt.Circle((0, 0), r, fill=False, color='r'))
        ax.set_xlabel('uu (m)')
        ax.set_ylabel('vv (m)')
        if not x_lim is None:
            ax.set_xlim(x_lim)
        if not y_lim is None:
            ax.set_ylim(y_lim)
        if filename is not None:
            fig.savefig(filename)
        if show:
            plt.show()
        if filename is not None or show:
            plt.close(fig)
        else:
            return fig


    def uv_hist(self, num_bins=100, b_min=None, b_max=None, plot=True):
        b_max = self.r_uv_m.max() if b_max is None else b_max
        b_min = self.r_uv_m.min() if b_min is None else b_min

        if plot:
            fig, (ax, ax2, ax3, ax4) = plt.subplots(figsize=(8, 8), nrows=4)

            bins = np.linspace(b_min, b_max, num_bins + 1)
            n, _ = np.histogram(self.r_uv_m, bins=bins, density=False)
            x = (bins[1:] + bins[:-1]) / 2

            ax.bar(x, n, width=np.diff(bins), alpha=0.8, align='center', lw=0.5)
            ax.set_xlabel('baseline length (m)')
            ax.set_ylabel('Number of baselines')
            ax.set_xlim(0, b_max * 1.1)

            ax2.bar(x, n, width=np.diff(bins), alpha=0.8, align='center', lw=0.5)
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_xlabel('baseline length (m)')
            ax2.set_ylabel('Number of baselines')
            ax2.set_xlim(0, b_max * 1.1)

            bins = np.logspace(log10(b_min), log10(b_max), num_bins + 1)
            n, _ = np.histogram(self.r_uv_m, bins=bins, density=False)
            x = (bins[1:] + bins[:-1]) / 2

            ax3.bar(x, n, width=np.diff(bins), alpha=0.8, align='center', lw=0.5)
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3.set_xlabel('baseline length (m)')
            ax3.set_ylabel('Number of baselines')
            ax3.set_xlim(0, b_max * 1.1)

            ax4.bar(x, n, width=np.diff(bins), alpha=0.8, align='center', lw=0.5)
            ax4.set_xlabel('baseline length (m)')
            ax4.set_ylabel('Number of baselines')
            ax4.set_xlim(0, b_max * 1.1)

            plt.show()


    def network_graph(self):
        pass

    def psfrms(self):
        pass

    def uvgap(self):
        pass

    def psf(self):
        pass


class SKA1_v5(TelescopeAnalysis):
    def __init__(self):
        TelescopeAnalysis.__init__(self)
        self.add_ska1_v5()
        self.lon_deg = 116.63128900
        self.lat_deg = -26.69702400


class SKA1_low(TelescopeAnalysis):
    def __init__(self):
        TelescopeAnalysis.__init__(self)
        self.lon_deg = 116.63128900
        self.lat_deg = -26.69702400
