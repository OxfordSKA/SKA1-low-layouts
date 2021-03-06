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
from . import sensitivity
from pyuvwsim import evaluate_baseline_uvw_ha_dec, convert_enu_to_ecef
from math import pi, radians, degrees, ceil, asin, sin, log10, floor, sqrt
from oskar.imager import Imager
from astropy import constants as const
from astropy.visualization import (HistEqStretch, SqrtStretch,
                                   LogStretch)
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


class TelescopeAnalysis(telescope.Telescope):
    def __init__(self, name=''):
        telescope.Telescope.__init__(self, name)
        self.dec_deg = 0
        self.grid_cell_size_m = self.station_diameter_m
        self.freq_hz = 100e6
        self.bandwidth_hz = 100e3
        self.obs_length_h = 0
        self.num_times = 1
        self.uu_m = None
        self.vv_m = None
        self.ww_m = None
        self.r_uv_m = None
        self.uv_grid = None
        self.tree = None
        self.tree_length = 0
        self.hist_n = None
        self.hist_x = None
        self.psf_rms = 0
        self.psf_rms_r = None
        self.psf_rms_r_x = None
        self.psf = None
        self.psf_fov_deg = 0

    def clear_layouts(self):
        super(TelescopeAnalysis, self).clear_layouts()
        self.uu_m = None
        self.vv_m = None
        self.ww_m = None
        self.r_uv_m = None
        self.uv_grid = None
        self.tree = None
        self.tree_length = 0
        self.hist_n = None
        self.hist_x = None
        self.psf_rms = 0
        self.psf_rms_r = None
        self.psf_rms_r_x = None
        self.psf = None
        self.psf_fov_deg = 0

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
        return self.uu_m.size if self.uu_m is not None else 0

    def grid_uvw_coords(self):
        if self.uu_m is None:
            self.gen_uvw_coords()
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
        imager = Imager('single')
        imager.set_grid_kernel('pillbox', 1,  1)
        imager.set_size(grid_size)
        imager.set_fov(uv_grid_fov_deg)
        weight_ = np.ones_like(self.uu_m)
        amps_ = np.ones_like(self.uu_m, dtype='c8')
        self.uv_grid = np.zeros((grid_size, grid_size), dtype='c8')
        norm = imager.update_plane(self.uu_m / wavelength,
                                   self.vv_m / wavelength,
                                   self.ww_m / wavelength,
                                   amps_, weight_, self.uv_grid,
                                   plane_norm=0.0)
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
        if self.uv_grid is None:
            self.grid_uvw_coords()
        grid_size = self.uv_grid.shape[0]
        wavelength = const.c.value / self.freq_hz
        fov_rad = Imager.uv_cellsize_to_fov(self.grid_cell_size_m / wavelength,
                                            grid_size)
        extent = Imager.grid_extent_wavelengths(degrees(fov_rad), grid_size)
        extent = np.array(extent) * wavelength

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

    def uv_hist(self, num_bins=100, b_min=None, b_max=None, make_plot=True,
                log_bins=True, bar=False, filename=None):
        if self.uu_m is None:
            self.gen_uvw_coords()

        b_max = self.r_uv_m.max() if b_max is None else b_max
        b_min = self.r_uv_m.min() if b_min is None else b_min

        if log_bins:
            bins = np.logspace(log10(b_min), log10(b_max), num_bins + 1)
        else:
            bins = np.linspace(b_min, b_max, num_bins + 1)

        self.hist_n, _ = np.histogram(self.r_uv_m, bins=bins, density=False)
        self.hist_x = (bins[1:] + bins[:-1]) / 2
        self.hist_bins = bins

        if make_plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            if bar:
                ax.bar(self.hist_x, self.hist_n, width=np.diff(bins),
                       alpha=0.8, align='center', lw=0.5)
            else:
                ax.plot(self.hist_x, self.hist_n)
            if log_bins:
                ax.set_xscale('log')
            ax.set_xlabel('baseline length (m)')
            ax.set_ylabel('Number of baselines')
            ax.set_xlim(0, b_max * 1.1)
            if filename is not None:
                fig.savefig(filename)
            else:
                plt.show()
            plt.close(fig)

    def uv_sensitivity(self, num_bins=100, b_min=None, b_max=None,
                       log_bins=True):
        if self.hist_n is None:
            self.uv_hist(num_bins, b_min, b_max, log_bins=log_bins,
                         make_plot=False)
        t_int = self.obs_length_h * 3600 if self.obs_length_h > 0 else 1
        b_max = self.r_uv_m.max() if b_max is None else b_max
        beam_solid_angle, _ = sensitivity.beam_solid_angle(self.freq_hz, b_max)
        sigma_t = sensitivity.brightness_temperature_sensitivity(
            self.freq_hz, beam_solid_angle, t_int,
            self.bandwidth_hz)
        print(sigma_t)
        # TODO(BM) check that the sum of hist n == number of baselines
        #  --- ie that the telescope is normalised correctly.
        bar = False
        fig, ax = plt.subplots(figsize=(8, 8))
        y = sigma_t / np.sqrt(self.hist_n)
        if bar:
            ax.bar(self.hist_x, y, width=np.diff(self.hist_bins),
                   alpha=0.8, align='center', lw=0.5)
        else:
            ax.plot(self.hist_x, y)
        if log_bins:
            ax.set_xscale('log')
        ax.set_xlabel('baseline length (m)')
        ax.set_ylabel('Brightness sensitivity (K)')
        ax.set_yscale('log')
        ax.set_xlim(0, b_max * 1.1)
        plt.show()
        plt.close(fig)


    def network_graph(self):
        x, y, _ = self.get_coords_enu()
        coords = np.transpose(np.vstack([x, y]))
        self.tree = minimum_spanning_tree(squareform(pdist(coords))).toarray()
        self.tree_length = np.sum(self.tree)

    def plot_network(self):
        if self.tree is None:
            self.network_graph()
        x, y, _ = self.get_coords_enu()
        fig, ax = plt.subplots()
        self.plot_layout(mpl_ax=ax)
        for i in range(y.size):
            for j in range(x.size):
                if self.tree[i, j] > 0:
                    ax.plot([x[i], x[j]], [y[i], y[j]], 'g-', alpha=0.5,
                                   lw=1.0)
        plt.show()
        plt.close(fig)

    def eval_psf_rms(self, num_bins=100, b_min=None, b_max=None):
        """Single PSFRMS for the grid"""
        if self.uv_grid is None:
            self.grid_uvw_coords()
        self.psf_rms = (np.sqrt(np.sum(self.uv_grid.real**2)) /
                        (self.uu_m.size * 2))
        return self.psf_rms

    def eval_psf_rms_r(self, num_bins=100, b_min=None, b_max=None):
        """PSFRMS radial profile"""
        if self.uv_grid is None:
            self.grid_uvw_coords()
        grid_size = self.uv_grid.shape[0]
        gx, gy = Imager.grid_pixels(self.grid_cell_size_m, grid_size)
        gr = (gx**2 + gy**2)**0.5

        b_max = self.r_uv_m.max() if b_max is None else b_max
        b_min = self.r_uv_m.min() if b_min is None else b_min
        r_bins = np.logspace(log10(b_min), log10(b_max), num_bins + 1)
        self.psf_rms_r = np.zeros(num_bins)
        for i in range(num_bins):
            pixels = self.uv_grid[np.where(gr <= r_bins[i + 1])]
            uv_idx = np.where(self.r_uv_m <= r_bins[i + 1])[0]
            uv_count = uv_idx.shape[0] * 2
            self.psf_rms_r[i] = 1.0 if uv_count == 0 else \
                np.sqrt(np.sum(pixels.real**2)) / uv_count
        self.psf_rms_r_x = r_bins[1:]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.psf_rms_r_x, self.psf_rms_r)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('radius (m)')
        ax.set_ylabel('PSFRMS')
        ax.set_xlim(10**floor(log10(self.psf_rms_r_x[0])),
                    10**ceil(log10(self.psf_rms_r_x[-1])))
        ax.set_ylim(10**floor(log10(self.psf_rms_r.min())), 1.05)
        ax.grid(True)
        plt.show()

    def uvgap(self):
        pass

    def eval_psf(self, im_size=None, fov_deg=None, plot2d=False, plot1d=True):
        if self.uu_m is None:
            self.gen_uvw_coords()

        # Evaluate grid size and fov if needed.
        if im_size is None:
            b_max = self.r_uv_m.max()
            grid_size = int(ceil(b_max / self.grid_cell_size_m)) * 2 + \
                        self.station_diameter_m
            if grid_size % 2 == 1:
                grid_size += 1
        else:
            grid_size = im_size
        wavelength = const.c.value / self.freq_hz
        if fov_deg is None:
            cellsize_wavelengths = self.grid_cell_size_m / wavelength
            fov_rad = Imager.uv_cellsize_to_fov(cellsize_wavelengths,
                                                grid_size)
            fov_deg = degrees(fov_rad)
        else:
            fov_deg = fov_deg

        uu = self.uu_m / wavelength
        vv = self.vv_m / wavelength
        ww = self.ww_m / wavelength
        amp = np.ones_like(uu, dtype='c8')
        psf = Imager.make_image(uu, vv, ww, np.ones_like(uu, dtype='c8'),
                                fov_deg, grid_size)
        extent = Imager.image_extent_lm(fov_deg, grid_size)
        self.psf = psf
        self.psf_fov_deg = fov_deg

        # --- Plotting ----
        if plot2d:
            fig, ax = plt.subplots(figsize=(8, 8))
            norm = SymLogNorm(linthresh=0.05, linscale=1.0, vmin=-0.05, vmax=0.5,
                              clip=False)
            opts = dict(interpolation='nearest', origin='lower', cmap='gray_r',
                        extent=extent, norm=norm)
            im = ax.imshow(psf, **opts)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.03)
            cbar = ax.figure.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize='small')
            ax.set_xlabel('l')
            ax.set_ylabel('m')
            plt.savefig('psf.png')
            plt.show()
            plt.close(fig)

        if plot1d:
            l, m = Imager.image_pixels(self.psf_fov_deg, grid_size)
            r_lm = (l**2 + m**2)**0.5
            r_lm = r_lm.flatten()
            idx_sorted = np.argsort(r_lm)
            r_lm = r_lm[idx_sorted]
            psf_1d = self.psf.flatten()[idx_sorted]
            psf_hwhm = (wavelength / (r_lm[-1] * 2.0)) / 2

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(r_lm, psf_1d, 'k.', ms=2, alpha=0.1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            fig.savefig('TEST_psf1d.png')
            plt.close(fig)

            num_bins = 100  # FIXME(BM) make this a function arg
            psf_1d_mean = np.zeros(num_bins)
            psf_1d_abs_mean = np.zeros(num_bins)
            psf_1d_abs_max = np.zeros(num_bins)
            psf_1d_min = np.zeros(num_bins)
            psf_1d_max = np.zeros(num_bins)
            psf_1d_std = np.zeros(num_bins)
            bin_edges = np.linspace(r_lm[0], r_lm[-1], num_bins + 1)
            # bin_edges = np.logspace(log10(r_lm[1]), log10(r_lm[-1]),
            #                         num_bins + 1)
            psf_1d_bin_r = (bin_edges[1:] + bin_edges[:-1]) / 2
            bin_idx = np.digitize(r_lm, bin_edges)
            for i in range(1, num_bins + 1):
                values = psf_1d[bin_idx == i]
                if values.size > 0:
                    psf_1d_mean[i - 1] = np.mean(values)
                    psf_1d_abs_mean[i - 1] = np.mean(np.abs(values))
                    psf_1d_abs_max[i - 1] = np.max(np.abs(values))
                    psf_1d_min[i - 1] = np.min(values)
                    psf_1d_max[i - 1] = np.max(values)
                    psf_1d_std[i - 1] = np.std(values)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(psf_1d_bin_r, psf_1d_abs_mean, '-', c='b', lw=1, label='abs mean')
            ax.plot(psf_1d_bin_r, psf_1d_abs_max, '-', c='r', lw=1, label='abs max')
            ax.plot(psf_1d_bin_r, psf_1d_std, '-', c='g', lw=1, label='std')
            # ax.set_ylim(-0.1, 0.5)
            # ax.set_xlim(0, psf_1d_bin_r[-1] / 2**0.5)
            # ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1)
            ax.set_xlim(0, psf_1d_bin_r[-1] / 2**0.5)
            ax.set_xlabel('PSF radius (direction cosines)')
            # ax.set_ylabel('PSF amplitude')
            ax.set_title('Azimuthally averaged PSF (FoV: %.2f)' % fov_deg)
            ax.legend()
            plt.show()
            plt.close(fig)

    def cable_cost(self):
        cluster_cable_length = 0.0
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        spiral_id = 0
        for key in self.layouts.keys():
            print('%-20s' % key, end=' ')
            if key == 'ska1_v5':
                print(': CORE')
                layout = self.layouts[key]
                ax.plot(layout['x'], layout['y'], 'k+')
                ax.add_artist(plt.Circle((0, 0), 500, fill=False, color='r'))
            elif 'log_spiral_section' in key:
                print(': SPIRAL')
                #if spiral_id == 0:
                layout = self.layouts[key]
                cx, cy = (layout['cx'], layout['cy'])
                x, y = (layout['x'], layout['y'])
                ax.plot(x, y, 'b+')
                ax.plot(cx, cy, 'rx', lw=2, ms=5)
                cable_length = 0.0
                for x_, y_ in zip(x, y):
                    ax.plot([x_, cx], [y_, cy], 'k--', alpha=0.5)
                # ax.add_artist(plt.Circle((0, 0), 500, fill=False, color='y'))
                #spiral_id += 1
            else:
                print(': CLUSTER')
                continue
                layout = self.layouts[key]
                cx, cy = (layout['cx'], layout['cy'])
                x, y = (layout['x'], layout['y'])
                ax.plot(x, y, 'b+', mew=1.5)
                ax.plot(cx, cy, 'ro', mew=1.5, mec='None')
                cluster_total = 0.0
                for x_, y_ in zip(x, y):
                    dx = x_ - cx
                    dy = y_ - cy
                    r = (dx**2 + dy**2)**0.5
                    cluster_total += r
                    ax.plot([x_, cx], [y_, cy], 'k--', alpha=0.5)
                ax.text(cx, cy + self.station_diameter_m / 2,
                        '%.1f' % cluster_total, fontsize='small')
                cluster_cable_length += cluster_total
        plt.show()
        plt.close(fig)
        print('cluster cable length:', cluster_cable_length)


class SKA1_low_analysis(TelescopeAnalysis):
    def __init__(self, name=''):
        TelescopeAnalysis.__init__(self, name)
        self.lon_deg = 116.63128900
        self.lat_deg = -26.69702400

