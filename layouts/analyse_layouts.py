# -*- coding: utf-8 -*-

# TODO(BM) single log-spiral rotated
# TODO(BM) tapered random generator
# TODO(BM) add beam convolution
# TODO(BM) Metrics (see apple notes, ALMA paper etc)
# TODO(BM) add path length code
# LOOK AT THE PSF METRIC FOR THE CORE REGION ONLY AS THIS WILL BE THE PSF
# USED AFTER THE LAMBDA CUT?


from __future__ import division, absolute_import, print_function

import math
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

from utilities.analysis import (generate_baseline_coords,
                                generate_uv_grid_2,
                                generate_psf_2,
                                get_psf_coords)
from utilities.plotting import (plot_layout,
                                plot_uv_scatter_2,
                                plot_uv_scatter_3,
                                plot_uv_density,
                                plot_uv_grid,
                                plot_psf_2,
                                plot_psf_1d_new,
                                plot_psf_histogram_2)


def fig_save_and_clear(fig, file_name):
    fig.savefig(file_name)
    fig.clear()


def eval_uv_gap(uu, vv, q_max):
    q_max = 300.0e3
    uu_ = numpy.copy(uu)
    vv_ = numpy.copy(vv)
    uu_ = numpy.hstack((uu_, -uu_))
    vv_ = numpy.hstack((vv_, -vv_))
    q = (uu_**2 + vv_**2)**0.5
    theta = numpy.arctan2(vv_, uu_) + math.pi
    # theta[theta < 0.0] += math.pi
    keep_idx = numpy.where(q <= q_max)
    q = q[keep_idx]
    theta = theta[keep_idx]

    # Bin in theta
    num_bins = 50
    theta_edges = numpy.linspace(0, 2.0 * math.pi, num_bins + 1)
    q_edges = numpy.linspace(0, q_max + 1.0e-10, num_bins + 1)
    theta_idx = numpy.digitize(theta, theta_edges)

    uv_gap_im = numpy.zeros((num_bins, num_bins))
    uv_gap_2 = numpy.zeros((num_bins))
    for j in range(num_bins):  # loop over theta bins

        # Get q values in the theta bin
        theta_bin = theta[theta_idx == 1 + j]
        q_bin = q[theta_idx == 1 + j]

        # Add zero and q_max points.
        t = theta_edges[j] + (theta_edges[j + 1] - theta_edges[j]) / 2.0
        theta_bin = numpy.append(theta_bin, [t, t])
        q_bin = numpy.append(q_bin, [0.0, q_max])
        q_bin = numpy.append(q_bin, [0.0])

        # Sort into q order
        q_idx = numpy.argsort(q_bin)
        q_bin = q_bin[q_idx]
        theta_bin = theta_bin[q_idx]

        q_diff = numpy.diff(q_bin) / q_bin[1:]
        q_bin_diff_idx = numpy.digitize(q_bin[1:], q_edges)
        q_bin_idx = numpy.digitize(q_bin, q_edges)
        print(j, q_bin[0], q_diff[0])

        for i in range(num_bins):
            bin_diffs = q_diff[q_bin_diff_idx == 1 + i]
            x = q_bin[q_bin_idx == 1 + i]
            y = theta_bin[q_bin_idx == 1 + i]
            count = x.shape[0]
            diff_count = bin_diffs.shape[0]
            if j == 0 and i < 5:
                print('%i, %i: %i values, %i diffs' % (j, i, count, bin_diffs.shape[0]))
                print('  values:', x)
                print('  diffs:', bin_diffs)
            # uv_gap_im[j, i] = numpy.nan if diff_count == 0 else numpy.mean(bin_diffs)
            uv_gap_im[i, j] = 1.0 if diff_count == 0 else numpy.mean(bin_diffs)


    # Azimuthal averages of delta_u (to plot radial profile)
    uv_gap_r = numpy.zeros((2, num_bins))
    for i in range(num_bins):
        uv_gap_r[0, i] = q_edges[i] + (q_edges[i+1] - q_edges[i]) / 2
        # uv_gap_r[1, i] = numpy.nanmean(uv_gap_im[:, i])
        uv_gap_r[1, i] = numpy.nanmin(uv_gap_im[i, :])

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(121)
    cmap = plt.cm.get_cmap('gray', 2048)
    extent = [0, 2.0 * math.pi, 0/1.0e3, q_max/1.0e3]
    im = ax.imshow(uv_gap_im, interpolation='nearest', cmap=cmap,
                   alpha=0.8, origin='lower', extent=extent, norm=LogNorm())
    ax.figure.colorbar(im, ax=ax)
    ax.plot(theta, q/1.0e3, 'r+', ms=8.0, alpha=0.7)
    ax.set_aspect('auto')
    ax.set_ylim(-(q_max / 20) / 1.0e3, (q_max + q_max / 20) / 1.0e3)
    ax.set_xlim(0.0 - (2.0 * math.pi) / 16,  (2.0 * math.pi) +  (2.0 * math.pi) / 16)

    ax = fig.add_subplot(122)
    ax.plot(uv_gap_r[0, :] / 1.0e3, uv_gap_r[1, :], '+', alpha=1.0)
    ax.grid(True)
    # ax.set_ylim(0.0001, 0.5)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    # FIXME(BM) respect output directory name
    # plt.show()
    plt.savefig(join('results', 'uv_gap', 'foo.png'))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar='True')
    # used_theta = theta_edges[:-1] + numpy.diff(theta_edges)
    # used_rad = q_edges[:-1] + numpy.diff(q_edges)
    # used_rad /= 1.0e3
    used_theta = theta_edges[:-1]
    used_rad = q_edges[:-1]
    used_rad /= 1.0e3
    data = uv_gap_im
    p_theta, p_rad = numpy.meshgrid(used_theta, used_rad)
    im = ax.pcolormesh(p_theta, p_rad, data, cmap='gray', norm=LogNorm())
    ax.figure.colorbar(im, ax=ax)
    ax.plot(theta, q/1.0e3, 'g.', ms=5.0, alpha=0.5)
    plt.savefig(join('results', 'uv_gap', 'foo_p.png'))
    plt.close(fig)


    # FIXME(BM) run the same config though iantconfig to get its values of uvgap?

    # # bin in theta and evaluate uv-gap (use scipy custom function method?)
    # # scipy.stats.binned_statistic
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.plot(q, theta, '.', color='b', ms=3.0, alpha=0.5)
    # ax.set_ylim(0, math.pi)
    # for edge in theta_edges:
    #     ax.plot(ax.set_xlim(), [edge, edge], 'c-', alpha=0.4)
    # ax.plot(q[theta_idx == 1], theta[theta_idx == 1], 'x', color='g', alpha=0.5)
    # ax.plot(q[theta_idx == 2], theta[theta_idx == 2], 'x', color='r', alpha=0.5)
    #
    # plt.show()
    # plt.close(fig)


def analyse_layouts(layouts, settings, r_min, r_max):
    """Analyse station layouts"""
    # TODO(BM) Define what analysis to do and work out what plots to make
    # - produce separate plots for each layout and result
    # TODO(BM) loop over layouts and add curves to plots.
    font_size = 'small'
    layout_colors = ['b', 'r', 'g']
    fig = plt.figure(figsize=(8, 8))

    # outputs = ['layout', 'uv_dist', 'psf']
    # outputs = ['layout', 'uv_dist', 'uv_grid']
    # outputs = ['layout', 'uv_grid']
    outputs = ['layout', 'uv_dist', 'uv_gap']
    for output in outputs:
        if not os.path.isdir(join(settings['results_dir'], output)):
            os.makedirs(join(settings['results_dir'], output))
    if 'psf' in outputs:
        for sub_dir in ('1d', '2d'):
            if not os.path.isdir(join(settings['results_dir'], 'psf', sub_dir)):
                os.makedirs(join(settings['results_dir'], 'psf', sub_dir))

    wavelength_m = 299792458.0 / settings['freq_hz']
    psf_fwhm = wavelength_m / (r_max * 2.0)
    mag = settings['psf_fov_deg'] / math.degrees(psf_fwhm)

    if 'layout' in outputs:
        fig_layout_compare = plt.figure(figsize=(8, 8))
        ax_layout_compare = fig_layout_compare.add_subplot(111, aspect='equal')

    if 'uv_dist' in outputs:
        fig_uv_density_compare = plt.figure(figsize=(8, 8))
        ax_uv_density_compare = fig_uv_density_compare.add_subplot(111)

    if 'psf' in outputs:
        fig_psf_1d_compare = plt.figure(figsize=(8, 8))
        ax_psf_1d_compare = fig_psf_1d_compare.add_subplot(111)
        fig_psf_hist_compare = plt.figure(figsize=(8, 8))
        ax_psf_hist_compare = fig_psf_hist_compare.add_subplot(111)

    for i_layout, layout_name in enumerate(layouts):
        layout = layouts[layout_name]
        if 'uv_dist' or 'psf' or 'uv_grid' or 'uv_gap' in outputs:
            uu, vv, ww = generate_baseline_coords(layout, settings)
        print('')
        print('Analysing layout %02i: %s' % (i_layout, layout_name))
        print('- No. antennas =', layout['x'].shape[0])

        # Layout plot(s) ------------------------------------------------------
        if 'layout' in outputs:
            plot_layout(fig.add_subplot(111, aspect='equal'), layout, settings, r_min, r_max, font_size)
            fig_save_and_clear(fig, join(settings['results_dir'], 'layout', '%02i_%s.png' % (i_layout, layout_name)))
            plot_layout(ax_layout_compare, layout, settings, r_min, r_max, font_size)

        # Generate and plot uv coordinates ------------------------------------
        if 'uv_dist' in outputs:
            plot_uv_scatter_2(fig.add_subplot(111, aspect='equal'), uu, vv, r_max, settings, font_size)
            fig_save_and_clear(fig, join(settings['results_dir'], 'uv_dist', '%02i_%s_uv_scatter.png' % (i_layout, layout_name)))
            plot_uv_scatter_3(fig.add_subplot(111, aspect='equal'), uu, vv, r_max, settings, font_size)
            fig_save_and_clear(fig, join(settings['results_dir'], 'uv_dist', '%02i_%s_uv_scatter_wavelengths.png' % (i_layout, layout_name)))

            plot_uv_density(ax_uv_density_compare, uu, vv, wavelength_m, r_min, r_max, layout_colors[i_layout], layout_name, font_size)

        if 'uv_grid' in outputs:
            uv_grid, uv_cell_size_wavelengths, fov_deg = generate_uv_grid_2(uu, vv, ww, r_max, settings)
            plot_uv_grid(fig.add_subplot(111), uv_grid, uv_cell_size_wavelengths, wavelength_m, r_max)
            fig_save_and_clear(fig, join(settings['results_dir'], 'uv_grid', '%02i_%s_uv_grid_2d.png' % (i_layout, layout_name)))

        if 'uv_gap' in outputs:
            eval_uv_gap(uu, vv, r_max)
            # Histogram of the distance matrix?

        # Generate and plot PSF
        if 'psf' in outputs:
            psf = generate_psf_2(uu, vv, ww, settings)
            l, m, r_lm = get_psf_coords(settings)
            plot_psf_2(fig.add_subplot(111), psf, settings)
            fig_save_and_clear(fig, join(settings['results_dir'], 'psf', '2d', '%02i_%s.png' % (i_layout, layout_name)))
            plot_psf_1d_new(fig.add_subplot(111), psf, r_lm, settings, r_max)
            fig_save_and_clear(fig, join(settings['results_dir'], 'psf', '1d', '%02i_%s.png' % (i_layout, layout_name)))
            plot_psf_1d_new(ax_psf_1d_compare, psf, r_lm, settings, r_max, layout_colors[i_layout], layout_name)
            plot_psf_histogram_2(ax_psf_hist_compare, psf, layout_colors[i_layout], font_size, layout_name,  layout['x'].shape[0], mag)

    if 'layout' in outputs:
        fig_layout_compare.savefig(join(settings['results_dir'], 'layout', 'layout_compare.png'))
        plt.close(fig_layout_compare)

    if 'uv_dist' in outputs:
        ax_uv_density_compare.legend()
        fig_uv_density_compare.savefig(join(settings['results_dir'], 'uv_dist', 'uv_density_compare.png'))
        plt.close(fig_uv_density_compare)

    if 'psf' in outputs:
        ax_psf_1d_compare.legend()
        ax_psf_1d_compare.set_ylim(-0.1, 0.5)
        ax_psf_1d_compare.set_xlim(1.0, ax_psf_1d_compare.get_xlim()[1])
        ax_psf_1d_compare.set_xscale('log')
        fig_psf_1d_compare.savefig(join(settings['results_dir'], 'psf', 'psf_1d_compare.png'))
        plt.close(fig_psf_1d_compare)
        ax_psf_hist_compare.legend()
        fig_psf_hist_compare.savefig(join(settings['results_dir'], 'psf', 'psf_hist_compare.png'))
        plt.close(fig_psf_hist_compare)


def main():
    settings = {
        'station_r': 45.0 / 2.0,
        'lon': math.radians(116.63128900),
        'lat': math.radians(-26.69702400),
        'ra': math.radians(68.698903779331502),
        'dec': math.radians(-26.568851215532160),
        'mjd_mid': 57443.4375000000,
        'num_times': 1,
        'obs_length_s': 0.0,
        'interval_s': 0.0,
        'freq_hz': 8.46e9,
        'psf_im_size': 2048,
        'results_dir': 'results'
    }

    if not os.path.isdir(settings['results_dir']):
        os.makedirs(settings['results_dir'])

    wavelength_m = 299792458.0 / settings['freq_hz']
    pb_fwhm = math.degrees(wavelength_m / (settings['station_r'] * 2.0))
    settings['psf_fov_deg'] = 3.0 * pb_fwhm
    r_min = 500.0
    r_max = 1700.0
    psf_fwhm = wavelength_m / (r_max * 2.0)
    print(settings['results_dir'])
    print('psf_fwhm', psf_fwhm)
    print('psf_fov_deg', settings['psf_fov_deg'])
    print('mag: %f' % (settings['psf_fov_deg'] / math.degrees(psf_fwhm)))

    # FIXME(BM) multiple PSF FoV's
    # FIXME(BM) think about PSF FoV vs frequency .... (cant just use lambda=1m!)
    # FIXME(BM) think about how to modularise analysis and plotting
    #           so that analysis is not run when plots are not used.
    #           ---> combine plotting and analysis?

    layouts = dict()

    # Single arms
    # b, num_arms, n = 0.5, 6, 12
    # layouts['log_spiral'] = inner_arms(b, num_arms, n, r_min, r_max)

    # # Cluster arms
    # b, num_arms, clusters_per_arm, stations_per_cluster = 0.5, 3, 4, 6
    # cluster_diameter_m, station_radius_m = 200, settings['station_r']
    # layouts['clusters'] = inner_arms_clusters(b, num_arms, clusters_per_arm,
    #                                           stations_per_cluster,
    #                                           cluster_diameter_m,
    #                                           station_radius_m,
    #                                           r_min, r_max)
    #
    # Random uniform
    # num_stations = layouts['clusters']['x'].shape[0]
    # num_stations = 72
    # station_radius_m = settings['station_r']
    # layouts['random_uniform'] = inner_arms_rand_uniform(num_stations,
    #                                                     station_radius_m,
    #                                                     r_min, r_max)

    coords = numpy.loadtxt(join('models', 'vlaB.enu.27x3.txt'))
    layouts['vla_b'] = {'x': coords[:-1, 1], 'y': coords[:-1, 2]}
    r = (coords[:-1, 1]**2 + coords[:-1, 2]**2)**0.5
    r_min = r.min()
    r_max = r.max()

    analyse_layouts(layouts, settings, r_min, r_max)


if __name__ == '__main__':
    main()
