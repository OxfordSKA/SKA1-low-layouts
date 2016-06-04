# -*- coding: utf-8 -*-

# TODO(BM) single log-spiral rotated
# TODO(BM) tapered random generator
# TODO(BM) add beam convolution
# TODO(BM) Metrics (see apple notes, ALMA paper etc)


from __future__ import print_function

import math
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy
import pyuvwsim

from utilities.analysis import (generate_uv_coords,
                                generate_psf,
                                bin_psf_1d)
from utilities.generators import (inner_arms,
                                  inner_arms_clusters,
                                  inner_arms_rand_uniform)
from utilities.plotting import (plot_layout,
                                plot_uv_scatter,
                                plot_psf,
                                plot_psf_1d_bins,
                                plot_psf_histogram)


def add_observation(obs_length_s, obs_list, duty_cycle_s=3.0*60.0):
    if obs_length_s == 0.0:
        num_times = 1
        interval_s = 0.0
    else:
        num_times = int(obs_length_s / duty_cycle_s)
        interval_s = obs_length_s / float(num_times)
    obs_list.append({'num_times': num_times,
                     'obs_length_s': obs_length_s,
                     'interval_s': interval_s})
    return obs_list


def analyse_layouts(layouts, settings, r_min, r_max):

    # TODO(BM) Define what analysis to do and work out what plots to make
    # - produce separate plots for each layout and result
    # TODO(BM) loop over layouts and add curves to plots.

    fontsize='small'

    fig_layout_compare = plt.figure(figsize=(8, 8))
    ax_layout_compare = fig_layout_compare.add_subplot(111, aspect='equal')

    for layout_name in layouts:
        fig = plt.figure(figsize=(8, 8))
        plot_layout(fig.add_subplot(111, aspect='equal'),
                    layouts[layout_name], settings, r_min, r_max, fontsize)
        fig.savefig(join(settings['results_dir'], '%s_coords.png' % layout_name))
        plt.close(fig)
        plot_layout(ax_layout_compare, layouts[layout_name], settings, r_min,
                    r_max, fontsize)

    fig_layout_compare.savefig(join(settings['results_dir'],
                                    'layout_compare.png'))
    plt.close(fig_layout_compare)



def analyse_layout(layout, settings):
    observations = list()
    add_observation(0.0, observations)  # snapshot
    # add_observation(4.0 * 3600.0, observations)  # 4h observation
    # Combine components to a single set of station coordinates.
    x, y = list(), list()
    for coords in layout:
        x.extend(coords['x'])
        y.extend(coords['y'])
    x, y, z = numpy.array(x), numpy.array(y), numpy.zeros_like(x)
    ecef_x, ecef_y, ecef_z = pyuvwsim.convert_enu_to_ecef(x, y, z,
                                                          settings['lon'],
                                                          settings['lat'])
    # Generate results
    results = list()
    for i_obs, obs in enumerate(observations):
        t0 = time.time()
        print('- Analysing observation %i, no. times = %i ... ' %
              (i_obs, obs['num_times']), end=' ')

        # Generate uvw coordinates (in wavelengths).
        uu, vv, ww = generate_uv_coords(ecef_x, ecef_y, ecef_z, settings, obs)

        # # Generate uv grid image
        # t1 = time.time()
        # uv_grid, uv_cell_size_wavelengths, fov_deg = \
        #     generate_uv_grid(uu, vv, ww, x, y, settings)
        # print('uv_grid: %.2fs,' % (time.time() - t1), end=' ')

        # TODO(BM) convolve with beam (use gaussian or lookup table)

        # Make the PSF
        t1 = time.time()
        psf, psf_cell_size, psf_az_1d, r_lm = generate_psf(uu, vv, ww, settings)
        print('psf: %.2fs,' % (time.time() - t1), end=' ')

        # Bin the 1d psf
        t1 = time.time()
        psf_bins = bin_psf_1d(psf, psf_cell_size, r_lm, psf_az_1d, num_bins=200)
        print('bin_psf: %.2fs,' % (time.time() - t1), end=' ')

        # TODO(BM) Metrics UVGAP, PSFRMS etc

        results.append({
            'num_antennas': x.shape[0],
            'uu': uu,
            'vv': vv,
            # 'uv_grid': uv_grid,
            # 'uv_grid_fov': fov_deg,
            # 'uv_grid_cell_size_wavelengths': uv_cell_size_wavelengths,
            'psf': psf,
            'psf_r_lm': r_lm,
            'psf_az_1d': psf_az_1d,
            'psf_cell_size': psf_cell_size,
            'psf_bins': psf_bins,
            'obs': obs,
            'settings': settings
        })
        print('total: %.2f s' % (time.time() - t0))

    return results


def plotting(layout, results, settings, save_root_name, r_min, r_max):
    # FIXME(BM) generate multi page plot (or multiple plots)

    t0 = time.time()
    nx, ny = 3, 4
    #
    # 1   2  3
    # 4   5  6
    # 7   8  9
    # 10 11 12
    #

    vmin_0h = -0.05
    linthresh_0h = 0.3

    vmin_4h = -0.05
    linthresh_4h = 0.3

    fontsize_ = 'small'
    # fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig = plt.figure(figsize=(11.69, 16.53))  # A3

    fig.subplots_adjust(left=0.1, bottom=0.08, right=0.95, top=0.92,
                        hspace=0.2, wspace=0.2)

    # ------ Stations --------------
    plot_layout(fig.add_subplot(ny, nx, 1, aspect='equal'), layout,
                settings, r_min, r_max, fontsize_)

    # UV scatter
    plot_uv_scatter(fig.add_subplot(ny, nx, 2, aspect='equal'),
                    results[0], settings, r_min, r_max, fontsize_)
    # plot_uv_scatter(fig.add_subplot(ny, nx, 3, aspect='equal'),
    #                 results[1], settings, r_min, r_max, fontsize_)

    nx, ny = 2, 4
    # 1   2
    # 3   4
    # 5   6
    # 7   8


    # UV histogram
    # plot_uv_hist(fig.add_subplot(ny, nx, 3), results[0], settings, r_min, r_max,
    #              fontsize_)
    # plot_uv_hist(fig.add_subplot(ny, nx, 4), results[1], settings, r_min, r_max,
    #              fontsize_)

    # plot_uv_hist_2(fig.add_subplot(ny, nx, 3), results[0], settings, r_min, r_max,
    #              fontsize_)
    # plot_uv_hist_2(fig.add_subplot(ny, nx, 4), results[1], settings, r_min, r_max,
    #              fontsize_)

    # 2D PSF
    vmin_psf_im = -0.03
    linthresh_psf_im = 0.05
    plot_psf(fig.add_subplot(ny, nx, 3), results[0], settings, fontsize_,
             vmin_psf_im, linthresh_psf_im)
    # plot_psf(fig.add_subplot(ny, nx, 8), results[1], settings, fontsize_,
    #          vmin_psf_im, linthresh_psf_im)


    # TODO(BM) UV density histogram (see alma memo 390)

    # nbins = 500
    # plot_psf_1d_2(fig.add_subplot(ny, nx, 3), results[0], settings, nbins,
    #               fontsize_, vmin_0h, linthresh_0h)
    # plot_psf_1d_2(fig.add_subplot(ny, nx, 4), results[1], settings, nbins,
    #               fontsize_, vmin_0h, linthresh_0h)

    # TODO(BM) PSF peak average and RMS plot (see alma memo 390)


    # # 1D PSF
    # plot_psf_1d(fig.add_subplot(ny, nx, 11), results[0], settings, fontsize_,
    #             vmin, linthresh)
    # plot_psf_1d(fig.add_subplot(ny, nx, 12), results[1], settings, fontsize_,
    #             vmin, linthresh)

    # 1D PSF (bins)
    # http://stackoverflow.com/questions/15556930/turn-scatter-data-into-binned-data-with-errors-bars-equal-to-standard-deviation
    plot_psf_1d_bins(fig.add_subplot(ny, nx, 5), results[0], settings,
                     fontsize_, vmin_0h, linthresh_0h)
    # plot_psf_1d_bins(fig.add_subplot(ny, nx, 6), results[1], settings,
    #                  fontsize_, vmin_4h, linthresh_4h)


    # PSF histogram
    plot_psf_histogram(fig.add_subplot(ny, nx, 7), results[0], settings,
                       fontsize_, y_max=0.5)
    # plot_psf_histogram(fig.add_subplot(ny, nx, 8), results[1], settings,
    #                    fontsize_, y_max=0.5)

    plt.savefig(join(settings['results_dir'], '%s.png' % (save_root_name)))
    plt.close(fig)
    print('- Plotting took %.2f s' % (time.time() - t0))


def main():
    settings = {
        'station_r': 45.0 / 2.0,
        'lon': math.radians(116.63128900),
        'lat': math.radians(-26.69702400),
        'ra': math.radians(68.698903779331502),
        'dec': math.radians(-26.568851215532160),
        'mjd_mid': 57443.4375000000,
        'freq_hz': 200e6,
        'uv_grid_size': 64,
        'psf_im_size': 8192,
        'psf_fov_deg': 180.0,
        'results_dir': 'results'
    }
    r_min = 500.0
    r_max = 1700.0

    # FIXME(BM) multiple PSF FoV's
    # FIXME(BM) think about PSF FoV vs frequency .... (cant just use lambda=1m!)
    # FIXME(BM) think about how to modularise analysis and plotting
    #           so that analysis is not run when plots are not used.
    #           ---> combine plotting and analysis?

    layouts = dict()

    # Single arms
    b, num_arms, n = 0.5, 6, 12
    layouts['log_spiral'] = inner_arms(b, num_arms, n, r_min, r_max)

    # Cluster arms
    b, num_arms, clusters_per_arm, stations_per_cluster = 0.5, 3, 4, 6
    cluster_diameter_m, station_radius_m = 200, settings['station_r']
    layouts['clusters'] = inner_arms_clusters(b, num_arms, clusters_per_arm,
                                             stations_per_cluster,
                                             cluster_diameter_m,
                                             station_radius_m,
                                             r_min, r_max)

    # Random uniform
    num_stations = layouts['clusters']['x'].shape[0]
    station_radius_m = settings['station_r']
    layouts['random_uniform'] = inner_arms_rand_uniform(num_stations,
                                                        station_radius_m,
                                                        r_min, r_max)


    results = analyse_layouts(layouts, settings, r_min, r_max)



    return
    plotting(layout, results, settings, 'spiral_b%.1f_%02ix%02i'
             % (b, num_arms, n), r_min, r_max)

    # Clusters
    layout = inner_arms_clusters(b, num_arms, clusters_per_arm,
                                 stations_per_cluster,
                                 cluster_diameter_m, station_radius_m,
                                 r_min, r_max, layout)
    results = analyse_layout(layout, settings)
    plotting(layout, results, settings,
             'spiral_clusters_b%.1f_%02ix%02ix%02i' %
             (b, num_arms, clusters_per_arm, stations_per_cluster))



    # Random uniform
    num_stations = 72
    station_radius_m = settings['station_r']
    for iter in range(1):
        seed = numpy.random.randint(1, 20000)
        numpy.random.seed(seed)
        layout = inner_arms_rand_uniform(num_stations, station_radius_m, r_min,
                                         r_max)
        results = analyse_layout(layout, settings)
        plotting(layout, results, settings,
                 'random_uniform_%02i_%i' % (num_stations, iter))

    # TODO(BM) Consider comparing multiple layouts together
    # (one or more obs lengths)?

    # LOOK AT THE PSF METRIC FOR THE CORE REGION ONLY AS THIS WILL BE THE PSF
        # USED AFTER THE LAMBDA CUT?

if __name__ == '__main__':
    main()
