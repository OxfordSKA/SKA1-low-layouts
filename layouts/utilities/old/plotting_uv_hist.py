# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import time
from math import ceil
from os.path import join

import matplotlib.pyplot as pyplot
import numpy


def plot_uv_hist(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, wave_length, out_dir):
    t0 = time.time()
    v4d_uv_dist = (uu_v4d**2 + vv_v4d**2)**0.5
    v4d_uv_dist.sort()
    v4o1_uv_dist = (uu_v4o1**2 + vv_v4o1**2)**0.5
    v4o1_uv_dist.sort()
    hist_plot_1(v4d_uv_dist, v4o1_uv_dist, wave_length, 800.0, out_dir)
    # hist_plot_1(v4d_uv_dist, v4o1_uv_dist, wave_length * 5.0, 1500.0, out_dir)
    # hist_plot_1(v4d_uv_dist, v4o1_uv_dist, wave_length, 1500.0, out_dir)
    # hist_plot_1(v4d_uv_dist, v4o1_uv_dist, wave_length * 10.0, 3000.0, out_dir)
    # hist_plot_2(v4d_uv_dist, v4o1_uv_dist, wave_length, out_dir)
    # hist_plot_3(v4d_uv_dist, v4o1_uv_dist, wave_length, out_dir)
    print('- histograms took %.2f s' % (time.time() - t0))


def hist_plot_1(v4d_uv_dist, v4o1_uv_dist, bin_width, uv_dist_max,
                out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    v4d_uv_dist = v4d_uv_dist[v4d_uv_dist < uv_dist_max * 2.0]
    v4o1_uv_dist = v4o1_uv_dist[v4o1_uv_dist < uv_dist_max * 2.0]
    num_bins = int(ceil(uv_dist_max / bin_width))
    bins = numpy.arange(num_bins) * bin_width
    v4d_hist, v4d_bin_edges = numpy.histogram(v4d_uv_dist, bins=bins)
    v4o1_hist, v4o1_bin_edges = numpy.histogram(v4o1_uv_dist, bins=bins)
    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1]
    y = v4d_hist
    ax.bar(x, y, width=numpy.diff(v4d_bin_edges), color='0.6', fill=True,
           alpha=0.5, lw=1.5, edgecolor='k', label='v4d')
    x = v4o1_bin_edges[:-1]
    y = v4o1_hist
    ax.bar(x, y, width=numpy.diff(v4o1_bin_edges), color='r', fill=True,
           alpha=0.6, lw=1.0, edgecolor='r', label='v4o1')  # hatch='//'
    ax.set_xlim(0, uv_dist_max)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Number')
    ax.legend(fontsize='small')
    pyplot.savefig(join(out_dir, 'hist_%04.1fm_%06.1fm.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + numpy.diff(v4d_bin_edges) / 2.0
    y = v4d_hist
    ax.plot(x, y, 'k-', lw=1.5, label='v4d')
    x = v4o1_bin_edges[:-1] + numpy.diff(v4o1_bin_edges) / 2.0
    y = v4o1_hist
    ax.plot(x, y, 'r-', lw=1.5, label='v4o1')
    ax.set_xlim(0, uv_dist_max)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Number')
    ax.legend()
    pyplot.savefig(join(out_dir, 'hist_%04.1fm_%06.1fm_v2.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)


def hist_plot_2(v4d_uv_dist, v4o1_uv_dist, wave_length, out_dir):
    bin_width = wave_length
    num_vis = v4d_uv_dist.shape[0]
    assert(v4d_uv_dist.shape[0] == v4o1_uv_dist.shape[0])
    uv_dist_max = 70.0e3
    uv_dist_min = 20.0
    num_bins = int(ceil(uv_dist_max / bin_width))
    bins = numpy.arange(num_bins) * bin_width
    v4d_hist, v4d_bin_edges = numpy.histogram(v4d_uv_dist, bins=bins)
    v4o1_hist, v4o1_bin_edges = numpy.histogram(v4o1_uv_dist, bins=bins)
    v4d_hist = numpy.array(v4d_hist)
    v4o1_hist = numpy.array(v4o1_hist)
    v4d_bin_widths = numpy.diff(v4d_bin_edges)
    v4o1_bin_widths = numpy.diff(v4o1_bin_edges)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.divide(v4d_hist, v4d_bin_widths)
    ax.loglog(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v4o1_bin_edges[:-1] + v4o1_bin_widths / 2.0
    y = numpy.divide(v4o1_hist, v4o1_bin_widths)
    ax.loglog(x, y, 'r-', label='v4o1', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Visibility count (m$^{-1}$)')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'hist_loglog_%04.1fm_%06.1fm.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.cumsum(v4d_hist) / float(num_vis)
    ax.semilogx(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v4o1_bin_edges[:-1] + v4o1_bin_widths / 2.0
    y = numpy.cumsum(v4o1_hist) / float(num_vis)
    ax.semilogx(x, y, 'r-', label='v4o1', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Cumulative visibility density')
    ax.legend(loc=2)
    ax.grid()
    v4d_median = numpy.median(v4d_uv_dist)
    v4o1_median = numpy.median(v4o1_uv_dist)
    ax.plot([v4d_median, v4d_median], ax.get_ylim(), 'k:')
    ax.plot([v4o1_median, v4o1_median], ax.get_ylim(), 'r--')
    ax.text(0.02, 0.73, 'v4d median = %.3f km' % (v4d_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    ax.text(0.02, 0.65, 'v4o1 median = %.3f km' % (v4o1_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    pyplot.savefig(join(out_dir, 'hist_cumsum_%04.1fm_%06.1fm.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)


def hist_plot_3(v4d_uv_dist, v4o1_uv_dist, wave_length, out_dir):
    bin_width = wave_length
    num_vis = v4d_uv_dist.shape[0]
    assert(v4d_uv_dist.shape[0] == v4o1_uv_dist.shape[0])
    uv_dist_max = 70.0e3
    uv_dist_min = 20.0
    num_bins = 200
    bins = numpy.logspace(numpy.log10(uv_dist_min),
                          numpy.log10(uv_dist_max), num_bins)
    v4d_hist, v4d_bin_edges = numpy.histogram(v4d_uv_dist, bins=bins)
    v4o1_hist, v4o1_bin_edges = numpy.histogram(v4o1_uv_dist, bins=bins)
    v4d_hist = numpy.array(v4d_hist)
    v4o1_hist = numpy.array(v4o1_hist)
    v4d_bin_widths = numpy.diff(v4d_bin_edges)
    v4o1_bin_widths = numpy.diff(v4o1_bin_edges)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.divide(v4d_hist, v4d_bin_widths)
    ax.loglog(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v4o1_bin_edges[:-1] + v4o1_bin_widths / 2.0
    y = numpy.divide(v4o1_hist, v4o1_bin_widths)
    ax.loglog(x, y, 'r-', label='v4o1', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Visibility count (m$^{-1}$)')
    ax.legend()
    ax.grid()
    pyplot.savefig(join(out_dir, 'hist_loglog_%04.1fm_%06.1fm_log_bins.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = v4d_bin_edges[:-1] + v4d_bin_widths / 2.0
    y = numpy.cumsum(v4d_hist) / float(num_vis)
    ax.semilogx(x, y, 'k-', label='v4d', lw=1.5, alpha=0.8)
    x = v4o1_bin_edges[:-1] + v4o1_bin_widths / 2.0
    y = numpy.cumsum(v4o1_hist) / float(num_vis)
    ax.semilogx(x, y, 'r-', label='v4o1', lw=1.5, alpha=0.6)
    ax.set_xlim(uv_dist_min, uv_dist_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('uv-distance (m)')
    ax.set_ylabel('Cumulative visibility density')
    ax.legend(loc=2)
    ax.grid()
    v4d_median = numpy.median(v4d_uv_dist)
    v4o1_median = numpy.median(v4o1_uv_dist)
    ax.plot([v4d_median, v4d_median], ax.get_ylim(), 'k:')
    ax.plot([v4o1_median, v4o1_median], ax.get_ylim(), 'r--')
    ax.text(0.02, 0.73, 'v4d median = %.3f km' % (v4d_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    ax.text(0.02, 0.65, 'v4o1 median = %.3f km' % (v4o1_median / 1.0e3),
            ha='left', va='center', style='italic', color='k',
            transform=ax.transAxes)
    pyplot.savefig(join(out_dir, 'hist_cumsum_%04.1fm_%06.1fm_log_bins.png'
                        % (bin_width, uv_dist_max)))
    pyplot.close(fig)
