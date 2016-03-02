# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as pyplot


def plot_hist(uu, vv, file_name, title):
    uv_dist = (uu**2 + vv**2)**0.5
    uv_dist_range = uv_dist.max() - uv_dist.min()
    bin_width_m = 10.0
    num_bins = numpy.ceil(uv_dist_range / bin_width_m)
    bin_inc = uv_dist_range / num_bins
    bins = numpy.arange(num_bins) * bin_inc + uv_dist.min()
    hist, bin_edges = numpy.histogram(uv_dist, bins=bins, density=False)
    x = bin_edges[:-1] + numpy.diff(bin_edges) / 2.0
    fig = pyplot.figure(figsize=(11, 6))
    ax = fig.add_subplot(121)
    ax.grid()
    ax.loglog(x, hist, '-')
    ax.set_xlim(20, 2.0e5)
    ax.set_ylim(0, 2.0e3)
    ax.set_xlabel('Radius (m)', fontsize='small')
    ax.set_ylabel('Relative visibility density', fontsize='small')
    ax.set_title(title)

    ax = fig.add_subplot(122)
    uv_dist_range = uv_dist.max() - uv_dist.min()
    bin_width_m = 10.0
    num_bins = numpy.ceil(uv_dist_range / bin_width_m)
    bin_inc = uv_dist_range / num_bins
    bins = numpy.arange(num_bins) * bin_inc + uv_dist.min()
    hist, bin_edges = numpy.histogram(uv_dist, bins=bins, density=True)
    hist *= bin_width_m
    cum_hist = numpy.cumsum(hist)
    x = bin_edges[:-1] + numpy.diff(bin_edges) / 2.0
    ax.semilogx(x, cum_hist, '-')
    ax.grid()
    ax.set_xlim(20, 2.0e5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Radius (m)', fontsize='small')
    ax.set_ylabel('Cumulative visibility number', fontsize='small')
    ax.set_title(title)
    pyplot.savefig(file_name)
    pyplot.close(fig)


def plot_uv_dist(uu, vv, file_root):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu, vv, 'k.', alpha=0.3, ms=2.0)
    ax.plot(-uu, -vv, 'k.', alpha=0.3, ms=2.0)
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(file_root + '_03km.png')
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    pyplot.savefig(file_root + '_01km.png')
    ax.set_xlim(-50000, 50000)
    ax.set_ylim(-50000, 50000)
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    pyplot.savefig(file_root + '_50km.png')
    pyplot.close(fig)
