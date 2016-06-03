# -*- coding: utf-8 -*-
"""Script to generate v5 comparison station coordinates.

Changes:
    26/05/2016: Initial version.
"""
from __future__ import print_function
import math
import os
import sys
from os.path import join
import numpy
import shutil
import matplotlib.pyplot as pyplot
import utilities.layout_utils as utils
import tsp


# TODO(BM) this wont work as need an overlap constraint
#          this could be done by rejecting close antennas or using a normal
#          distribution rejection distance based on a growing radius
#          as for the taylor window.
def gaussian():
    maximum_baseline_m = 200.0e3
    mean = [0.0, 0.0]
    rho = 0.0  # correlation between x and y
    sigma_x = 1.0
    sigma_y = 1.0
    cov = [[sigma_x**2, rho * sigma_x * sigma_y],
           [rho * sigma_y * sigma_x, sigma_y**2]]
    num_points = 50
    samples = numpy.random.multivariate_normal(mean, cov, num_points)

    print(samples.shape)
    points = list()
    for s in samples:
        points.append(tuple(s))
    t = tsp.tsp(points)
    t.solve(keep_files=True)


    # Plotting
    xy_extent = numpy.abs(samples).max()
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(samples[:, 0], samples[:, 1], '.')
    circle = pyplot.Circle((0, 0), xy_extent, color='r', fill=False,
                           alpha=1.0, linewidth=1.0, linestyle='--')
    ax.add_artist(circle)
    ax.grid(True)
    ax.set_xlim(-xy_extent * 1.1, xy_extent * 1.1)
    ax.set_ylim(-xy_extent * 1.1, xy_extent * 1.1)
    pyplot.show()


def test1():
    n = 512
    b_max = 100e3
    station_d = 35.0
    x, y, _ = utils.gridgen(n, b_max * 2.0, station_d, max_trials=1000)

    # Plotting
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, y, '.')
    circle = pyplot.Circle((0, 0), b_max, color='r', fill=False,
                           alpha=1.0, linewidth=1.0, linestyle='--')
    ax.add_artist(circle)
    ax.grid(True)
    ax.set_xlim(-b_max * 1.1, b_max * 1.1)
    ax.set_ylim(-b_max * 1.1, b_max * 1.1)
    pyplot.show()


def test2():
    n = 512
    b_max = 100e3
    station_d = 100.0
    sll = -50
    x, y, _ = utils.gridgen_taylor(n, b_max * 2.0, station_d, sll,
                                   n_miss_max=1000)

    # Plotting
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, y, '.')
    circle = pyplot.Circle((0, 0), b_max, color='r', fill=False,
                           alpha=1.0, linewidth=1.0, linestyle='--')
    ax.add_artist(circle)
    ax.grid(True)
    ax.set_xlim(-b_max * 1.1, b_max * 1.1)
    ax.set_ylim(-b_max * 1.1, b_max * 1.1)
    pyplot.show()


def test3():
    # http://uk.mathworks.com/matlabcentral/fileexchange/35797-generate-random-numbers-from-a-2d-discrete-distribution
    # https://en.wikipedia.org/wiki/Inverse_transform_sampling
    # https://en.wikipedia.org/wiki/Rejection_sampling
    # http://python-for-signal-processing.blogspot.co.uk/2014/02/methods-of-random-sampling-using.html
    # http://stackoverflow.com/questions/25911552/sampling-from-a-multivariate-pdf-in-python
    # https://jyyuan.wordpress.com/2014/04/06/generating-random-samples-pt-2-rejection-sampling/
    # https://gist.github.com/rsnemmen/d1c4322d2bc3d6e36be8


    # https://en.wikipedia.org/wiki/Beta_distribution

    # TODO(BM)
    # Plot tapered random placement with restriction radius around points
    # as we have done before to see how the algorithm work.
    # Then plot radial distribution to test if this method matches
    # that of a 2d gaussian if the restriction growth is gaussian.

    # TODO(BM)
    # Otherwise, generate a uniform dense distribution and thin
    # to a given radial profile (in rings).
    pass


def test4():
    # http://uk.mathworks.com/help/stats/lognpdf.html
    x = numpy.random.lognormal(0, 2.0, 512)

    # Plotting
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, '.')
    ax.grid(True)
    pyplot.show()

if __name__ == '__main__':
    # gaussian()
    test4()
