# -*- coding: utf-8 -*-
"""
Module for generating station layouts.

Layout generation functions are expected to return a dictionary describing the
layout.

The following keys are required.
 'x': 1d numpy array of station x (east) coordinates in meters.
 'y': 1d numpy array of station y (north) coordinates in meters.

The following keys are optional.

'color': used to determine the color of stations in plotting
'type': Used to identify the type of layout which can be used for additional
        options or to inform other functions to expect additional type
        specific keys.

Any number of additional keys may be present depending on the type.
"""

from __future__ import absolute_import, division, print_function
import numpy
import sys
from .layout import (rotate_coords,
                     log_spiral_2,
                     log_spiral_clusters,
                     rand_uniform_2d,
                     generate_clusters)


def inner_arms(b, num_arms, n, r_min, r_max):
    """
    Generate inner spiral arms in ENU coordinates

    Args:
        b:
        num_arms:
        n:
        r_min:
        r_max:

    Returns:
    """
    x = numpy.zeros((num_arms, n))
    y = numpy.zeros_like(x)
    for i in range(num_arms):
        x_, y_ = log_spiral_2(r_min, r_max, b, n)
        theta = i * (360.0 / num_arms)
        x[i, :], y[i, :] = rotate_coords(x_, y_, theta)
    return {'x': x.flatten(), 'y': y.flatten()}


def inner_arms_clusters(b, num_arms, clusters_per_arm, stations_per_cluster,
                        cluster_diameter_m, station_diameter_m, r_min, r_max,
                        trail_timeout, tries_per_cluster, seed=None,
                        verbose=False):

    # Generate all the clusters
    num_clusters = clusters_per_arm * num_arms
    min_sep = station_diameter_m
    cluster_r = cluster_diameter_m / 2
    num_trials = tries_per_cluster
    cluster_r_min = 0.0
    x_, y_, info = generate_clusters(num_clusters, stations_per_cluster,
                                     cluster_r, min_sep,
                                     trail_timeout, num_trials,
                                     seed, cluster_r_min, verbose)

    # Generate cluster centres.
    theta_inc = 360.0 / num_arms
    cx, cy = numpy.zeros(num_clusters), numpy.zeros(num_clusters)

    print(r_min, r_max, b, clusters_per_arm)
    cx_, cy_ = log_spiral_2(r_min, r_max, b, clusters_per_arm)
    for i in range(num_arms):
        print(i, theta_inc * i)
        cx_r, cy_r = rotate_coords(cx_, cy_, theta_inc * i)
        cx[i * clusters_per_arm:(i + 1) * clusters_per_arm] = cx_r
        cy[i * clusters_per_arm:(i + 1) * clusters_per_arm] = cy_r

    # Shift clusters to centres
    x = numpy.zeros((num_clusters, stations_per_cluster))
    y = numpy.zeros((num_clusters, stations_per_cluster))
    for i in range(num_clusters):
        x[i, :] = x_[i] + cx[i]
        y[i, :] = y_[i] + cy[i]

    return {'x': x.flatten(), 'y': y.flatten(), 'cx': cx, 'cy': cy}


def inner_arms_rand_uniform(num_stations, station_diameter_m,
                            r_min, r_max, seed=None):
    x, y, _ = rand_uniform_2d(n=num_stations, r_max=r_max,
                              min_sep=station_diameter_m, timeout=10.0,
                              r_min=r_min, seed=seed)
    if not x.shape[0] == num_stations:
        raise RuntimeError('Failed to generate enough stations.')
    return {'x': x, 'y': y}
