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

import numpy

from .layout import (rotate_coords,
                     log_spiral_2,
                     log_spiral_clusters,
                     rand_uniform_2d)


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
    return {'x': x.flatten(), 'y': y.flatten(), 'color': 'k'}


def inner_arms_clusters(b, num_arms, clusters_per_arm, stations_per_cluster,
                        cluster_diameter_m, station_radius_m, r_min, r_max):
    x = numpy.zeros((num_arms, stations_per_cluster * clusters_per_arm))
    y = numpy.zeros_like(x)
    cx = numpy.zeros((num_arms, clusters_per_arm))
    cy = numpy.zeros_like(cx)
    theta_inc = 360.0 / num_arms
    for i in range(num_arms):
        x[i, :], y[i, :], cx[i, :], cy[i, :] = \
            log_spiral_clusters(r_min, r_max, b, clusters_per_arm,
                                stations_per_cluster, cluster_diameter_m / 2.0,
                                station_radius_m * 2.0)
        x[i, :], y[i, :] = rotate_coords(x[i, :], y[i, :], i * theta_inc)
        cx[i, :], cy[i, :] = rotate_coords(cx[i, :], cy[i, :], i * theta_inc)
    return {'type': 'clusters',
            'x': x.flatten(),
            'y': y.flatten(),
            'cluster_x': cx.flatten(),
            'cluster_y': cy.flatten(),
            'cluster_diameter': cluster_diameter_m,
            'color': 'b'
            }


def inner_arms_rand_uniform(num_stations, station_radius_m,
                            r_min, r_max):
    x, y, _ = rand_uniform_2d(num_points=num_stations, r1=r_max,
                              min_sep=station_radius_m*2.0, r0=r_min, max_tries=1000)
    if not x.shape[0] == num_stations:
        raise RuntimeError('Failed to generate enough stations.')
    return {'x': x, 'y': y, 'color': 'g'}
