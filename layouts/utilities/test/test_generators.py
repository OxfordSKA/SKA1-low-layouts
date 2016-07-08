# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from utilities.layout import (rand_uniform_2d,
                              rand_uniform_2d_trials,
                              rand_uniform_2d_trials_r_max,
                              generate_clusters)
from utilities.generators import inner_arms_clusters
import unittest
import matplotlib.pyplot as plt
import numpy as np
import time


# class Test1(unittest.TestCase):
#     @staticmethod
#     def runTest():
#         n = 6
#         r_min, r_max = 0.0, 18.0
#         min_sep = 10.0
#         timeout = 10.0
#         t0 = time.time()
#         x, y, info = rand_uniform_2d(n, r_max, min_sep, timeout, r_min)
#         print('Time taken = %.2f s' % (time.time() - t0))
#
#         print(x.shape)
#         print(info)
#         fig, ax = plt.subplots(1, 1)
#         ax.set_aspect('equal')
#         for x_, y_, in zip(x, y):
#             c = plt.Circle((x_, y_), min_sep / 2, fill=True, color='k',
#                            alpha=0.5, lw=0)
#             ax.add_artist(c)
#         c = plt.Circle((0, 0), r_max, fill=False, color='r', alpha=0.5)
#         ax.add_artist(c)
#         c = plt.Circle((0, 0), r_min, fill=False, color='r', alpha=0.5)
#         ax.add_artist(c)
#         ax.set_xlim(-r_max * 1.05, r_max * 1.05)
#         ax.set_ylim(-r_max * 1.05, r_max * 1.05)
#         plt.show()


# class Test2(unittest.TestCase):
#     @staticmethod
#     def runTest():
#         # Quicker to do more tries with small timeout or visa versa?
#         # - probably more tries...?
#         n = 6
#         r_min, r_max = 0.0, 17.0
#         min_sep = 10.0
#         trial_timeout = 1.0
#         num_trials = 15
#         seed = None
#         verbose = True
#         x, y, info = rand_uniform_2d_trials(n, r_max, min_sep, trial_timeout,
#                                             num_trials, seed, r_min, verbose)
#         for key in info:
#             print(key, ':',  info[key])
#         fig, ax = plt.subplots(1, 1)
#         ax.set_aspect('equal')
#         for x_, y_, in zip(x, y):
#             c = plt.Circle((x_, y_), min_sep / 2, fill=True, color='k',
#                            alpha=0.5, lw=0)
#             ax.add_artist(c)
#         c = plt.Circle((0, 0), r_max, fill=False, color='r', alpha=0.5)
#         ax.add_artist(c)
#         c = plt.Circle((0, 0), r_min, fill=False, color='r', alpha=0.5)
#         ax.add_artist(c)
#         ax.set_xlim(-r_max * 1.05, r_max * 1.05)
#         ax.set_ylim(-r_max * 1.05, r_max * 1.05)
#         plt.show()

# class Test3(unittest.TestCase):
#     @staticmethod
#     def runTest():
#         # Quicker to do more tries with small timeout or visa versa?
#         # - probably more tries...?
#         n = 256
#         r_min, r_max = 0.0, np.linspace(35/2, 45/2, 20)
#         min_sep = 1.5
#         trial_timeout = 0.5
#         num_trials = 5
#         seed = None
#         verbose = False
#         t0 = time.time()
#         x, y, info = rand_uniform_2d_trials_r_max(n, r_max, min_sep,
#                                                   trial_timeout,
#                                                   num_trials, seed, r_min,
#                                                   verbose)
#         print('time taken = %0.2f s' % (time.time() - t0))
#
#         print(info['final_seed'])
#         print(info['final_radius'] * 2, info['final_radius_id'])
#         print(info[info['final_radius']][info['attempt_id']])
#         fig, ax = plt.subplots(1, 1)
#         ax.set_aspect('equal')
#         for x_, y_, in zip(x, y):
#             c = plt.Circle((x_, y_), min_sep / 2, fill=True, color='k',
#                            alpha=0.5, lw=0)
#             ax.add_artist(c)
#         r_max = info['final_radius']
#         c = plt.Circle((0, 0), r_max, fill=False, color='r', alpha=0.5)
#         ax.add_artist(c)
#         c = plt.Circle((0, 0), r_min, fill=False, color='r', alpha=0.5)
#         ax.add_artist(c)
#         ax.set_xlim(-r_max * 1.05, r_max * 1.05)
#         ax.set_ylim(-r_max * 1.05, r_max * 1.05)
#         plt.show()

# class Test4(unittest.TestCase):
#     @staticmethod
#     def runTest():
#         # num_clusters = 50
#         # n = 256
#         # r_min, r_max = 0.0, 40.0
#         # min_sep = 1.5
#
#         num_clusters = 72*10
#         n = 2
#         r_min, r_max = 0.0, 150.0/2
#         min_sep = 35.0
#         trial_timeout = 0.5
#         num_trials = 5
#         seed = None
#         verbose = False
#         t0 = time.time()
#         x, y, info = generate_clusters(num_clusters, n, r_max, min_sep,
#                                        trial_timeout,
#                                        num_trials, seed, r_min,
#                                        verbose)
#         print('time taken = %0.2f s' % (time.time() - t0))
#
#         fig, ax = plt.subplots(1, 1)
#         ax.set_aspect('equal')
#         for i in range(num_clusters):
#             for x_, y_, in zip(x[i], y[i]):
#                 c = plt.Circle((x_, y_), min_sep / 2, fill=True, color='k',
#                                alpha=0.01, lw=0)
#                 ax.add_artist(c)
#         ax.plot(x, y, 'r+')
#         c = plt.Circle((0, 0), min_sep/2, fill=False, color='b',
#                        alpha=1.0, lw=2.0)
#         ax.add_artist(c)
#
#         c = plt.Circle((0, 0), r_max, fill=False, color='b',
#                        alpha=1.0, lw=2.0)
#         ax.add_artist(c)
#         c = plt.Circle((0, 0), r_min, fill=False, color='b',
#                        alpha=1.0, lw=2.0)
#         ax.add_artist(c)
#         ax.set_xlim(-r_max * 1.05, r_max * 1.05)
#         ax.set_ylim(-r_max * 1.05, r_max * 1.05)
#         plt.show()


class Test5(unittest.TestCase):
    @staticmethod
    def runTest():

        b = 0.5
        num_arms = 3
        clusters_per_arm = 8
        stations_per_cluster = 3
        cluster_diameter_m = 200.0
        station_diameter_m = 35.0
        trial_timeout = 1.0
        tries_per_cluster = 5
        r_min, r_max = 500.0, 5000.0
        verbose = False
        seed = None

        t0 = time.time()
        layout = inner_arms_clusters(b, num_arms, clusters_per_arm,
                                     stations_per_cluster, cluster_diameter_m,
                                     station_diameter_m, r_min, r_max,
                                     trial_timeout, tries_per_cluster,
                                     seed, verbose)
        print('time taken = %0.2f s' % (time.time() - t0))

        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
        ax.set_aspect('equal')
        for x_, y_ in zip(layout['x'], layout['y']):
            c = plt.Circle((x_, y_), station_diameter_m/2, fill=True,
                           color='k', alpha=0.6)
            ax.add_artist(c)
        for x_, y_ in zip(layout['cx'], layout['cy']):
            c = plt.Circle((x_, y_), cluster_diameter_m/2, fill=False,
                           color='b')
            ax.add_artist(c)
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        c = plt.Circle((0, 0), r_min, fill=False, color='r')
        ax.add_artist(c)
        c = plt.Circle((0, 0), r_max, fill=False, color='r')
        ax.add_artist(c)
        ax.grid()
        plt.show()


if __name__ == '__main__':
    unittest.main()
