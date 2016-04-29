# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as pyplot
from os.path import join
import os


def plot_layouts(v4d, v4o1, station_radius_m, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(v4d.shape[0]):
        circle = pyplot.Circle((v4d[i, 0], v4d[i, 1]), station_radius_m,
                               color='b', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(True)
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    pyplot.savefig(join(out_dir, 'v4d_layout_2km.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(v4o1.shape[0]):
        circle = pyplot.Circle((v4o1[i, 0], v4o1[i, 1]), station_radius_m,
                               color='b', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(True)
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    pyplot.savefig(join(out_dir, 'v4o1_layout_2km.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(v4d.shape[0]):
        circle = pyplot.Circle((v4d[i, 0], v4d[i, 1]), station_radius_m,
                               color='k', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    for i in range(v4o1.shape[0]):
        circle = pyplot.Circle((v4o1[i, 0], v4o1[i, 1]), station_radius_m,
                               color='r', fill=True, alpha=0.5,
                               linewidth=0.0)
        ax.add_artist(circle)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('v4d = black, v4o1 = red')
    ax.grid(True)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    pyplot.savefig(join(out_dir, 'v4d_v4o1_layout_1km.png'))
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    pyplot.savefig(join(out_dir, 'v4d_v4o1_layout_2km.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'v4d_v4o1_layout_5km.png'))
    pyplot.close(fig)
