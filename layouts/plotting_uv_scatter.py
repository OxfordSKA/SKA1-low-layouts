# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as pyplot
from os.path import join
import os


def uv_plot(uu_v4d, vv_v4d, uu_v4o1, vv_v4o1, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4d, vv_v4d, 'k.', alpha=0.1, ms=2.0)
    ax.plot(-uu_v4d, -vv_v4d, 'k.', alpha=0.1, ms=2.0)
    ax.set_title('v4d')
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    pyplot.savefig(join(out_dir, 'scatter_v4d_300m.png'))
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    pyplot.savefig(join(out_dir, 'scatter_v4d_1000m.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'scatter_v4d_3000m.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'scatter_v4d_5000m.png'))
    pyplot.close(fig)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(uu_v4o1, vv_v4o1, 'k.', alpha=0.1, ms=2.0)
    ax.plot(-uu_v4o1, -vv_v4o1, 'k.', alpha=0.1, ms=2.0)
    ax.set_title('v4o1')
    ax.set_xlabel('uu [m]')
    ax.set_ylabel('vv [m]')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    pyplot.savefig(join(out_dir, 'scatter_v4o1_300m.png'))
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    pyplot.savefig(join(out_dir, 'scatter_v4o1_1000m.png'))
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)
    pyplot.savefig(join(out_dir, 'scatter_v4o1_3000m.png'))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    pyplot.savefig(join(out_dir, 'scatter_v4o1_5000m.png'))
    pyplot.close(fig)
