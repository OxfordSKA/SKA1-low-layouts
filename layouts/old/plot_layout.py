# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib.pyplot as pyplot
import numpy


def main():
    st_file_name = 'v7ska1lowN1v2arev3R.enu.564x4.txt'
    ss_file_name = 'v7ska1lowN1v2rev3R.enu.94x4.fixed.txt'

    st_layout = numpy.loadtxt(st_file_name)
    ss_layout = numpy.loadtxt(ss_file_name)
    ss_radius = 90.0 / 2.0  # m
    st_radius = 35.0 / 2.0  # m

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(ss_layout.shape[0]):
        xss = ss_layout[i, 1]
        yss = ss_layout[i, 2]
        circle = pyplot.Circle((xss, yss), ss_radius, color='b',
                               fill=True, alpha=0.2)
        ax.add_artist(circle)

    for i in range(st_layout.shape[0]):
        xst = st_layout[i, 1]
        yst = st_layout[i, 2]
        circle = pyplot.Circle((xst, yst), st_radius, color='r',
                               fill=True, alpha=0.2)
        ax.add_artist(circle)

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.grid()

    major_ticks = numpy.arange(-1000, 1200, 200)
    minor_ticks = numpy.arange(-1000, 1100, 100)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1.0)

    pyplot.show()


if __name__ == '__main__':
    main()
